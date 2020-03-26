from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet import ops
from ..registry import ROI_EXTRACTORS
from mmcv.cnn import xavier_init
import numpy as np

@ROI_EXTRACTORS.register_module
class PANetAllLevelRoIExtractor(nn.Module):
    """Extract RoI features from all level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(PANetAllLevelRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

        #self.spatial_attention_conv=nn.Sequential(nn.Conv2d(out_channels*len(featmap_strides), out_channels, 1), nn.ReLU(), nn.Conv2d(out_channels,len(featmap_strides),3, padding=1))
        self.fcs = nn.ModuleList()
        for i in range(len(featmap_strides)):
            self.fcs.append(nn.Sequential(nn.Linear(self.roi_layers[0].out_size * self.roi_layers[0].out_size * out_channels, 1024), nn.ReLU()))
            

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        for m in self.fcs.modules():
            for n in m.modules():
                if isinstance(n, nn.Linear):
                    xavier_init(n, distribution='uniform')

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers


    def forward(self, feats, rois):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)



        roi_feats = torch.cuda.FloatTensor(rois.size()[0], self.out_channels,
                                           out_size, out_size).fill_(0)
 
        roi_feats_list = []
        for i in range(num_levels):
            roi_feats_list.append(self.fcs[i](self.roi_layers[i](feats[i], rois).view(rois.size()[0],-1)))
        concat_roi_feats = torch.stack(roi_feats_list, dim=2)
        new_roi_feats = torch.max(concat_roi_feats, dim=2,keepdim=False)[0]
        return new_roi_feats
