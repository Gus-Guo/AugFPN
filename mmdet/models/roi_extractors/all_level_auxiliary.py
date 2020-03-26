from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class AuxAllLevelRoIExtractor(nn.Module):
    """Extract RoI features from all level feature map and concat them for consistent supervison.

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
        super(AuxAllLevelRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

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

        roi_feats = torch.cuda.FloatTensor(rois.size()[0]*num_levels, self.out_channels,
                                           out_size, out_size).fill_(0)

        for i in range(num_levels):
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            roi_feats[i::num_levels] = roi_feats_t
        return roi_feats

