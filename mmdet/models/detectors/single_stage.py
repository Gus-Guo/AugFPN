import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
import torch

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if 'use_consistent_supervision' in self.train_cfg:
            self.use_consistent_supervision = self.train_cfg.use_consistent_supervision
        else:
            self.use_consistent_supervision = False
        if self.use_consistent_supervision:

            bbox_roi_extractor=dict(
                        type='AuxAllLevelRoIExtractor',
                        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
                        out_channels=256,
                        featmap_strides=[8,16,32]) # only apply to feature map belonging to FPN
            
            self.auxiliary_bbox_roi_extractor = builder.build_roi_extractor(
            bbox_roi_extractor)
            bbox_head=dict(
                    type='AuxiliarySharedFCBBoxHead',
                    num_fcs=2,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=81,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                    reg_class_agnostic=False)
            
            self.auxiliary_bbox_head = builder.build_head(bbox_head)            

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
        if self.use_consistent_supervision:  
            self.auxiliary_bbox_head.init_weights()
            
    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            if self.use_consistent_supervision:
                x, y = self.neck(x)
                return x,y
            else:
                x = self.neck(x)
                return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        if self.use_consistent_supervision:
            x, y = self.extract_feat(img)
            gt_bboxes_auxiliary = [gt.clone() for gt in gt_bboxes]
            gt_labels_auxiliary = [label.clone() for label in gt_labels]
        else:
            x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if self.use_consistent_supervision:
            proposal_cfg = self.train_cfg.auxiliary.proposal
            proposal_inputs = outs + (img_metas, proposal_cfg)
            proposal_list = self.bbox_head.get_bboxes_auxiliary(*proposal_inputs)
        
            bbox_assigner = build_assigner(self.train_cfg.auxiliary.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.auxiliary.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):

                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes_auxiliary[i], gt_bboxes_ignore[i],
                    gt_labels_auxiliary[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes_auxiliary[i],
                    gt_labels_auxiliary[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats_raw = self.auxiliary_bbox_roi_extractor(y[:self.auxiliary_bbox_roi_extractor.num_inputs], rois)
            cls_score_auxiliary, bbox_pred_auxiliary = self.auxiliary_bbox_head(bbox_feats_raw)

            bbox_targets = self.auxiliary_bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.auxiliary.rcnn)
        
            loss_bbox_auxiliary = self.auxiliary_bbox_head.loss(cls_score_auxiliary, bbox_pred_auxiliary,
                                            *bbox_targets, alpha=0.25, num_level=3)   
            losses.update(loss_bbox_auxiliary) 

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        if self.use_consistent_supervision:
            x, y = self.extract_feat(img)
        else:
            x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

