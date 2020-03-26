from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead, SharedFCBBoxHead
from .convfc_bbox_head_auxiliary import AuxiliaryBBoxHead, AuxiliaryConvFCBBoxHead, AuxiliarySharedFCBBoxHead
__all__ = ['BBoxHead', 'ConvFCBBoxHead', 'SharedFCBBoxHead', 'AuxiliaryBBoxHead', 'AuxiliaryConvFCBBoxHead','AuxiliarySharedFCBBoxHead']
