from .binary_head import BinaryHead
from .caption_head import CaptionHead

# from .inst_head import InstHead
from .kd_head import KDHeadTemplate
from .text_seg_head import TextSegHead

__all__ = {
    "TextSegHead": TextSegHead,
    "BinaryHead": BinaryHead,
    "CaptionHead": CaptionHead,
    # "InstHead": InstHead,
    "KDHeadTemplate": KDHeadTemplate,
}
