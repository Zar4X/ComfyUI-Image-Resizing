from .nodes.image_crop_by_percentage import ImageCropByPercentage
from .nodes.mask_crop_by_percentage import MaskCropByPercentage
from .nodes.extend_canvas_by_percentage import ExtendCanvasByPercentage
from .nodes.resize_to_multiple import ResizeToMultiple
from .nodes.image_resolution_extractor import ImageResolutionExtractor
from .nodes.image_aspect_ratio_extractor import ImageAspectRatioExtractor
from .nodes.smart_image_resize import SmartImageResize
from .nodes.remove_padding import RemovePadding
from .nodes.apply_mask_to_image import ApplyMaskToImage

NODE_CLASS_MAPPINGS = {
    "ImageCropByPercentage": ImageCropByPercentage,
    "MaskCropByPercentage": MaskCropByPercentage,
    "ExtendCanvasByPercentage": ExtendCanvasByPercentage,
    "ResizeToMultiple": ResizeToMultiple,
    "ImageResolutionExtractor": ImageResolutionExtractor,
    "ImageAspectRatioExtractor": ImageAspectRatioExtractor,
    "SmartImageResize": SmartImageResize,
    "RemovePadding": RemovePadding,
    "ApplyMaskToImage": ApplyMaskToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByPercentage": "Image Crop By Percentage",
    "MaskCropByPercentage": "Mask Crop By Percentage",
    "ExtendCanvasByPercentage": "Extend Canvas By Percentage",
    "ResizeToMultiple": "Resize To Multiple",
    "ImageResolutionExtractor": "Image Resolution Extractor",
    "ImageAspectRatioExtractor": "Image Aspect Ratio Extractor",
    "SmartImageResize": "Smart Image Resize",
    "RemovePadding": "Remove Padding",
    "ApplyMaskToImage": "Apply Mask To Image",
}
