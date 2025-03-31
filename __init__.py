from .nodes.image_crop_by_percentage import ImageCropByPercentage
from .nodes.mask_crop_by_percentage import MaskCropByPercentage
from .nodes.extend_canvas_by_percentage import ExtendCanvasByPercentage
from .nodes.calculate_aspect_ratio_extension import CalculateAspectRatioExtension
from .nodes.calculate_upscale_factor import CalculateUpscaleFactor
from .nodes.calculate_upscale_rounds import CalculateUpscaleRounds

NODE_CLASS_MAPPINGS = {
    "ImageCropByPercentage": ImageCropByPercentage,
    "MaskCropByPercentage": MaskCropByPercentage,
    "ExtendCanvasByPercentage": ExtendCanvasByPercentage,
    "CalculateAspectRatioExtension": CalculateAspectRatioExtension,
    "CalculateUpscaleFactor": CalculateUpscaleFactor,
    "CalculateUpscaleRounds": CalculateUpscaleRounds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByPercentage": "Image Crop By Percentage",
    "MaskCropByPercentage": "Mask Crop By Percentage",
    "ExtendCanvasByPercentage": "Extend Canvas By Percentage",
    "CalculateAspectRatioExtension": "Calculate Aspect Ratio Extension",
    "CalculateUpscaleFactor": "Calculate Upscale Factor",
    "CalculateUpscaleRounds": "Calculate Upscale Rounds",
}
