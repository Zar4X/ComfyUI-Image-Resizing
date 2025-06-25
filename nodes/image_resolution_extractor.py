import torch
import numpy as np
from PIL import Image
from .ImgMods import tensor2pil


class ImageResolutionExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (["shortest side", "longest side", "width", "height"], {"default": "width"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("pixels",)
    FUNCTION = "extract_resolution"
    CATEGORY = "Image Resizing/Resolution"
    DESCRIPTION = "Extracts specific resolution values from input images. Options: shortest side, longest side, width, or height."

    def extract_resolution(self, image, resolution):
        """
        Extracts specific resolution values from input images.
        
        Args:
            image: Input image tensor (can be single image or batch)
            resolution: Which dimension to extract ("shortest side", "longest side", "width", or "height")
            
        Returns:
            Integer value representing the selected dimension
        """
        # Validate input
        if len(image) == 0:
            raise ValueError("Input image tensor is empty")
            
        # Convert tensor to PIL image for easier manipulation
        pil_images = [tensor2pil(img) for img in image]
        
        # Get dimensions of the first image (assuming all images in batch have same dimensions)
        width, height = pil_images[0].size
        
        # Validate dimensions
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")
        
        # Extract the requested resolution
        if resolution == "shortest side":
            result = min(width, height)
        elif resolution == "longest side":
            result = max(width, height)
        elif resolution == "width":
            result = width
        elif resolution == "height":
            result = height
        else:
            raise ValueError(f"Unknown resolution option: {resolution}")
            
        print(f"Image resolution: {width}x{height}, extracted {resolution}: {result}")
        
        return (result,) 