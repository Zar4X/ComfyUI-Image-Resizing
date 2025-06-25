import torch
import numpy as np
from PIL import Image
from .ImgMods import tensor2pil, pil2tensor


class ResizeToMultiple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "multiple": ("INT", {"default": 32, "min": 1, "max": 1024, "step": 1}),
                "method": (["stretch", "crop"], {"default": "crop"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize_to_multiple"
    CATEGORY = "Image Resizing/Resize"

    def resize_to_multiple(self, image, multiple, method):
        """
        Resize images to be multiples of a specified number.
        
        Args:
            image: Input image tensor (can be single image or batch)
            multiple: The number that dimensions should be multiples of
            method: "stretch" to resize, "crop" to crop from center
        
        Returns:
            Resized image tensor, calculated width, calculated height
        """
        # Convert tensor to PIL images for easier manipulation
        pil_images = [tensor2pil(img) for img in image]
        processed_images = []
        final_width = 0
        final_height = 0

        for img_index, pil_img in enumerate(pil_images):
            original_width, original_height = pil_img.size
            print(f"Original Image at index {img_index} size: {original_width}x{original_height}")

            # Calculate target dimensions (rounded down to nearest multiple)
            target_width = (original_width // multiple) * multiple
            target_height = (original_height // multiple) * multiple
            
            # Ensure minimum size of at least 1 * multiple
            if target_width == 0:
                target_width = multiple
            if target_height == 0:
                target_height = multiple

            # Store the final dimensions (use first image dimensions for all)
            if img_index == 0:
                final_width = target_width
                final_height = target_height

            print(f"Target size: {target_width}x{target_height} (multiple of {multiple})")

            if method == "stretch":
                # Stretch/resize the image to target dimensions
                resized_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                processed_images.append(pil2tensor(resized_img))
                print(f"Stretched image to {target_width}x{target_height}")
                
            elif method == "crop":
                # Crop from center to target dimensions
                # Calculate crop coordinates (center crop)
                left = (original_width - target_width) // 2
                top = (original_height - target_height) // 2
                right = left + target_width
                bottom = top + target_height
                
                # Ensure coordinates are within bounds
                left = max(0, left)
                top = max(0, top)
                right = min(original_width, right)
                bottom = min(original_height, bottom)
                
                # Crop the image
                cropped_img = pil_img.crop((left, top, right, bottom))
                processed_images.append(pil2tensor(cropped_img))
                print(f"Cropped image from center to {target_width}x{target_height}")

        # Combine all processed images into a single tensor
        processed_images_tensor = torch.cat(processed_images, dim=0)
        
        print(f"Final output dimensions: {final_width}x{final_height}")
        return (processed_images_tensor, final_width, final_height) 