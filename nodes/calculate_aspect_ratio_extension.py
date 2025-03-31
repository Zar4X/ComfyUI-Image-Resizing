import torch
from PIL import Image


class CalculateAspectRatioExtension:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_ratio": (["9:16", "16:9", "1:1", "Custom"],),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "custom_height": ("INT", {"default": 1, "min": 1, "max": 10000}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width_extension_per_side", "height_extension_per_side")
    FUNCTION = "calculate_extension"
    CATEGORY = "Image/Analysis"

    def calculate_extension(self, image, target_ratio, custom_width=1, custom_height=1):
        height, width = image.shape[1:3]

        def calc_extension(target_w, target_h):
            current_ratio = width / height
            target_ratio = target_w / target_h

            if current_ratio < target_ratio:
                new_width = int(height * target_ratio)
                return (new_width - width) // 2, 0
            elif current_ratio > target_ratio:
                new_height = int(width / target_ratio)
                return 0, (new_height - height) // 2
            else:
                return 0, 0

        if target_ratio == "9:16":
            w_extension, h_extension = calc_extension(9, 16)
        elif target_ratio == "16:9":
            w_extension, h_extension = calc_extension(16, 9)
        elif target_ratio == "1:1":
            w_extension, h_extension = calc_extension(1, 1)
        else:  # Custom
            w_extension, h_extension = calc_extension(custom_width, custom_height)

        return (w_extension, h_extension)
