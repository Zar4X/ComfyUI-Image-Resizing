import math


class CalculateUpscaleFactor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "b": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "target_size": ("INT", {"default": 768, "min": 1, "max": 10000}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calculate"
    CATEGORY = "Image Resizing/Utils"

    def calculate(self, a, b, target_size):
        sum_sides = a + b
        if sum_sides >= target_size:
            return (0,)

        upscale_factor = target_size / sum_sides

        return (round(upscale_factor, 2),)
