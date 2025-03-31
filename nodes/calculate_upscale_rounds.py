import math


class CalculateUpscaleRounds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "b": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "target_size": ("INT", {"default": 768, "min": 1, "max": 10000}),
                "upscale_factor": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.01, "max": 10.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "calculate"
    CATEGORY = "Image Resizing/Utils"

    def calculate(self, a, b, target_size, upscale_factor):
        sum_sides = a + b
        if sum_sides >= target_size:
            return (0,)

        upscale_rounds = 0
        current_size = sum_sides

        while current_size < target_size:
            current_size *= upscale_factor
            upscale_rounds += 1

        return (upscale_rounds,)
