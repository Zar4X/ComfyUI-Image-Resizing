import torch
import numpy as np
from PIL import Image
from .ImgMods import tensor2pil, pil2tensor, num_round_up_to_multiple


class ImageCropByPercentage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_pixel": ("BOOLEAN", {"default": False}),
                "image": ("IMAGE",),
                "width": ("INT", {"default": 100, "min": 0, "max": 9999, "step": 1}),
                "height": ("INT", {"default": 100, "min": 0, "max": 9999, "step": 1}),
                "position": (
                    [
                        "top-left",
                        "top-center",
                        "top-right",
                        "right-center",
                        "bottom-right",
                        "bottom-center",
                        "bottom-left",
                        "left-center",
                        "center",
                    ],
                ),
                "x_offset": (
                    "INT",
                    {"default": 0, "min": -9999, "max": 9999, "step": 1},
                ),
                "y_offset": (
                    "INT",
                    {"default": 0, "min": -9999, "max": 9999, "step": 1},
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "Cropped Image",
        "x",
        "y",
    )
    FUNCTION = "image_crop"
    CATEGORY = "Image Resizing/Crop Image"

    def image_crop(self, image, use_pixel, width, height, position, x_offset, y_offset):
        # Convert tensor to PIL image for easier manipulation
        pil_images = [tensor2pil(img) for img in image]
        cropped_images = []
        crop_coords_x = []
        crop_coords_y = []

        for img_index, pil_img in enumerate(pil_images):
            ow, oh = pil_img.size
            print(f"Original Image at index {img_index} size: {ow}x{oh}")

            if use_pixel:
                new_width = min(ow, width)
                new_height = min(oh, height)
                x_offset_pixel = min(ow, max(-ow, x_offset))
                y_offset_pixel = min(oh, max(-oh, y_offset))
            else:
                new_width = min(ow, round(ow * (width / 100.0)))
                new_height = min(oh, round(oh * (height / 100.0)))
                x_offset_pixel = round(ow * (x_offset / 100.0))
                y_offset_pixel = round(oh * (y_offset / 100.0))

            print(
                f"Computed for cropping - width: {new_width}, height: {new_height}, x_offset: {x_offset_pixel}, y_offset: {y_offset_pixel}"
            )

            x = 0
            y = 0

            if "center" in position:
                x = round((ow - new_width) / 2)
                y = round((oh - new_height) / 2)
            if "top" in position:
                y = 0
            if "bottom" in position:
                y = oh - new_height
            if "left" in position:
                x = 0
            if "right" in position:
                x = ow - new_width

            x += x_offset_pixel
            y += y_offset_pixel

            x2 = x + new_width
            y2 = y + new_height

            # Ensure x and y are within bounds
            if x2 > ow:
                x2 = ow
            if x < 0:
                x = 0
            if y2 > oh:
                y2 = oh
            if y < 0:
                y = 0

            # Crop the image using computed coordinates
            cropped_img = pil_img.crop((x, y, x2, y2))

            # Convert cropped PIL image back to tensor
            cropped_images.append(pil2tensor(cropped_img))
            crop_coords_x.append(x)
            crop_coords_y.append(y)

            print(
                f"Cropped Image at index {img_index} from ({x}, {y}) to ({x2}, {y2}), resulting size: {cropped_img.size}"
            )

        # Combine all cropped images into a single tensor
        cropped_images_tensor = torch.cat(cropped_images, dim=0)
        return (
            cropped_images_tensor,
            torch.tensor(crop_coords_x),
            torch.tensor(crop_coords_y),
        )
