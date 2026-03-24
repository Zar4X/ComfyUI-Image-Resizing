import torch
import numpy as np
from PIL import Image
from .ImgMods import tensor2pil, pil2tensor, num_round_up_to_multiple


class ImageCropByPercentage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_pixel": ("BOOLEAN", {"default": False}),
                "top": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "left": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
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

    def image_crop(self, image, use_pixel, top, bottom, left, right):
        # Convert tensor to PIL image for easier manipulation
        pil_images = [tensor2pil(img) for img in image]
        cropped_images = []
        crop_coords_x = []
        crop_coords_y = []

        for img_index, pil_img in enumerate(pil_images):
            img_w, img_h = pil_img.size
            print(f"Original Image at index {img_index} size: {img_w}x{img_h}")

            if use_pixel:
                # Use pixel values directly
                crop_left = min(left, img_w)
                crop_right = min(right, img_w)
                crop_top = min(top, img_h)
                crop_bottom = min(bottom, img_h)
            else:
                # Convert percentage to pixels
                crop_left = int(left / 100 * img_w)
                crop_right = int(right / 100 * img_w)
                crop_top = int(top / 100 * img_h)
                crop_bottom = int(bottom / 100 * img_h)

            # Calculate crop box
            x1 = crop_left
            y1 = crop_top
            x2 = img_w - crop_right
            y2 = img_h - crop_bottom

            # Ensure valid crop box
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(x1, min(x2, img_w))
            y2 = max(y1, min(y2, img_h))

            print(
                f"Crop mode: {'pixel' if use_pixel else 'percentage'} - top:{top} bottom:{bottom} left:{left} right:{right}"
            )
            print(
                f"Cropped Image at index {img_index} from ({x1}, {y1}) to ({x2}, {y2}), resulting size: {x2-x1}x{y2-y1}"
            )

            # Crop the image
            cropped_img = pil_img.crop((x1, y1, x2, y2))

            # Convert cropped PIL image back to tensor
            cropped_images.append(pil2tensor(cropped_img))
            crop_coords_x.append(x1)
            crop_coords_y.append(y1)

        # Combine all cropped images into a single tensor
        cropped_images_tensor = torch.cat(cropped_images, dim=0)
        return (
            cropped_images_tensor,
            torch.tensor(crop_coords_x),
            torch.tensor(crop_coords_y),
        )
