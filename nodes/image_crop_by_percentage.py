import torch
from .ImgMods import tensor2pil, pil2tensor


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

    @staticmethod
    def _clamp_pair_to_size(start_crop, end_crop, size):
        if size <= 1:
            return 0, 0
        start_crop = max(0, min(start_crop, size - 1))
        end_crop = max(0, min(end_crop, size - 1 - start_crop))
        return start_crop, end_crop

    @staticmethod
    def _value_to_pixels(value, size, use_pixel):
        if use_pixel:
            return max(0, int(value))
        clamped_percent = max(0.0, min(float(value), 100.0))
        # Percentage mode: each side (left/right/top/bottom) is defined over the half axis.
        # Example: left=100 means "crop from left edge to center" => half of full width.
        return int(clamped_percent / 100.0 * (size / 2.0))

    @staticmethod
    def _normalize_crop_inputs(use_pixel, top, bottom, left, right):
        if use_pixel:
            return max(0, top), max(0, bottom), max(0, left), max(0, right), False
        n_top = max(0, min(100, top))
        n_bottom = max(0, min(100, bottom))
        n_left = max(0, min(100, left))
        n_right = max(0, min(100, right))
        was_clamped = (n_top, n_bottom, n_left, n_right) != (top, bottom, left, right)
        return n_top, n_bottom, n_left, n_right, was_clamped

    @staticmethod
    def _ensure_non_empty_crop_box(x1, y1, x2, y2, img_w, img_h):
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        return x1, y1, x2, y2

    def image_crop(self, image, use_pixel, top, bottom, left, right):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 0)
        top, bottom, left, right, was_clamped = self._normalize_crop_inputs(
            use_pixel, top, bottom, left, right
        )

        # Convert tensor to PIL image for easier manipulation
        pil_images = [tensor2pil(torch.unsqueeze(img, 0)) for img in image]
        cropped_images = []
        crop_coords_x = []
        crop_coords_y = []

        for img_index, pil_img in enumerate(pil_images):
            img_w, img_h = pil_img.size
            print(f"Original Image at index {img_index} size: {img_w}x{img_h}")

            crop_left = self._value_to_pixels(left, img_w, use_pixel)
            crop_right = self._value_to_pixels(right, img_w, use_pixel)
            crop_top = self._value_to_pixels(top, img_h, use_pixel)
            crop_bottom = self._value_to_pixels(bottom, img_h, use_pixel)

            crop_left, crop_right = self._clamp_pair_to_size(crop_left, crop_right, img_w)
            crop_top, crop_bottom = self._clamp_pair_to_size(crop_top, crop_bottom, img_h)

            if was_clamped:
                print(
                    "Percentage crop values are clamped to 0-100. "
                    f"Received top:{top} bottom:{bottom} left:{left} right:{right}."
                )

            # Calculate crop box
            x1 = crop_left
            y1 = crop_top
            x2 = img_w - crop_right
            y2 = img_h - crop_bottom

            # Ensure valid crop box and prevent zero-sized crops
            x1, y1, x2, y2 = self._ensure_non_empty_crop_box(
                x1, y1, x2, y2, img_w, img_h
            )

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
