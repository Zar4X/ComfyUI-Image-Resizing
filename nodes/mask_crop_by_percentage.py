import torch
import cv2
import numpy as np
from PIL import ImageOps
from .ImgMods import (
    tensor2pil,
    pil2tensor,
    mask2image,
    image2mask,
    gaussian_blur,
    min_bounding_rect,
    max_inscribed_rect,
    mask_area,
    num_round_up_to_multiple,
    draw_rect,
    pil2cv2,
)


class MaskCropByPercentage:
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
        # Example: left=100 means "crop from left edge to center" => half of the region width.
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
    def _ensure_non_empty_crop_box(x1, y1, x2, y2, max_w, max_h):
        x1 = max(0, min(x1, max_w - 1))
        y1 = max(0, min(y1, max_h - 1))
        x2 = max(x1 + 1, min(x2, max_w))
        y2 = max(y1 + 1, min(y2, max_h))
        return x1, y1, x2, y2

    @staticmethod
    def find_all_regions(image):
        """Find all bounding rectangles for all connected components in the mask."""
        cv2_image = pil2cv2(image)
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very small regions (noise)
            if w * h > 100:  # Minimum area threshold
                rects.append((x, y, w, h))
        return rects

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = [
            "mask_area",
            "min_bounding_rect",
            "max_inscribed_rect",
            "crop_each_region",
        ]
        multiple_list = ["8", "16", "32", "64", "128", "256", "512", "None"]
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "use_pixel": ("BOOLEAN", {"default": False}),
                "detect": (detect_mode,),
                "top": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999, "step": 1},
                ),
                "bottom": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999, "step": 1},
                ),
                "left": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999, "step": 1},
                ),
                "right": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999, "step": 1},
                ),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "crop_box": ("BOX",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "BOX",
        "IMAGE",
    )
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_box", "box_preview")
    FUNCTION = "mask_crop_by_percentage"
    CATEGORY = "Image Resizing/Resizing"

    def mask_crop_by_percentage(
        self,
        image,
        mask,
        invert_mask,
        use_pixel,
        detect,
        top,
        bottom,
        left,
        right,
        round_to_multiple,
        crop_box=None,
    ):
        top, bottom, left, right, was_clamped = self._normalize_crop_inputs(
            use_pixel, top, bottom, left, right
        )
        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []

        if image.dim() == 3:
            image = torch.unsqueeze(image, 0)

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        if mask.shape[0] > 1:
            mask = torch.unsqueeze(mask[0], 0)
        if invert_mask:
            mask = 1 - mask
        l_masks.append(tensor2pil(torch.unsqueeze(mask, 0)).convert("L"))

        _mask = mask2image(mask)
        preview_image = tensor2pil(mask).convert("RGB")
        all_rects = []  # Initialize all_rects outside the if block
        if crop_box is None:
            bluredmask = gaussian_blur(_mask, 20).convert("L")
            x = 0
            y = 0
            width = 0
            height = 0

            if detect == "crop_each_region":
                all_rects = self.find_all_regions(bluredmask)
                if all_rects:
                    # For crop_each_region mode, we don't need a union crop_box
                    # Just use the first rect for fallback (won't be used for cropping)
                    x, y, w, h = all_rects[0]
                else:
                    # Fallback if no regions found
                    (x, y, w, h) = mask_area(_mask)
            elif detect == "min_bounding_rect":
                (x, y, w, h) = min_bounding_rect(bluredmask)
            elif detect == "max_inscribed_rect":
                (x, y, w, h) = max_inscribed_rect(bluredmask)
            else:
                (x, y, w, h) = mask_area(_mask)

            canvas_width, canvas_height = (
                tensor2pil(torch.unsqueeze(image[0], 0)).convert("RGB").size
            )
            if w <= 0 or h <= 0:
                x, y, w, h = 0, 0, canvas_width, canvas_height

            crop_top_pixels = self._value_to_pixels(top, h, use_pixel)
            crop_bottom_pixels = self._value_to_pixels(bottom, h, use_pixel)
            crop_left_pixels = self._value_to_pixels(left, w, use_pixel)
            crop_right_pixels = self._value_to_pixels(right, w, use_pixel)

            crop_left_pixels, crop_right_pixels = self._clamp_pair_to_size(
                crop_left_pixels, crop_right_pixels, w
            )
            crop_top_pixels, crop_bottom_pixels = self._clamp_pair_to_size(
                crop_top_pixels, crop_bottom_pixels, h
            )
            if was_clamped:
                print(
                    "Percentage crop values are clamped to 0-100. "
                    f"Received top:{top} bottom:{bottom} left:{left} right:{right}."
                )

            x1 = max(x + crop_left_pixels, 0)
            y1 = max(y + crop_top_pixels, 0)
            x2 = min(x + w - crop_right_pixels, canvas_width)
            y2 = min(y + h - crop_bottom_pixels, canvas_height)

            if round_to_multiple != "None":
                multiple = int(round_to_multiple)
                width = num_round_up_to_multiple(x2 - x1, multiple)
                height = num_round_up_to_multiple(y2 - y1, multiple)
                x1 = x1 - (width - (x2 - x1)) // 2
                y1 = y1 - (height - (y2 - y1)) // 2
                x2 = x1 + width
                y2 = y1 + height

            x1, y1, x2, y2 = self._ensure_non_empty_crop_box(
                x1, y1, x2, y2, canvas_width, canvas_height
            )
            crop_box = (x1, y1, x2, y2)

            # Draw red rectangles for all detected regions (with crop applied)
            if detect == "crop_each_region" and all_rects:
                canvas_width, canvas_height = (
                    tensor2pil(torch.unsqueeze(image[0], 0)).convert("RGB").size
                )
                for rect_x, rect_y, rect_w, rect_h in all_rects:
                    # Calculate available space from rectangle to image edges
                    space_left = rect_x
                    space_right = canvas_width - (rect_x + rect_w)
                    space_top = rect_y
                    space_bottom = canvas_height - (rect_y + rect_h)

                    # Calculate actual crop pixels for this rectangle
                    actual_crop_left = min(
                        self._value_to_pixels(left, rect_w, use_pixel), space_left
                    )
                    actual_crop_right = min(
                        self._value_to_pixels(right, rect_w, use_pixel), space_right
                    )
                    actual_crop_top = min(
                        self._value_to_pixels(top, rect_h, use_pixel), space_top
                    )
                    actual_crop_bottom = min(
                        self._value_to_pixels(bottom, rect_h, use_pixel), space_bottom
                    )

                    # Calculate cropped rectangle for drawing
                    cropped_x = max(0, rect_x + actual_crop_left)
                    cropped_y = max(0, rect_y + actual_crop_top)
                    cropped_x2 = min(canvas_width, rect_x + rect_w - actual_crop_right)
                    cropped_y2 = min(
                        canvas_height, rect_y + rect_h - actual_crop_bottom
                    )
                    cropped_w = max(0, cropped_x2 - cropped_x)
                    cropped_h = max(0, cropped_y2 - cropped_y)

                    preview_image = draw_rect(
                        preview_image,
                        cropped_x,
                        cropped_y,
                        cropped_w,
                        cropped_h,
                        line_color="#F00000",
                        line_width=max(1, (cropped_w + cropped_h) // 100),
                    )
            else:
                # Draw single red rectangle for other detection modes
                preview_image = draw_rect(
                    preview_image,
                    x,
                    y,
                    w,
                    h,
                    line_color="#F00000",
                    line_width=(w + h) // 100,
                )
        # Only draw green rectangle for non-crop_each_region modes
        if detect != "crop_each_region":
            preview_image = draw_rect(
                preview_image,
                crop_box[0],
                crop_box[1],
                crop_box[2] - crop_box[0],
                crop_box[3] - crop_box[1],
                line_color="#00F000",
                line_width=(crop_box[2] - crop_box[0] + crop_box[3] - crop_box[1])
                // 200,
            )

        # If crop_each_region mode and rectangles detected, crop each one
        if detect == "crop_each_region" and all_rects and len(all_rects) > 0:
            canvas_width, canvas_height = (
                tensor2pil(torch.unsqueeze(image[0], 0)).convert("RGB").size
            )

            # Calculate crop pixels (will be applied per rectangle with clamping)
            # Note: values are already converted to pixels if use_pixel is True,
            # otherwise they are percentages to be calculated per rectangle

            all_crop_boxes = []
            # Crop each rectangle with reserves applied (clamped to image boundaries)
            for rect_x, rect_y, rect_w, rect_h in all_rects:
                # Calculate available space from rectangle to image edges
                space_left = rect_x
                space_right = canvas_width - (rect_x + rect_w)
                space_top = rect_y
                space_bottom = canvas_height - (rect_y + rect_h)

                # Calculate actual crop pixels for this rectangle
                actual_crop_left = min(
                    self._value_to_pixels(left, rect_w, use_pixel), space_left
                )
                actual_crop_right = min(
                    self._value_to_pixels(right, rect_w, use_pixel), space_right
                )
                actual_crop_top = min(
                    self._value_to_pixels(top, rect_h, use_pixel), space_top
                )
                actual_crop_bottom = min(
                    self._value_to_pixels(bottom, rect_h, use_pixel), space_bottom
                )

                actual_crop_left, actual_crop_right = self._clamp_pair_to_size(
                    actual_crop_left, actual_crop_right, rect_w
                )
                actual_crop_top, actual_crop_bottom = self._clamp_pair_to_size(
                    actual_crop_top, actual_crop_bottom, rect_h
                )

                # Crop rectangle (clamped to boundaries)
                rect_x1 = max(0, rect_x + actual_crop_left)
                rect_y1 = max(0, rect_y + actual_crop_top)
                rect_x2 = min(canvas_width, rect_x + rect_w - actual_crop_right)
                rect_y2 = min(canvas_height, rect_y + rect_h - actual_crop_bottom)

                # Apply rounding if needed (this may expand the crop slightly)
                if round_to_multiple != "None":
                    multiple = int(round_to_multiple)
                    current_width = rect_x2 - rect_x1
                    current_height = rect_y2 - rect_y1
                    width = num_round_up_to_multiple(current_width, multiple)
                    height = num_round_up_to_multiple(current_height, multiple)
                    # Expand evenly from center to maintain the red rectangle roughly centered
                    width_diff = width - current_width
                    height_diff = height - current_height
                    rect_x1 = max(0, rect_x1 - width_diff // 2)
                    rect_y1 = max(0, rect_y1 - height_diff // 2)
                    rect_x2 = min(canvas_width, rect_x1 + width)
                    rect_y2 = min(canvas_height, rect_y1 + height)
                    # Adjust if we hit boundaries
                    if rect_x2 >= canvas_width:
                        rect_x1 = canvas_width - width
                        rect_x2 = canvas_width
                    if rect_y2 >= canvas_height:
                        rect_y1 = canvas_height - height
                        rect_y2 = canvas_height

                rect_x1, rect_y1, rect_x2, rect_y2 = self._ensure_non_empty_crop_box(
                    rect_x1, rect_y1, rect_x2, rect_y2, canvas_width, canvas_height
                )
                individual_crop_box = (rect_x1, rect_y1, rect_x2, rect_y2)
                all_crop_boxes.append(individual_crop_box)

                # Crop each image in the batch for this rectangle
                # This creates one crop per image per rectangle
                for i in range(len(l_images)):
                    _canvas = tensor2pil(l_images[i]).convert("RGB")
                    _mask = l_masks[0]
                    ret_images.append(pil2tensor(_canvas.crop(individual_crop_box)))
                    ret_masks.append(image2mask(_mask.crop(individual_crop_box)))
        else:
            # Original single crop behavior
            for i in range(len(l_images)):
                _canvas = tensor2pil(l_images[i]).convert("RGB")
                _mask = l_masks[0]
                ret_images.append(pil2tensor(_canvas.crop(crop_box)))
                ret_masks.append(image2mask(_mask.crop(crop_box)))
            all_crop_boxes = [list(crop_box)]

        # 确保所有图像都是3通道
        ret_images = [
            img[:, :3, :, :] if img.shape[1] == 4 else img for img in ret_images
        ]

        # If multi-crop mode, pad all images and masks to the same size before concatenating
        if (
            detect == "crop_each_region"
            and all_rects
            and len(all_rects) > 0
            and len(ret_images) > 0
        ):
            # Find maximum dimensions from both images and masks
            max_height = max(
                max(img.shape[2] for img in ret_images),
                max(mask.shape[1] for mask in ret_masks),
            )
            max_width = max(
                max(img.shape[3] for img in ret_images),
                max(mask.shape[2] for mask in ret_masks),
            )

            # Pad all images and masks to max dimensions using PIL for more reliable padding
            padded_images = []
            padded_masks = []
            for img, mask in zip(ret_images, ret_masks):
                # Convert to PIL for reliable padding
                pil_img = tensor2pil(img).convert("RGB")
                pil_mask = tensor2pil(mask).convert("L")

                # Get current dimensions
                img_w, img_h = pil_img.size
                mask_w, mask_h = pil_mask.size

                # Calculate padding needed
                img_pad_w = max_width - img_w
                img_pad_h = max_height - img_h
                mask_pad_w = max_width - mask_w
                mask_pad_h = max_height - mask_h

                # Pad image using ImageOps.expand (left, top, right, bottom)
                if img_pad_w > 0 or img_pad_h > 0:
                    padded_pil_img = ImageOps.expand(
                        pil_img, border=(0, 0, img_pad_w, img_pad_h), fill=(0, 0, 0)
                    )
                else:
                    padded_pil_img = pil_img

                # Pad mask using ImageOps.expand
                if mask_pad_w > 0 or mask_pad_h > 0:
                    padded_pil_mask = ImageOps.expand(
                        pil_mask, border=(0, 0, mask_pad_w, mask_pad_h), fill=0
                    )
                else:
                    padded_pil_mask = pil_mask

                # Convert back to tensors
                padded_img = pil2tensor(padded_pil_img)
                padded_mask = image2mask(padded_pil_mask)

                padded_images.append(padded_img)
                padded_masks.append(padded_mask)

            ret_images = padded_images
            ret_masks = padded_masks

        # Return crop boxes - use all_crop_boxes if multi-crop, otherwise single box
        if detect == "crop_each_region" and all_rects and len(all_rects) > 0:
            return_crop_boxes = all_crop_boxes
        else:
            return_crop_boxes = [list(crop_box)]

        return (
            torch.cat(ret_images, dim=0),
            torch.cat(ret_masks, dim=0),
            return_crop_boxes,
            pil2tensor(preview_image),
        )
