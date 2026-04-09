import torch
import numpy as np
from PIL import Image
from .ImgMods import tensor2pil, pil2tensor, image2mask


class RemovePadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "INT",
                    {"default": 5, "min": 0, "max": 255, "step": 1},
                ),
                "use_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_box")
    FUNCTION = "remove_padding"
    CATEGORY = "Image Resizing/Resizing"

    def remove_padding(
        self, image, threshold, use_mask, mask=None
    ):
        """
        Remove black/transparent padding from images by detecting content boundaries.
        
        Args:
            image: Input image tensor(s)
            threshold: Pixel value threshold (0-255). Pixels below this are considered padding.
            use_mask: If True, use mask to determine content area. If False, detect from image.
            mask: Optional mask to use for detection (if use_mask is True)
        """
        ret_images = []
        ret_masks = []
        crop_boxes = []

        # Process each image in the batch
        for img_idx, img in enumerate(image):
            pil_img = tensor2pil(img).convert("RGB")
            
            # Determine content area
            if use_mask and mask is not None:
                # Use mask to find content boundaries
                if mask.dim() == 2:
                    mask_tensor = mask
                elif mask.dim() == 3:
                    mask_tensor = mask[img_idx] if img_idx < mask.shape[0] else mask[0]
                else:
                    mask_tensor = mask[0]
                
                pil_mask = tensor2pil(mask_tensor).convert("L")
                content_box = self._find_content_from_mask(pil_mask, threshold)
            else:
                # Detect content from image itself
                content_box = self._find_content_from_image(pil_img, threshold)
            
            if content_box is None:
                # No content found, return original
                ret_images.append(img)
                w, h = pil_img.size
                if mask is not None:
                    if mask.dim() == 2:
                        ret_masks.append(mask)
                    elif mask.dim() == 3:
                        ret_masks.append(mask[img_idx] if img_idx < mask.shape[0] else mask[0])
                    else:
                        ret_masks.append(mask[0])
                else:
                    # Create empty mask
                    ret_masks.append(torch.zeros((1, h, w)))
                crop_boxes.append([0, 0, w, h])
                continue
            
            x1, y1, x2, y2 = content_box
            
            # Crop image
            cropped_pil_img = pil_img.crop((x1, y1, x2, y2))
            ret_images.append(pil2tensor(cropped_pil_img))
            
            # Crop mask if provided
            if mask is not None:
                if mask.dim() == 2:
                    mask_tensor = mask
                elif mask.dim() == 3:
                    mask_tensor = mask[img_idx] if img_idx < mask.shape[0] else mask[0]
                else:
                    mask_tensor = mask[0]
                
                pil_mask = tensor2pil(mask_tensor).convert("L")
                cropped_pil_mask = pil_mask.crop((x1, y1, x2, y2))
                ret_masks.append(image2mask(cropped_pil_mask))
            else:
                # Create mask for cropped area (full white mask)
                w, h = cropped_pil_img.size
                ret_masks.append(torch.ones((1, h, w)))
            
            crop_boxes.append([x1, y1, x2, y2])

        # Handle mask output - pad all masks to the same size if needed
        if len(ret_masks) > 0:
            # Check if all masks have the same size
            first_mask_shape = ret_masks[0].shape
            all_same_size = all(m.shape == first_mask_shape for m in ret_masks)
            
            if not all_same_size:
                # Pad all masks to the maximum size
                max_h = max(m.shape[1] for m in ret_masks)
                max_w = max(m.shape[2] for m in ret_masks)
                
                padded_masks = []
                for m in ret_masks:
                    if m.shape[1] == max_h and m.shape[2] == max_w:
                        padded_masks.append(m)
                    else:
                        # Pad mask to max size
                        pad_h = max_h - m.shape[1]
                        pad_w = max_w - m.shape[2]
                        padded_mask = torch.nn.functional.pad(
                            m, (0, pad_w, 0, pad_h), mode='constant', value=0
                        )
                        padded_masks.append(padded_mask)
                output_mask = torch.cat(padded_masks, dim=0)
            else:
                output_mask = torch.cat(ret_masks, dim=0)
        else:
            output_mask = torch.zeros((1, 1, 1))

        return (
            torch.cat(ret_images, dim=0),
            output_mask,
            crop_boxes[0] if len(crop_boxes) == 1 else crop_boxes,
        )

    def _find_content_from_image(self, pil_image, threshold):
        """Find content boundaries from image by detecting non-black pixels."""
        img_array = np.array(pil_image.convert("RGB"))
        
        # Find rows and columns that have content (non-black pixels)
        # Content is defined as pixels where any channel is above threshold
        has_content = np.any(img_array > threshold, axis=2)
        
        if not np.any(has_content):
            return None
        
        # Find bounding box of content
        rows = np.any(has_content, axis=1)
        cols = np.any(has_content, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y1 = np.argmax(rows)
        y2 = len(rows) - np.argmax(rows[::-1])
        x1 = np.argmax(cols)
        x2 = len(cols) - np.argmax(cols[::-1])
        
        return (x1, y1, x2, y2)

    def _find_content_from_mask(self, pil_mask, threshold):
        """Find content boundaries from mask."""
        mask_array = np.array(pil_mask.convert("L"))
        
        # Find rows and columns that have content (pixels above threshold)
        has_content = mask_array > threshold
        
        if not np.any(has_content):
            return None
        
        rows = np.any(has_content, axis=1)
        cols = np.any(has_content, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y1 = np.argmax(rows)
        y2 = len(rows) - np.argmax(rows[::-1])
        x1 = np.argmax(cols)
        x2 = len(cols) - np.argmax(cols[::-1])
        
        return (x1, y1, x2, y2)

