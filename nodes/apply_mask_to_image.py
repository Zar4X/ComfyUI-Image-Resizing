import torch


class ApplyMaskToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("masked",)
    FUNCTION = "apply_mask_to_image"
    CATEGORY = "Image Resizing/Mask"

    def apply_mask_to_image(self, image, mask):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 0)
        if image.dim() != 4:
            raise ValueError(f"Image should have shape [B, H, W, C], got {tuple(image.shape)}")

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        elif mask.dim() == 4:
            # Be tolerant to mask layouts like [B, H, W, 1] or [B, 1, H, W].
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            elif mask.shape[1] == 1:
                mask = mask[:, 0, :, :]
            else:
                raise ValueError(
                    f"Unsupported 4D mask shape {tuple(mask.shape)}. Expected [...,1] channel."
                )

        if mask.dim() != 3:
            raise ValueError(f"Mask should have shape [B, H, W], got {tuple(mask.shape)}")

        batch_size, img_h, img_w, channels = image.shape
        if channels not in (3, 4):
            raise ValueError(
                f"Image should have 3 or 4 channels in BHWC layout, got {channels} channels."
            )

        if mask.shape[-2:] != (img_h, img_w):
            raise ValueError(
                f"Image size {(img_h, img_w)} must match mask size {tuple(mask.shape[-2:])}."
            )

        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        elif mask.shape[0] != batch_size:
            raise ValueError(
                f"Mask batch {mask.shape[0]} must be 1 or match image batch {batch_size}."
            )

        mask = mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
        alpha = torch.unsqueeze(mask, -1)

        # Always output RGBA so masked-out area is transparent instead of black.
        if channels == 3:
            rgb = image
            out = torch.cat([rgb, alpha], dim=-1)
        else:
            out = image.clone()
            out[..., 3:4] = out[..., 3:4] * alpha

        return (out,)
