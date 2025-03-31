import torch
from PIL import Image, ImageFilter, ImageColor
from .ImgMods import tensor2pil, pil2tensor, image2mask


class ExtendCanvasByPercentage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "use_pixels": ("BOOLEAN", {"default": False}),
                "top": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "left": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "color": (
                    "STRING",
                    {
                        "default": "#7f7f7f",
                        "color": {"type": "color"},
                        "description": "支持格式：#RRGGBB 或 R,G,B（如 255,127,0）",
                    },
                ),
            },
            "optional": {
                "mask": ("MASK",),  # 唯一的mask输入参数
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "extend_canvas"
    CATEGORY = "Image Resizing/Extend Canvas"

    def parse_color(self, color_str):
        """智能颜色解析"""
        try:
            color_str = color_str.strip().lower()

            if color_str.startswith("#"):
                return ImageColor.getrgb(color_str)

            if "," in color_str:
                rgb = tuple(map(int, color_str.split(",")))
                if len(rgb) == 3 and all(0 <= c <= 255 for c in rgb):
                    return rgb

            return ImageColor.getrgb(color_str)
        except:
            print(f"无法解析颜色输入：{color_str}，使用默认灰色")
            return (127, 127, 127)

    def extend_canvas(
        self,
        image,
        invert_mask,
        use_pixels,
        top,
        bottom,
        left,
        right,
        feather,
        color,
        mask=None,
    ):
        # 颜色解析
        parsed_color = self.parse_color(color)
        print(f"[ExtendCanvas] 输入颜色：'{color}' → 解析为：{parsed_color}")

        # 处理mask逻辑
        if mask is not None:
            # 优先使用外部传入的mask
            l_masks = [tensor2pil(m).convert("L") for m in mask]
        else:
            # 次选使用image的alpha通道
            l_masks = []
            for img in image:
                pil_img = tensor2pil(img)
                if pil_img.mode == "RGBA":
                    l_masks.append(pil_img.split()[-1])
                else:
                    # 生成白色默认mask
                    l_masks.append(Image.new("L", pil_img.size, 255))

        # 图像处理
        ret_images = []
        ret_masks = []
        fill_color = parsed_color

        max_batch = max(len(image), len(l_masks))
        for i in range(max_batch):
            _image = image[i] if i < len(image) else image[-1]
            _image_pil = tensor2pil(_image).convert("RGB")
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            # 计算画布尺寸
            orig_w, orig_h = _image_pil.size
            if use_pixels:
                top_px, bottom_px = top, bottom
                left_px, right_px = left, right
            else:
                top_px = int(orig_h * (top / 100))
                bottom_px = int(orig_h * (bottom / 100))
                left_px = int(orig_w * (left / 100))
                right_px = int(orig_w * (right / 100))

            new_w = orig_w + left_px + right_px
            new_h = orig_h + top_px + bottom_px

            # 创建画布
            canvas = Image.new("RGB", (new_w, new_h), fill_color)
            mask_canvas = Image.new("L", (new_w, new_h), 0)

            # 合成图像
            canvas.paste(_image_pil, (left_px, top_px))
            mask_canvas.paste(_mask, (left_px, top_px))

            # 边缘羽化
            if feather > 0:
                mask_canvas = mask_canvas.filter(
                    ImageFilter.GaussianBlur(radius=feather)
                )

            # 反相蒙版
            if invert_mask:
                mask_canvas = Image.eval(mask_canvas, lambda x: 255 - x)

            ret_images.append(pil2tensor(canvas))
            ret_masks.append(image2mask(mask_canvas))

        return (
            torch.cat(ret_images, dim=0),
            torch.cat(ret_masks, dim=0),
        )


NODE_CLASS_MAPPINGS = {"ExtendCanvasByPercentage (ZX)": ExtendCanvasByPercentage}
