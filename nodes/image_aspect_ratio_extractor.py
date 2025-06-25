import torch
import math
import numpy as np
from PIL import Image


def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL image"""
    return Image.fromarray(
        np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


class ImageAspectRatioExtractor:
    MINIMUM_RATIO = 1 / 4  # 0.25
    MAXIMUM_RATIO = 4 / 1  # 4.0
    MINIMUM_RATIO_STR = "1:4"
    MAXIMUM_RATIO_STR = "4:1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ratio_type": (["round to", "aspect ratio"], {"default": "round to"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("aspect_ratio",)
    FUNCTION = "extract_aspect_ratio"
    CATEGORY = "Image Resizing/Extract"
    DESCRIPTION = "Extracts aspect ratio from input images with constraints (0.25-4.0). Can output exact ratio or round to common ratios."

    def get_common_ratios(self):
        """Returns a list of common aspect ratios with their decimal values (within 0.25-4.0 range)"""
        return [
            ("1:4", 1/4),    # 0.25 - minimum ratio
            ("1:2", 1/2),    # 0.5
            ("3:4", 3/4),    # 0.75
            ("1:1", 1.0),    # 1.0
            ("4:3", 4/3),    # 1.33
            ("3:2", 3/2),    # 1.5
            ("16:9", 16/9),  # 1.78
            ("2:1", 2/1),    # 2.0
            ("4:1", 4/1),    # 4.0 - maximum ratio
        ]

    def clamp_ratio(self, ratio):
        """Clamp the aspect ratio to be within valid range (0.25-4.0)"""
        return max(self.MINIMUM_RATIO, min(self.MAXIMUM_RATIO, ratio))

    def extract_aspect_ratio(self, image, ratio_type):
        """
        Extracts aspect ratio from input images with constraints.
        
        Args:
            image: Input image tensor (can be single image or batch)
            ratio_type: "round to" for common ratios or "aspect ratio" for exact ratio
            
        Returns:
            String representing the aspect ratio (e.g., "16:9" or "65:128")
        """
        # 确保输入是张量
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input must be a tensor")
            
        # 检查空输入
        if image.nelement() == 0:
            raise ValueError("Input image tensor is empty")
            
        # 获取第一张图像的尺寸（假设批次中所有图像尺寸相同）
        if len(image.shape) == 4:  # 批次格式 [B, H, W, C]
            width, height = image.shape[2], image.shape[1]
        elif len(image.shape) == 3:  # 单张图像格式 [H, W, C]
            width, height = image.shape[1], image.shape[0]
        else:
            raise ValueError(f"Unsupported image tensor shape: {image.shape}")
        
        # 验证尺寸
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")
        
        # 计算原始宽高比
        original_ratio = width / height
        
        # 应用比例约束
        clamped_ratio = self.clamp_ratio(original_ratio)
        
        # 如果原始比例超出范围，发出警告
        if abs(original_ratio - clamped_ratio) > 0.001:
            print(f"Warning: Original aspect ratio {original_ratio:.3f} is outside valid range ({self.MINIMUM_RATIO}-{self.MAXIMUM_RATIO}). Clamped to {clamped_ratio:.3f}")
        
        # 计算宽高比
        if ratio_type == "aspect ratio":
            # 基于约束后的比例计算精确比例
            if clamped_ratio == self.MINIMUM_RATIO:
                result = self.MINIMUM_RATIO_STR
            elif clamped_ratio == self.MAXIMUM_RATIO:
                result = self.MAXIMUM_RATIO_STR
            else:
                # 对于中间值，尝试找到简单的整数比例
                # 使用原始尺寸但确保比例在范围内
                if original_ratio < self.MINIMUM_RATIO:
                    # 调整到最小比例
                    result = self.MINIMUM_RATIO_STR
                elif original_ratio > self.MAXIMUM_RATIO:
                    # 调整到最大比例
                    result = self.MAXIMUM_RATIO_STR
                else:
                    # 计算最大公约数
                    gcd = math.gcd(width, height)
                    result = f"{width//gcd}:{height//gcd}"
        else:  # "round to"
            common_ratios = self.get_common_ratios()
            # 找到最接近的常见比例（基于约束后的比例）
            closest_ratio = min(common_ratios, key=lambda x: abs(x[1] - clamped_ratio))
            result = closest_ratio[0]
        
        print(f"Image resolution: {width}x{height}, Original ratio: {original_ratio:.3f}, Clamped ratio: {clamped_ratio:.3f}, {ratio_type}: {result}")
        return (result,) 