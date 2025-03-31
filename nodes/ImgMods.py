import os
import sys
import math
import re
import glob
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw
from typing import Union, List
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    return Image.fromarray(
        np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def mask2image(mask: torch.Tensor) -> Image.Image:
    masks = tensor2np(mask)
    for m in masks:
        _mask = Image.fromarray(m).convert("L")
        _image = Image.new("RGBA", _mask.size, color="white")
        _image = Image.composite(
            _image, Image.new("RGBA", _mask.size, color="black"), _mask
        )
    return _image


def image2mask(image: Image.Image) -> torch.Tensor:
    if image.mode == "L":
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    return torch.tensor([pil2tensor(image.convert("RGB").split()[0])[0, :, :].tolist()])


def gaussian_blur(image: Image.Image, radius: int) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def min_bounding_rect(image: Image.Image) -> tuple:
    cv2_image = pil2cv2(image)
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, 1, 2)
    max_area = 0
    final_rect = (0, 0, 0, 0)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > max_area:
            max_area = w * h
            final_rect = (x, y, w, h)
    return final_rect


def max_inscribed_rect(image: Image) -> tuple:
    img = pil2cv2(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0].reshape(len(contours[0]), 2)
    rect = []
    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2 - y1) * abs(x2 - x1)
            rect.append(((x1, y1), (x2, y2), area))
    all_rect = sorted(rect, key=lambda x: x[2], reverse=True)
    if all_rect:
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)
        while not best_rect_found and index_rect < nb_rect:
            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]
            valid_rect = True
            x = min(x1, x2)
            while x < max(x1, x2) + 1 and valid_rect:
                if any(img[y1, x]) == 0 or any(img[y2, x]) == 0:
                    valid_rect = False
                x += 1
            y = min(y1, y2)
            while y < max(y1, y2) + 1 and valid_rect:
                if any(img[y, x1]) == 0 or any(img[y, x2]) == 0:
                    valid_rect = False
                y += 1
            if valid_rect:
                best_rect_found = True
            index_rect += 1
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return (x1, y1, x2 - x1, y2 - y1)


def mask_area(image: Image) -> tuple:
    cv2_image = pil2cv2(image.convert("RGBA"))
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    locs = np.where(thresh == 255)
    try:
        x1 = np.min(locs[1])
        x2 = np.max(locs[1])
        y1 = np.min(locs[0])
        y2 = np.max(locs[0])
    except ValueError:
        x1, y1, x2, y2 = -1, -1, 0, 0
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return (x1, y1, x2 - x1, y2 - y1)


def num_round_up_to_multiple(number: int, multiple: int) -> int:
    remainder = number % multiple
    return number if remainder == 0 else ((number // multiple) + 1) * multiple


def draw_rect(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    line_color: str,
    line_width: int,
    box_color: str = None,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (x, y, x + width, y + height),
        fill=box_color,
        outline=line_color,
        width=line_width,
    )
    return image


def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
    return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]


def pil2cv2(pil_img: Image.Image) -> np.ndarray:
    np_img = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


class AnyType(str):

    def __eq__(self, __value: object) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


def apply_to_batch(func):
    def wrapper(self, image, *args, **kwargs):
        return torch.cat([func(self, img, *args, **kwargs) for img in image], dim=0)

    return wrapper
