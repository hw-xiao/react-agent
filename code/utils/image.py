"""
图片加载与预处理工具。
支持 base64 / URL / 本地路径三种来源，自动缩放防 OOM。
"""

import base64
import os
from io import BytesIO
from typing import Any

from PIL import Image

from code.config.settings import MAX_IMAGE_DIMENSION
from code.utils.logger import get_logger

log = get_logger("image")


def resize_image(img: Image.Image) -> Image.Image:
    """等比缩放图片，确保最大边不超过 MAX_IMAGE_DIMENSION 像素。"""
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_DIMENSION:
        return img
    scale = MAX_IMAGE_DIMENSION / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    log.debug(f"缩放图片: {w}x{h} -> {new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)


def load_image(value: Any) -> Image.Image:
    """
    从多种来源加载图片为 PIL.Image 对象。

    支持:
      - data:image/...;base64,xxxx 格式的 base64 编码
      - http/https URL
      - 本地文件路径

    Returns:
        RGB 模式的 PIL.Image，已缩放至安全尺寸。
    """
    if not isinstance(value, str):
        raise ValueError("Image content must be a string path/URL/base64")

    if value.startswith("data:image"):
        header, data = value.split(",", 1)
        image_bytes = base64.b64decode(data)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        log.debug(f"从 base64 加载图片")
        return resize_image(img)

    if value.startswith("http://") or value.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(value, timeout=30) as resp:
            image_bytes = resp.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        log.debug(f"从 URL 加载图片: {value}")
        return resize_image(img)

    if os.path.exists(value):
        img = Image.open(value).convert("RGB")
        log.debug(f"从本地路径加载图片: {value}")
        return resize_image(img)

    raise ValueError(f"Unsupported image reference: {value}")
