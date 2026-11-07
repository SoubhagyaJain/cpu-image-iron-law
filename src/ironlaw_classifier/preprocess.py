"""Equivalent 0..1 RGB preprocessing implementations.

Timing includes in-memory PNG decoding + RGB conversion + resizing + tensor
creation and normalization. It excludes disk reads and model loading.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
import torch

from .data import IMAGE_SIZE


def _decode_resize(png_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(png_bytes)) as opened:
        return opened.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)


def preprocess_python(png_bytes: bytes) -> torch.Tensor:
    """Beginner-style Python scalar loop. Correct, but expensive per pixel."""
    image = _decode_resize(png_bytes)
    flattened = []
    pixels = image.load()
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            red, green, blue = pixels[x, y]
            flattened.extend((red / 255.0, green / 255.0, blue / 255.0))
    return torch.tensor(flattened, dtype=torch.float32).reshape(
        1, IMAGE_SIZE, IMAGE_SIZE, 3
    ).permute(0, 3, 1, 2).contiguous()


def preprocess_numpy(png_bytes: bytes) -> torch.Tensor:
    """Vectorized conversion of the identical decoded/resized RGB image."""
    image = _decode_resize(png_bytes)
    array = np.asarray(image, dtype=np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
