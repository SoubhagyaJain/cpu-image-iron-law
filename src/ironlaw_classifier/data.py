"""Deterministic, offline synthetic RGB image data.

Classes are visually different patterns; the model learns real labels without
having to download a dataset. This is a systems experiment, not a claim of
performance on ImageNet or photographic images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

IMAGE_SIZE = 64
CLASS_NAMES = ("red_horizontal", "green_vertical", "blue_circle")


def make_pattern(label: int, rng: np.random.Generator, size: int = IMAGE_SIZE) -> np.ndarray:
    if label not in (0, 1, 2):
        raise ValueError(f"invalid class index: {label}")
    image = rng.integers(0, 28, size=(size, size, 3), dtype=np.uint8)
    center = size // 2
    shift = int(rng.integers(-8, 9))
    thickness = int(rng.integers(9, 15))
    if label == 0:
        y = center + shift
        image[max(0, y - thickness // 2):min(size, y + thickness // 2), 5:-5] = (235, 28, 28)
    elif label == 1:
        x = center + shift
        image[5:-5, max(0, x - thickness // 2):min(size, x + thickness // 2)] = (28, 235, 28)
    else:
        cy, cx = center + shift, center - shift
        radius = int(rng.integers(10, 18))
        yy, xx = np.ogrid[:size, :size]
        image[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = (28, 28, 235)
    return image


def make_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n, dtype=np.int64) % len(CLASS_NAMES)
    rng.shuffle(labels)
    images = np.stack([make_pattern(int(label), rng) for label in labels])
    # In memory training matches 0..1 RGB conversion used at inference.
    x = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
    return x, torch.from_numpy(labels.copy())


def create_sample_images(directory: str | Path, seed: int = 17) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for label, name in enumerate(CLASS_NAMES):
        Image.fromarray(make_pattern(label, rng)).save(path / f"{name}.png")
