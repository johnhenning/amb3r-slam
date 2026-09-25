from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .contracts import Array


def image_tensor(images: Sequence[Array], size: tuple[int, int]) -> Array:
    """NCHW RGB, resize, ImageNet normalization; size=(height,width)."""
    h, w = size
    batch = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in images])
    batch = batch.astype(np.float32) / 255.0
    batch = (batch - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2)[None])
