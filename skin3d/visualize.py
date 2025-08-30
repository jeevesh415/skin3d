from __future__ import annotations

import numpy as np
import pandas as pd


def embed_box_borders(img, x, y, w, h, color: tuple[int, int, int], pad: int):
    p = pad

    img[y : y + h, x - p : x, :] = color
    img[y : y + h, x + w : x + w + p, :] = color
    img[y - p : y, x - p : x + w + p, :] = color
    img[y + h : y + h + p, x - p : x + w + p, :] = color


def embed_annotatations(
    img: np.ndarray, annotations: pd.DataFrame, color: tuple[int, int, int], pad: int
) -> None:
    """Embed the annotations in the pixels of the image."""
    for _, row in annotations.iterrows():
        embed_box_borders(img, row.x, row.y, row.width, row.height, color, pad)
