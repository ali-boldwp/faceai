import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def draw_landmarks_image(
    image: np.ndarray,
    landmarks: Sequence[Sequence[int]],
    out_path: str,
    radius: int = 1,
    thickness: int = -1,
) -> str:
    """Draws face landmarks as small points on a copy of the image and saves it.

    Args:
        image: BGR image (as loaded by OpenCV).
        landmarks: Iterable of (x, y) pixel coordinates.
        out_path: Where to save the landmark overlay image.
        radius: Point radius.
        thickness: Point thickness (-1 = filled circle).

    Returns:
        The path where the image was saved.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis = image.copy()

    for pt in landmarks:
        if len(pt) >= 2:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(vis, (x, y), radius, (0, 255, 0), thickness)

    cv2.imwrite(out_path, vis)
    return out_path
