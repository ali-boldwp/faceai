# app/services/profile_landmark_extraction.py

from typing import Optional, Tuple, List
import numpy as np
import cv2
import torch
import face_alignment

# Use GPU if present
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2D FAN model, trained for large poses (LS3D-W)
_fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D,
    device=_DEVICE,
    flip_input=False,
)


def extract_profile_landmarks(
    img_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Landmarks for side/profile faces using FAN.


    Returns:
        landmarks: (N, 2) float32 pixel coords or None
        score: dummy confidence (1.0 if landmarks found, else 0.0)
    """
    if img_bgr is None:
        return None, 0.0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    preds: Optional[List[np.ndarray]] = _fa.get_landmarks(img_rgb)
    if preds is None or len(preds) == 0:
        return None, 0.0

    lm = preds[0]  # (68, 2)
    return lm.astype(np.float32), 1.0
