# app/services/ear_landmark_extraction.py

from typing import Optional, Tuple
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# ---- paths (adjust if you put them somewhere else) ----
EAR_MODEL_PATH = os.path.join("app", "models", "ear", "my_model.h5")
EAR_CASCADE_PATH = os.path.join("app", "models", "ear", "haarcascade_mcs_rightear.xml")

# ---- load model & optional cascade once ----
_ear_model = load_model(EAR_MODEL_PATH)

_ear_cascade = cv2.CascadeClassifier(EAR_CASCADE_PATH) if os.path.exists(
    EAR_CASCADE_PATH
) else None


def _find_ear_roi(
    img_bgr: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Find a rough ear bounding box on a 90° right-profile head.

    1. Try Haar cascade if available.
    2. Fallback to a heuristic crop on the left/middle side of the image.

    Returns:
        (x, y, w, h) in image coordinates.
    """
    h, w = img_bgr.shape[:2]

    if _ear_cascade is not None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ears = _ear_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(int(w * 0.15), int(h * 0.15)),
        )
        if len(ears) > 0:
            # take the largest detection
            x, y, ew, eh = max(ears, key=lambda b: b[2] * b[3])
            return int(x), int(y), int(ew), int(eh)

    # Fallback: heuristic crop for a right-facing profile
    # Ear is roughly on left 60% horizontally and mid 70% vertically.
    x = int(w * 0.05)
    y = int(h * 0.15)
    ew = int(w * 0.55)
    eh = int(h * 0.70)
    # Clamp to image
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    ew = max(10, min(ew, w - x))
    eh = max(10, min(eh, h - y))
    return x, y, ew, eh


def extract_ear_landmarks(
    side_img_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Run the 55-landmark ear CNN on the side image and map results back to
    full-image coordinates.

    Returns:
        landmarks: (55, 2) float32 [x, y] in side image pixel coords or None
        score: dummy confidence (1.0 if success, 0.0 otherwise)
    """
    if side_img_bgr is None:
        return None, 0.0

    H, W = side_img_bgr.shape[:2]
    if H < 50 or W < 50:
        return None, 0.0

    # 1) crop ROI that should contain the ear
    x, y, ew, eh = _find_ear_roi(side_img_bgr)
    roi = side_img_bgr[y : y + eh, x : x + ew]

    if roi.size == 0:
        return None, 0.0

    # 2) resize to the model's expected 224x224
    roi_resized = cv2.resize(roi, (224, 224))
    roi_resized = roi_resized.astype(np.float32)

    # 3) Keras preprocess_input (ImageNet style)
    x_in = np.expand_dims(roi_resized, axis=0)
    x_in = preprocess_input(x_in)

    # 4) predict 110 values: first 55 x, next 55 y in [0..224) pixel space
    pred = _ear_model.predict(x_in, verbose=0)[0]  # shape (110,)
    xs = pred[:55]
    ys = pred[55:]

    # 5) map from 224x224 ROI coords back to original side image
    xs_norm = xs / 224.0
    ys_norm = ys / 224.0

    xs_img = x + xs_norm * ew
    ys_img = y + ys_norm * eh

    landmarks = np.stack([xs_img, ys_img], axis=1).astype(np.float32)

    return landmarks, 1.0
