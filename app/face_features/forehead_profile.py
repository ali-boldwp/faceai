import math
from typing import Dict

import numpy as np

# Adjust these to your actual landmark indices (Mediapipe mesh or your map)
# Example assuming you have a LANDMARKS dict:
# from app.landmarks.constants import LANDMARKS
# TRICHION = LANDMARKS["Tr"]
# GLABELLA = LANDMARKS["G"]
# NASION   = LANDMARKS["N"]
# GNATHION = LANDMARKS["Gn"]

TRICHION = 10   # 👉 replace with REAL index for Tr (hairline center) in profile
GLABELLA = 6    # 👉 replace with REAL index for G (between eyebrows)
NASION   = 5    # 👉 replace with REAL index for N (nasal root)
GNATHION = 200  # 👉 replace with REAL index for Gn (chin bottom)


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def compute_forehead_profile_measurements(side_landmarks_px: np.ndarray) -> Dict[str, float]:
    """
    Compute forehead-related measurements from PROFILE landmarks.

    Parameters
    ----------
    side_landmarks_px : np.ndarray
        Array of shape (N, 2) or (N, 3) with profile landmarks in pixels.

    Returns
    -------
    Dict[str, float]
        Keys are added to the measurements dict and later consumed by
        classify_forehead_traits. This function is ONLY used for forehead traits,
        never for face type.
    """
    if side_landmarks_px.ndim != 2 or side_landmarks_px.shape[1] < 2:
        raise ValueError("side_landmarks_px must be (N, 2) or (N, 3)")

    Tr = side_landmarks_px[TRICHION][:2]
    G  = side_landmarks_px[GLABELLA][:2]
    N  = side_landmarks_px[NASION][:2]
    Gn = side_landmarks_px[GNATHION][:2]

    # Basic distances
    tr_g  = _dist(Tr, G)   # Forehead height I (Tr–G)
    tr_n  = _dist(Tr, N)   # Forehead height II (Tr–N)
    tr_gn = _dist(Tr, Gn)  # Approx total facial height in profile (Tr–Gn)

    if tr_gn > 0:
        rel_tr_g = tr_g / tr_gn
        rel_tr_n = tr_n / tr_gn
    else:
        rel_tr_g = 0.0
        rel_tr_n = 0.0

    # Forehead slope vs vertical, based on line Tr–G
    # Image coordinates: x → right, y → down
    # Vector from G to Tr:
    v = Tr - G
    vx, vy = float(v[0]), float(v[1])

    # Angle from vertical; 0° = perfectly vertical, sign ~ direction
    angle_rad = math.atan2(vx, vy)  # vertical reference (0,1)
    angle_deg = math.degrees(angle_rad)
    angle_abs = abs(angle_deg)

    return {
        # Raw profile distances
        "forehead_height_tr_g_profile": tr_g,
        "forehead_height_tr_n_profile": tr_n,
        "face_height_tr_gn_profile": tr_gn,

        # Ratios used for "high / low forehead" etc.
        "forehead_height_ratio_tr_g_profile": rel_tr_g,
        "forehead_height_ratio_tr_n_profile": rel_tr_n,

        # Slope metrics used for "bombată / dreaptă / înclinată"
        "forehead_slope_signed_deg_profile": angle_deg,
        "forehead_slope_deg_profile": angle_abs,
    }
