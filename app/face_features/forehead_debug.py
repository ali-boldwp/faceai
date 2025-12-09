import os
from typing import Optional

import cv2
import numpy as np

# Reuse the landmark indices from your profile helper
from app.face_features.forehead_profile import (
    TRICHION,
    GLABELLA,
    NASION,
    GNATHION,
)


def _ensure_color(img: np.ndarray) -> np.ndarray:
    """Guarantee we draw on a 3-channel BGR image."""
    if img.ndim == 2 or img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def draw_forehead_front_debug(
    img: np.ndarray,
    landmarks_px: np.ndarray,
    hairline_y: Optional[int],
    forehead_width: Optional[float],
    face_height: Optional[float],
    cheekbone_width: Optional[float],
    out_path: str,
) -> None:
    """
    Draw forehead-related lines on the FRONT image:

    - forehead_width (horizontal line near top of face)
    - face_height (vertical line through the middle of the face)
    - cheekbone_width (horizontal line around cheekbones)
    - optional hairline_y marker
    """
    vis = _ensure_color(img)
    h, w = vis.shape[:2]

    pts = landmarks_px.astype(np.int32)
    x_min = int(np.min(pts[:, 0]))
    x_max = int(np.max(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    y_max = int(np.max(pts[:, 1]))
    cx = (x_min + x_max) // 2

    # --- forehead_width line ---
    if forehead_width is not None and forehead_width > 0:
        half = forehead_width / 2.0
        x0 = int(round(cx - half))
        x1 = int(round(cx + half))

        # clamp to image
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w - 1, x1))

        # put it in the upper part of the face
        y_fw = y_min + (y_max - y_min) // 5
        y_fw = max(0, min(h - 1, y_fw))

        cv2.line(vis, (x0, y_fw), (x1, y_fw), (0, 255, 0), 2)
        cv2.putText(
            vis,
            "forehead_width",
            (x0, max(0, y_fw - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # --- face_height line (approx) ---
    if face_height is not None:
        cv2.line(vis, (cx, y_min), (cx, y_max), (255, 0, 0), 2)
        cv2.putText(
            vis,
            "face_height",
            (cx + 5, (y_min + y_max) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # --- cheekbone_width line (approx mid face) ---
    if cheekbone_width is not None:
        y_cb = y_min + (y_max - y_min) * 2 // 5
        y_cb = max(0, min(h - 1, y_cb))

        cv2.line(vis, (x_min, y_cb), (x_max, y_cb), (0, 165, 255), 2)
        cv2.putText(
            vis,
            "cheekbone_width",
            (x_min, max(0, y_cb - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    # --- hairline marker ---
    if hairline_y is not None:
        hy = int(hairline_y)
        if 0 <= hy < h:
            cv2.line(vis, (0, hy), (w - 1, hy), (255, 255, 0), 1)
            cv2.putText(
                vis,
                "hairline",
                (5, max(0, hy - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(out_path, vis)
    print(f"[DEBUG] Forehead front debug saved to {out_path}")
    

def draw_forehead_profile_debug(
    side_img: np.ndarray,
    side_landmarks_px: np.ndarray,
    out_path: str,
) -> None:
    """
    Draw Tr–G, Tr–N and Tr–Gn lines on the SIDE image:

    - Tr–G   → Forehead Height I
    - Tr–N   → Forehead Height II
    - Tr–Gn  → Total face height in profile
    """
    if side_img is None or side_landmarks_px is None:
        return

    vis = _ensure_color(side_img)
    pts = side_landmarks_px.astype(np.int32)

    def to_pt(idx: int):
        p = pts[idx]
        return int(p[0]), int(p[1])

    Tr = to_pt(TRICHION)
    G = to_pt(GLABELLA)
    N = to_pt(NASION)
    Gn = to_pt(GNATHION)

    # Tr-G line (forehead height I)
    cv2.line(vis, Tr, G, (0, 255, 0), 2)
    mid_tg = ((Tr[0] + G[0]) // 2, (Tr[1] + G[1]) // 2)
    cv2.putText(
        vis,
        "Tr-G",
        (mid_tg[0] + 5, mid_tg[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    # Tr-N line (forehead height II)
    cv2.line(vis, Tr, N, (255, 0, 0), 2)
    mid_tn = ((Tr[0] + N[0]) // 2, (Tr[1] + N[1]) // 2)
    cv2.putText(
        vis,
        "Tr-N",
        (mid_tn[0] + 5, mid_tn[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    # Tr-Gn line (profile face height)
    cv2.line(vis, Tr, Gn, (0, 0, 255), 1)
    mid_tgn = ((Tr[0] + Gn[0]) // 2, (Tr[1] + Gn[1]) // 2)
    cv2.putText(
        vis,
        "Tr-Gn",
        (mid_tgn[0] + 5, mid_tgn[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(out_path, vis)
    print(f"[DEBUG] Forehead profile debug saved to {out_path}")
