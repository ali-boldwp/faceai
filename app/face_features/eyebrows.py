import numpy as np
from typing import Dict, List, Tuple


def classify_eyebrow_traits(landmarks: List[List[float]]) -> Tuple[Dict, List[Dict]]:
    """
    Basic eyebrow morphology classifier using MediaPipe FaceMesh landmarks.

    Input:
      landmarks: list of [x, y] pixel coordinates ordered by landmark index (0..467)

    Returns:
      (shape_info, traits)

      shape_info: {"label": <str>, "justification": <str>}
      traits: list of dicts with keys: meaning, explanation, source

    It covers a subset of your Eyebrows catalog:
      * New Moon Eyebrows (Arched)
      * Straight Eyebrows
      * Ascending / Descending Eyebrows
      * High-Set / Low-Set Eyebrows
      * Close (Narrow) Eyebrow Gap / Wide Brow Gap / Unibrow
    """

    if landmarks is None or len(landmarks) < 340:
        return (
            {
                "label": "Eyebrows (undetermined)",
                "justification": "Insufficient landmarks.",
            },
            [],
        )

    pts = np.asarray(landmarks, dtype=np.float32)

    # Indices from Mediapipe FaceMesh (via simplified landmarks repo)
    # Left Eyebrow  = [70,63,105,66,107,55,65,52,53,46]
    # Right Eyebrow = [300,293,334,296,336,285,295,282,283,276]
    # Eyes (same indices as in eyes.py)
    LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    LEFT_EYE = [33, 159, 158, 157, 133, 145, 144, 153]
    RIGHT_EYE = [263, 386, 385, 384, 362, 374, 373, 380]

    try:
        left_brow = pts[LEFT_BROW]
        right_brow = pts[RIGHT_BROW]
        left_eye = pts[LEFT_EYE]
        right_eye = pts[RIGHT_EYE]
    except IndexError:
        return (
            {
                "label": "Eyebrows (undetermined)",
                "justification": "Landmarks out of range.",
            },
            [],
        )

    # ---------- helpers ----------

    def brow_arch_ratio(brow: np.ndarray) -> float:
        # ends + mid as arch apex
        p1 = brow[0]
        p2 = brow[-1]
        mid = brow[len(brow) // 2]

        dx = p2[0] - p1[0]
        if abs(dx) < 1e-6:
            baseline_y = (p1[1] + p2[1]) / 2.0
        else:
            t = (mid[0] - p1[0]) / dx
            baseline_y = p1[1] + t * (p2[1] - p1[1])

        arch_height = baseline_y - mid[1]  # image coords: y downwards
        length = float(np.linalg.norm(p2 - p1)) + 1e-6
        return abs(arch_height) / length

    def brow_tilt_angle(brow: np.ndarray) -> float:
        # sort end points by x (p1 inner, p2 outer approx)
        p1 = brow[0]
        p2 = brow[-1]
        if p2[0] < p1[0]:
            p1, p2 = p2, p1
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        if abs(dx) < 1e-6:
            return 0.0
        return float(np.degrees(np.arctan2(dy, dx)))

    def brow_height_ratio(brow: np.ndarray, eye: np.ndarray, eye_dist: float) -> float:
        brow_center_y = brow[:, 1].mean()
        eye_center_y = eye[:, 1].mean()
        delta = eye_center_y - brow_center_y  # >0 = brow higher than eye
        return float(delta / (eye_dist + 1e-6))

    def eyebrow_gap_ratio(
        left_brow: np.ndarray, right_brow: np.ndarray, eye_dist: float
    ) -> float:
        # inner-most points (closest to midline) by x
        left_inner = left_brow[np.argmax(left_brow[:, 0])]
        right_inner = right_brow[np.argmin(right_brow[:, 0])]
        gap = right_inner[0] - left_inner[0]
        return float(gap / (eye_dist + 1e-6))

    # ---------- metrics ----------

    left_arch = brow_arch_ratio(left_brow)
    right_arch = brow_arch_ratio(right_brow)
    arch_ratio = (left_arch + right_arch) / 2.0

    left_angle = brow_tilt_angle(left_brow)
    right_angle = brow_tilt_angle(right_brow)
    tilt_angle = (left_angle + right_angle) / 2.0

    left_eye_center = left_eye.mean(axis=0)
    right_eye_center = right_eye.mean(axis=0)
    eye_dist = float(np.linalg.norm(right_eye_center - left_eye_center)) + 1e-6

    left_height = brow_height_ratio(left_brow, left_eye, eye_dist)
    right_height = brow_height_ratio(right_brow, right_eye, eye_dist)
    height_ratio = (left_height + right_height) / 2.0

    gap_ratio = eyebrow_gap_ratio(left_brow, right_brow, eye_dist)

    traits: List[Dict] = []

    # ---------- Shape: arched vs straight ----------
    if arch_ratio >= 0.10:
        shape_label = "New Moon Eyebrows (Arched)"
        justification = f"Pronounced arch in both eyebrows (arch ratio={arch_ratio:.2f})."
        traits.append({
            "meaning": "Romantic, idealistic, emotionally expressive",
            "explanation": "Strongly arched 'new moon' eyebrows are linked with sensibility and imaginative expression.",
            "source": "Tehnica de citire a feței.docx – Sprâncene lună nouă (arcuite)",
        })
    elif arch_ratio <= 0.04:
        shape_label = "Straight Eyebrows"
        justification = f"Eyebrow curve is minimal (arch ratio={arch_ratio:.2f})."
        traits.append({
            "meaning": "Direct, logical thinker",
            "explanation": "Straight eyebrows are associated with a rational, objective style of thinking.",
            "source": "Tehnica de citire a feței.docx – Sprâncene drepte",
        })
    else:
        shape_label = "Slightly Arched Eyebrows"
        justification = f"Eyebrows show a moderate arch (arch ratio={arch_ratio:.2f})."
        traits.append({
            "meaning": "Balanced between logic and emotion",
            "explanation": "A moderate eyebrow arch indicates both sensitivity and practicality.",
            "source": "Tehnica de citire a feței.docx – Sprâncene mediu arcuite",
        })

    # ---------- Tilt: ascending / descending / horizontal ----------
    if tilt_angle <= -3.0:
        traits.append({
            "meaning": "Ambitious, oriented toward goals",
            "explanation": "Ascending eyebrows (outer part higher) are linked to an active, progressive orientation.",
            "source": "Tehnica de citire a feței.docx – Sprâncene ascendente",
        })
        shape_label += ", Ascending Eyebrows"
    elif tilt_angle >= 3.0:
        traits.append({
            "meaning": "Reflective, sometimes more cautious",
            "explanation": "Descending eyebrows can reflect a tendency toward introspection and reservation.",
            "source": "Tehnica de citire a feței.docx – Sprâncene descendente",
        })
        shape_label += ", Descending Eyebrows"
    else:
        traits.append({
            "meaning": "Stable, steady temperament",
            "explanation": "Horizontally aligned eyebrows suggest balance and realism.",
            "source": "Tehnica de citire a feței.docx – Sprâncene drepte",
        })

    # ---------- Height: high-set vs low-set ----------
    if height_ratio >= 0.30:
        traits.append({
            "meaning": "More detached, reflective",
            "explanation": "High-set eyebrows indicate distance between thought and immediate reaction.",
            "source": "Tehnica de citire a feței.docx – Sprâncene înalte",
        })
    elif height_ratio <= 0.18:
        traits.append({
            "meaning": "Impulsive, fast reactions",
            "explanation": "Low-set eyebrows close to the eyes are linked with quick emotional responses.",
            "source": "Tehnica de citire a feței.docx – Sprâncene joase",
        })

    # ---------- Gap: close / unibrow / wide ----------
    if gap_ratio <= 0.03:
        traits.append({
            "meaning": "Intense focus, sometimes possessive",
            "explanation": "A virtually absent eyebrow gap (unibrow) is associated with strong concentration and persistence.",
            "source": "Tehnica de citire a feței.docx – Sprâncene unite",
        })
    elif gap_ratio <= 0.10:
        traits.append({
            "meaning": "Analytical, detail-oriented",
            "explanation": "A narrow gap between eyebrows suggests a mind that quickly connects details.",
            "source": "Tehnica de citire a feței.docx – Sprâncene apropiate",
        })
    elif gap_ratio >= 0.25:
        traits.append({
            "meaning": "Broad view, sometimes dispersed focus",
            "explanation": "A wide eyebrow gap indicates generosity and a tendency to see the big picture.",
            "source": "Tehnica de citire a feței.docx – Sprâncene depărtate",
        })

    return (
        {"label": shape_label.strip(), "justification": justification.strip()},
        traits,
    )
