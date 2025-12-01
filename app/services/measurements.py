import numpy as np
import math
import cv2
import os
from typing import Dict, Optional


def distance(p1, p2) -> float:
    """Euclidean distance between two 2D points."""
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


MEASUREMENT_LANDMARK_MAP: Dict[str, tuple[int, int]] = {
    "cheekbone_width": (123, 352),   # cheeks (numeric, but overridden below)
    "jaw_width": (58, 152),          # numeric jaw width (unchanged)
    "nose_width": (94, 331),
    "nose_height": (6, 195),
    "mouth_width": (61, 291),
    "mouth_height": (13, 14),
    "interocular_distance": (168, 197),
    "eye_width_left": (33, 133),
    "eye_width_right": (362, 263),
}


def all_measurements(
    landmarks: np.ndarray,
    front_path: Optional[str] = None,
    hairline_y: Optional[int] = None,      # optional fallback, main source = mask
    hair_mask_img: Optional[np.ndarray] = None,
    ID: str = "testing",
    draw: bool = True,
) -> Dict[str, float]:
    """
    a = face_height: vertical from HAIRLINE CENTER BOTTOM POINT (from mask)
                     to chin center (landmark 152)
    b = forehead_width: horizontal between forehead landmarks 103–332
    c = cheek width: horizontal between cheek landmarks 123–352
    d = jaw diagonal: from jaw point 215 to chin center (landmark 152)

    Measurements and drawing use the SAME endpoints.
    """
    measurements: Dict[str, float] = {}

    try:
        # ---------- load image (for size & drawing) ----------
        img = None
        if front_path:
            img = cv2.imread(front_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
        else:
            img_h = int(np.max(landmarks[:, 1])) + 1
            img_w = int(np.max(landmarks[:, 0])) + 1

        # ---------- core landmarks ----------
        # chin center – always landmark 152
        chin = landmarks[152]
        chin_x, chin_y = int(chin[0]), int(chin[1])

        # forehead points for b (103–332)
        fL = landmarks[103]
        fR = landmarks[332]
        fL_x, fL_y = int(fL[0]), int(fL[1])
        fR_x, fR_y = int(fR[0]), int(fR[1])
        b_y = int((fL_y + fR_y) / 2)
        p_b1 = (fL_x, b_y)
        p_b2 = (fR_x, b_y)

        # cheek points for c (123–352)
        cheek_L = landmarks[123]
        cheek_R = landmarks[352]
        chL_x, chL_y = int(cheek_L[0]), int(cheek_L[1])
        chR_x, chR_y = int(cheek_R[0]), int(cheek_R[1])
        c_y = int((chL_y + chR_y) / 2)
        p_c1 = (chL_x, c_y)
        p_c2 = (chR_x, c_y)

        # jaw start for d: ***215*** → chin center (152)
        d_start = landmarks[215]
        d_start_x, d_start_y = int(d_start[0]), int(d_start[1])
        p_d1 = (d_start_x, d_start_y)
        p_d2 = (chin_x, chin_y)

        # ---------- HAIRLINE CENTER BOTTOM from MASK ----------
        hair_x = None
        hair_y = None

        if hair_mask_img is not None:
            mask = hair_mask_img
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            mh, mw = mask.shape[:2]
            if (mh, mw) != (img_h, img_w):
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            ys, xs = np.where(mask > 0)
            if ys.size > 0:
                min_x, max_x = int(xs.min()), int(xs.max())
                hair_x = int((min_x + max_x) / 2)
                col = mask[:, hair_x]
                ys_col = np.where(col > 0)[0]
                if ys_col.size > 0:
                    hair_y = int(ys_col[-1])
                else:
                    hair_y = int(ys.max())

        # fallback if mask failed
        if hair_x is None or hair_y is None:
            hair_x = chin_x
            if hairline_y is not None and 0 <= hairline_y < img_h:
                hair_y = int(hairline_y)
            else:
                hair_y = int(landmarks[10][1])

        # a: from orange-dot hairline down to chin at same x
        p_a1 = (hair_x, hair_y)
        p_a2 = (hair_x, chin_y)

        # ---------- numeric measurements (exact endpoints) ----------
        measurements["face_height"] = distance(p_a1, p_a2)        # a
        measurements["forehead_width"] = distance(p_b1, p_b2)     # b
        measurements["cheekbone_width"] = distance(p_c1, p_c2)    # c
        measurements["jaw_diagonal"] = distance(p_d1, p_d2)       # d

        # jaw width using 58–152 (numeric only, unchanged)
        jaw_L = landmarks[58]
        jaw_R = landmarks[152]
        measurements["jaw_width"] = distance(jaw_L, jaw_R)

        # other pairwise distances
        for name, (i1, i2) in MEASUREMENT_LANDMARK_MAP.items():
            if name in ("jaw_width", "forehead_width", "cheekbone_width"):
                continue
            if i1 < len(landmarks) and i2 < len(landmarks):
                measurements[name] = distance(landmarks[i1], landmarks[i2])

        # ---------- draw a/b/c/d diagram ----------
        if draw and img is not None and front_path:
            diagram = img.copy()
            COLOR = (255, 0, 0)
            thickness = 2

            # a
            cv2.line(diagram, p_a1, p_a2, COLOR, thickness, cv2.LINE_AA)
            cv2.putText(
                diagram,
                "a",
                (p_a2[0] + 5, (p_a1[1] + p_a2[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR,
                2,
                cv2.LINE_AA,
            )

            # b
            cv2.line(diagram, p_b1, p_b2, COLOR, thickness, cv2.LINE_AA)
            cv2.putText(
                diagram,
                "b",
                ((p_b1[0] + p_b2[0]) // 2, b_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR,
                2,
                cv2.LINE_AA,
            )

            # c
            cv2.line(diagram, p_c1, p_c2, COLOR, thickness, cv2.LINE_AA)
            cv2.putText(
                diagram,
                "c",
                ((p_c1[0] + p_c2[0]) // 2, c_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR,
                2,
                cv2.LINE_AA,
            )

            # d (now 215 → chin center)
            cv2.line(diagram, p_d1, p_d2, COLOR, thickness, cv2.LINE_AA)
            mid_dx = (p_d1[0] + p_d2[0]) // 2
            mid_dy = (p_d1[1] + p_d2[1]) // 2
            cv2.putText(
                diagram,
                "d",
                (mid_dx, mid_dy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR,
                2,
                cv2.LINE_AA,
            )

            # legend
            a_len = measurements["face_height"]
            b_len = measurements["forehead_width"]
            c_len = measurements["cheekbone_width"]
            d_len = measurements["jaw_diagonal"]

            legend_lines = [
                f"a: {a_len:.1f}px",
                f"b: {b_len:.1f}px",
                f"c: {c_len:.1f}px",
                f"d: {d_len:.1f}px",
            ]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1

            line_sizes = [
                cv2.getTextSize(t, font, font_scale, text_thickness)[0]
                for t in legend_lines
            ]
            text_w = max(w for (w, h) in line_sizes)
            text_h = max(h for (w, h) in line_sizes)
            line_spacing = text_h + 4
            padding = 8
            box_w = text_w + 2 * padding
            box_h = len(legend_lines) * line_spacing + 2 * padding

            img_h2, img_w2 = diagram.shape[:2]
            box_x2 = img_w2 - 10
            box_x1 = box_x2 - box_w
            box_y1 = 10
            box_y2 = box_y1 + box_h

            cv2.rectangle(
                diagram,
                (box_x1, box_y1),
                (box_x2, box_y2),
                (255, 255, 255),
                thickness=-1,
            )

            y_text = box_y1 + padding + text_h
            for t in legend_lines:
                cv2.putText(
                    diagram,
                    t,
                    (box_x1 + padding, y_text),
                    font,
                    font_scale,
                    (0, 0, 0),
                    text_thickness,
                    cv2.LINE_AA,
                )
                y_text += line_spacing

            dir_name = os.path.join("tmp", str(ID))
            os.makedirs(dir_name, exist_ok=True)
            _, ext = os.path.splitext(front_path)
            out_path = os.path.join(dir_name, f"face_type_abcd.png")
            cv2.imwrite(out_path, diagram)
            print(f"[DEBUG] Face-type diagram (a,b,c,d) saved to {out_path}")

        return measurements

    except Exception as e:
        print(f"[ERROR] Failed to calculate facial measurements: {e}")
        return {}



def all_ratios(measurements: Dict[str, float]) -> Dict[str, float]:
    ratios = {}
    fw = measurements.get("forehead_width", 0.0)
    jw = measurements.get("jaw_width", 0.0)
    cbw = measurements.get("cheekbone_width", 0.0)
    fh = measurements.get("face_height", 0.0)
    nw = measurements.get("nose_width", 0.0)
    nh = measurements.get("nose_height", 0.0)

    if fw and jw:
        ratios["forehead_over_jaw"] = fw / jw
    if cbw:
        ratios["face_height_over_cheekbones"] = fh / cbw if fh else 0.0
        ratios["jaw_over_cheekbones"] = jw / cbw if cbw else 0.0
    if fw:
        ratios["cheekbones_over_forehead"] = cbw / fw if cbw else 0.0
    if nh:
        ratios["nasal_index"] = nw / nh

    return {k: float(v) for k, v in ratios.items()}


def angle_between(p1, p2, p3) -> float:
    v1 = np.array(p1, dtype=float) - np.array(p2, dtype=float)
    v2 = np.array(p3, dtype=float) - np.array(p2, dtype=float)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = max(min(cosang, 1.0), -1.0)
    ang = np.degrees(np.arccos(cosang))
    return float(ang)


def all_angles(landmarks: np.ndarray) -> Dict[str, float]:
    print(len(landmarks))

    if len(landmarks) < 455:
        raise ValueError(
            "Expected MediaPipe 468-point landmarks, but got fewer points."
        )

    angles = {}
    angles["left_jaw_angle"] = angle_between(
        landmarks[234], landmarks[454], landmarks[152]
    )
    angles["right_jaw_angle"] = angle_between(
        landmarks[454], landmarks[234], landmarks[152]
    )
    return {k: float(v) for k, v in angles.items()}
