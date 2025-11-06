import numpy as np
import math

def distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def all_measurements(landmarks):
    print("landmarks" , landmarks)
    try:
        # MediaPipe landmark index mapping
        return {
            "forehead_width": distance(landmarks[70], landmarks[300]),  # approx lateral forehead
            "jaw_width": distance(landmarks[234], landmarks[454]),      # Go-Go
            "cheekbone_width": distance(landmarks[93], landmarks[323]), # Zy-Zy
            "face_height": distance(landmarks[10], landmarks[152]),     # Tr-Gn

            "nose_width": distance(landmarks[94], landmarks[331]),      # Al-Al
            "nose_height": distance(landmarks[6], landmarks[195]),      # Nasion-Subnasale

            "mouth_width": distance(landmarks[61], landmarks[291]),     # Ch-Ch
            "mouth_height": distance(landmarks[13], landmarks[14]),     # Ls-Li

            "interocular_distance": distance(landmarks[168], landmarks[197]),  # En-En
            "eye_width_left": distance(landmarks[33], landmarks[133]),  # Ex-En left
            "eye_width_right": distance(landmarks[362], landmarks[263]) # Ex-En right
        }
    except Exception as e:
        print(f"[ERROR] Failed to calculate facial measurements: {e}")
        return {}

def all_ratios(measurements):
    ratios = {}
    fw = measurements.get("forehead_width", 0)
    jw = measurements.get("jaw_width", 0)
    cbw = measurements.get("cheekbone_width", 0)
    fh = measurements.get("face_height", 0)
    nw = measurements.get("nose_width", 0)
    nh = measurements.get("nose_height", 0)

    if fw and jw:
        ratios["forehead_over_jaw"] = fw / jw
    if cbw:
        ratios["face_height_over_cheekbones"] = fh / cbw if fh else 0
        ratios["jaw_over_cheekbones"] = jw / cbw if cbw else 0
    if fw:
        ratios["cheekbones_over_forehead"] = cbw / fw if cbw else 0
    if nh:
        ratios["nasal_index"] = nw / nh

    return {k: float(v) for k, v in ratios.items()}

def angle_between(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = max(min(cosang, 1.0), -1.0)
    ang = np.degrees(np.arccos(cosang))
    return ang

def all_angles(landmarks):

    print( len(landmarks) )

    if len(landmarks) < 455:
        raise ValueError("Expected MediaPipe 468-point landmarks, but got fewer points.")

    angles = {}

    angles["left_jaw_angle"] = angle_between(
        landmarks[234], landmarks[454], landmarks[152]
    )
    angles["right_jaw_angle"] = angle_between(
        landmarks[454], landmarks[234], landmarks[152]
    )

    return {k: float(v) for k, v in angles.items()}
