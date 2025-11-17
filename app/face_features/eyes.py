
import numpy as np

def classify_eye_traits(landmarks):
    LEFT_EYE = [33, 159, 158, 157, 133, 145, 144, 153]
    RIGHT_EYE = [263, 386, 385, 384, 362, 374, 373, 380]

    left_eye = np.array([landmarks[i] for i in LEFT_EYE])
    right_eye = np.array([landmarks[i] for i in RIGHT_EYE])

    traits = []
    shape_label = ""
    justification = ""

    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def eye_tilt(eye):
        dx = eye[3][0] - eye[0][0]
        dy = eye[3][1] - eye[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    # Determine eye shape
    if avg_ear < 0.25:
        shape_label = "Narrow Eyes"
        justification = "Low eye aspect ratio (EAR < 0.25)."
        traits.append({
            "meaning": "Reserved, cautious",
            "explanation": "Narrow eyes suggest restraint and protectiveness.",
            "source": "Tehnica de citire a feței.docx - Ochi înguști"
        })
    elif 0.25 <= avg_ear <= 0.35:
        shape_label = "Almond Eyes"
        justification = "Medium eye aspect ratio (0.25 ≤ EAR ≤ 0.35)."
        traits.append({
            "meaning": "Emotionally balanced",
            "explanation": "Almond eyes reflect inner harmony and emotional sensitivity.",
            "source": "Tehnica de citire a feței.docx - Ochi migdalați"
        })
    else:
        shape_label = "Round Eyes"
        justification = "High eye aspect ratio (EAR > 0.35)."
        traits.append({
            "meaning": "Expressive and open",
            "explanation": "Round eyes are linked to curiosity and receptivity.",
            "source": "Tehnica de citire a feței.docx - Ochi rotunzi"
        })

    # Check tilt
    tilt = (eye_tilt(left_eye) + eye_tilt(right_eye)) / 2.0
    if tilt > 5:
        shape_label += ", Upturned Eyes"
        justification += " Tilt angle indicates upturned outer corners."
        traits.append({
            "meaning": "Optimistic",
            "explanation": "Upturned eyes are associated with positive disposition and adaptability.",
            "source": "Tehnica de citire a feței.docx - Ochi ridicați"
        })
    elif tilt < -5:
        shape_label += ", Downturned Eyes"
        justification += " Tilt angle indicates downturned outer corners."
        traits.append({
            "meaning": "Empathetic, reflective",
            "explanation": "Downturned eyes suggest depth of feeling and introspection.",
            "source": "Tehnica de citire a feței.docx - Ochi coborâți"
        })
    else:
        shape_label += ", Horizontal Eyes"
        justification += " Eye corners are horizontally aligned."
        traits.append({
            "meaning": "Stable, realistic",
            "explanation": "Horizontally aligned eyes reflect mental stability and pragmatism.",
            "source": "Tehnica de citire a feței.docx - Ochi drepți"
        })

    return {"label": shape_label.strip(), "justification": justification.strip()}, traits
