# import cv2
# import mediapipe as mp

# mp_face_mesh = mp.solutions.face_mesh

# def extract_face_landmarks(image):
#     with mp_face_mesh.FaceMesh(
#         static_image_mode=True,
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5
#     ) as face_mesh:

#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_image)

#         if results.multi_face_landmarks:
#             landmarks = results.multi_face_landmarks[0].landmark
#             return [(lm.x, lm.y, lm.z) for lm in landmarks]
#         else:
#             return None


import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def extract_face_landmarks(image, min_face_conf=0.8, min_mesh_conf=0.8):
    """
    Returns (landmarks, face_score) or (None, 0.0) if not confident enough.

    landmarks are still normalized (x,y,z in [0,1] for x,y).
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---------- 1) run face detection with a high threshold ----------
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_face_conf
    ) as face_det:

        det_results = face_det.process(rgb_image)
        if not det_results.detections:
            return None, 0.0

        detection = det_results.detections[0]
        face_score = float(detection.score[0])  # confidence of the face itself

    # Optional: you can reject low-score faces here
    if face_score < min_face_conf:
        return None, face_score

    # ---------- 2) run FaceMesh for detailed landmarks ----------
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_mesh_conf
    ) as face_mesh:

        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None, face_score

        landmarks = results.multi_face_landmarks[0].landmark
        landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks]

    return landmarks, face_score
