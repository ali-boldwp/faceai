import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def extract_face_landmarks(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            return [(lm.x, lm.y, lm.z) for lm in landmarks]
        else:
            return None
