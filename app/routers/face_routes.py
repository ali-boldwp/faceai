from fastapi import FastAPI,APIRouter, HTTPException
from app.models.schemas import ImageURLs
from app.services.image_loader import url_to_image
from app.services.landmark_extraction import extract_face_landmarks
from app.services.measurements import all_measurements
from app.face_features.face_shape import classify_face_shape
import numpy as np
from app.services.bisenet import load_bisenet
from app.services.bisenet import preprocess
from app.services.bisenet import hair_mask
from app.services.bisenet import analyze_hairline
from app.services.bisenet import extract_metrics
import cv2
import os

router = APIRouter(prefix="/face", tags=["Face Analysis"])


def to_pixel_landmarks(landmarks_list, img_width, img_height):
    if hasattr(landmarks_list, "landmark"):
        landmarks_list = landmarks_list.landmark

    pts = []
    for lm in landmarks_list:
        if hasattr(lm, "x") and hasattr(lm, "y"):
            x = lm.x
            y = lm.y
            X = int(x * img_width)  if 0.0 <= x <= 1.0 else int(x)
            Y = int(y * img_height) if 0.0 <= y <= 1.0 else int(y)

        elif isinstance(lm, dict) and ("x" in lm and "y" in lm):
            x = lm["x"]; y = lm["y"]
            X = int(x * img_width)  if 0.0 <= x <= 1.0 else int(x)
            Y = int(y * img_height) if 0.0 <= y <= 1.0 else int(y)

        elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 2:
            x = float(lm[0]); y = float(lm[1])
            X = int(x * img_width)  if 0.0 <= x <= 1.0 else int(x)
            Y = int(y * img_height) if 0.0 <= y <= 1.0 else int(y)

        else:
            raise ValueError(f"Unsupported landmark format: {type(lm)} -> {lm}")

        pts.append([X, Y])

    if not pts:
        raise ValueError("Empty landmarks list after conversion.")

    arr = np.asarray(pts, dtype=np.int32)
    return arr

@router.post("/shape")
async def analyze_face(images: ImageURLs):
    try:
        front_img, front_path = url_to_image(images.front_image_url, prefix="front")
        side_img = url_to_image(images.side_image_url) if getattr(images, "side_image_url", None) else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    print("Front image saved as:", front_path)

    face_landmarks = extract_face_landmarks(front_img)
    if face_landmarks is None:
        raise HTTPException(status_code=422, detail="No face found in front image.")
    
    h, w = front_img.shape[:2]

    try:
        landmarks_px = to_pixel_landmarks(face_landmarks, w, h)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Landmark conversion error: {ve}")
    
    try:
        measurements = all_measurements(landmarks_px)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Measurement error: {e}")

    net = load_bisenet()
    tensor, rgb = preprocess(front_path)
    hair_mask_img, parsing = hair_mask(net, tensor)
    cv2.imwrite('tmp/hair_mask.png', hair_mask_img * 255)

    hairline_shape, hairline_y = analyze_hairline(hair_mask_img, rgb)

    print("Detected Hairline Shape:", hairline_shape)
    a, b, c, d = extract_metrics(parsing, hairline_y)
    print(f"Measurements → a={a}, b={b}, c={c}, d={d}")
    face_shape, romanian_label, earring_tip = classify_face_type(a, b, c, d)

    jaw = 2 * d



    # Classify shape using the measurements directly (avoids recomputing)
    try:
        primary_shape, why, debug, attrs = classify_face_shape(
            {},  # no landmarks needed since we pass explicit measurements
            measurements={
                "forehead_width": measurements.get("forehead_width"),
                "face_height": measurements.get("face_height"),
                "cheekbone_width": measurements.get("cheekbone_width"),
                "jaw_width": measurements.get("jaw_width"),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    # Optional: process side image later if needed
    # side_landmarks = None
    # if side_img is not None:
    #     side_landmarks = extract_face_landmarks(side_img)
    #     if side_landmarks is None:
    #         raise HTTPException(status_code=422, detail="No face found in side image.")

    return {
        "shape": primary_shape,
        "attributes": attrs,  # e.g., ["Wide Face"] / ["Narrow Face"] / []
        "justification": why,
        "ratios": {
            "R_hw": debug.get("R_hw"),
            "R_fc": debug.get("R_fc"),
            "R_jc": debug.get("R_jc"),
            "R_fj": debug.get("R_fj"),
        },
        "measurements": {
            "forehead_width": measurements.get("forehead_width"),
            "face_height": measurements.get("face_height"),
            "cheekbone_width": measurements.get("cheekbone_width"),
            "jaw_width": measurements.get("jaw_width"),
        },
        # "front_landmarks_px": landmarks_px.tolist(),  # uncomment if you want to return the points
        # "side_landmarks": side_landmarks[:10] if side_landmarks else None,
    }
