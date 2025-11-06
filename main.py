from fastapi import FastAPI, HTTPException
from app.models.schemas import ImageURLs
from app.services.image_loader import url_to_image
from app.services.landmark_extraction import extract_face_landmarks
from app.services.measurements import all_measurements
from app.face_features.face_shape import classify_face_shape
import numpy as np

app = FastAPI(title="Face Morphology API")


def to_pixel_landmarks(landmarks_list, img_width, img_height):
    """
    Convert a heterogeneous list of landmarks to integer pixel coordinates (Nx2).
    Accepts:
      - MediaPipe-like objects with .x, .y in [0,1]
      - dicts with 'x','y' (normalized or absolute)
      - tuples/lists (x,y) already absolute
    """
    pts = []
    for lm in landmarks_list:
        if hasattr(lm, "x") and hasattr(lm, "y"):
            # MediaPipe NormalizedLandmark (0..1)
            x = int(lm.x * img_width)
            y = int(lm.y * img_height)
        elif isinstance(lm, dict) and ("x" in lm and "y" in lm):
            # Could be normalized or absolute; treat <=1.0 as normalized
            x_val = lm["x"]
            y_val = lm["y"]
            x = int(x_val * img_width) if 0.0 <= x_val <= 1.0 else int(x_val)
            y = int(y_val * img_height) if 0.0 <= y_val <= 1.0 else int(y_val)
        elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
            # Already absolute pixel coords
            x = int(lm[0])
            y = int(lm[1])
        else:
            raise ValueError(f"Unsupported landmark format: {type(lm)} -> {lm}")
        pts.append([x, y])

    if not pts:
        raise ValueError("Empty landmarks list after conversion.")

    return np.asarray(pts, dtype=np.int32)


@app.post("/analyze-face")
async def analyze_face(images: ImageURLs):
    try:
        front_img = url_to_image(images.front_image_url)
        # side image is optional; only load if provided
        side_img = url_to_image(images.side_image_url) if getattr(images, "side_image_url", None) else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Extract landmarks from the front image
    face_landmarks = extract_face_landmarks(front_img)
    if face_landmarks is None:
        raise HTTPException(status_code=422, detail="No face found in front image.")

    # Convert to pixel coordinates (Nx2)
    h, w = front_img.shape[:2]
    try:
        landmarks_px = to_pixel_landmarks(face_landmarks, w, h)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Landmark conversion error: {ve}")

    # Compute measurements (expects pixel coords)
    try:
        measurements = all_measurements(landmarks_px)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Measurement error: {e}")

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


@app.get("/")
async def checking():
    print("API is running")
    return {"message": "API is running"}
