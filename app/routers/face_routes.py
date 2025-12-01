from fastapi import FastAPI, APIRouter, Body, HTTPException
from app.models.schemas import ImageURLs
from app.models.schemas import Measurements
from app.models.schemas import FaceLandmarkRequest

from app.services.image_loader import url_to_image
from app.services.landmark_extraction import extract_face_landmarks
from app.services.measurements import all_measurements
from app.services.measurements import all_ratios, all_angles
from app.face_features.face_shape import classify_face_type  
from app.face_features.forhead import classify_forehead_traits
from app.face_features.eyes import classify_eye_traits
from app.face_features.full_analysis import build_full_analysis
from app.utils.image_utils import draw_landmarks_image
from app.models.schemas import FullFaceAnalysis
from app.services.visualization import draw_landmarks_with_indices
import numpy as np
from app.services.bisenet import load_bisenet
from app.services.bisenet import preprocess
from app.services.bisenet import hair_mask
from app.services.bisenet import analyze_hairline
from app.services.bisenet import extract_metrics
import cv2
import os
from pydantic import BaseModel
from typing import Optional
import json



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

class ImageURLs(BaseModel):
    front_image_url: str
    side_image_url: Optional[str] = None

    
class ChatID(BaseModel):
    ID: str

@router.post("/shape")
async def analyze_face_shape_route(raw: str = Body(..., media_type="text/plain")):
    data = json.loads(raw)
    images = ImageURLs(**data)
    try:
        front_img, front_path = url_to_image(images.front_image_url, prefix="front")
        side_img = url_to_image(images.side_image_url) if getattr(images, "side_image_url", None) else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    print("Front image saved as:", front_path)

    face_landmarks = extract_face_landmarks(front_img, min_face_conf=0.9, min_mesh_conf=0.9)
    if face_landmarks is None:
        raise HTTPException(status_code=422, detail="No face found in front image.")

    h, w = front_img.shape[:2]
    landmarks = np.array([
        [int(lm[0] * w), int(lm[1] * h)]
        for lm in face_landmarks
    ])

    try:
        landmarks_px = to_pixel_landmarks(face_landmarks, w, h)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Landmark conversion error: {ve}")

    try:
        measurements = all_measurements(landmarks_px)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Measurement error: {e}")

    a = measurements.get("forehead_width")
    b = measurements.get("face_height")
    c = measurements.get("cheekbone_width")
    jaw_width = measurements.get("jaw_width")

    if None in (a, b, c, jaw_width):
        raise HTTPException(
            status_code=500,
            detail="Missing key measurements (forehead_width, face_height, cheekbone_width, jaw_width).",
        )

    net = load_bisenet()
    tensor, rgb = preprocess(front_path)
    hair_mask_img, parsing = hair_mask(net, tensor)
    cv2.imwrite('tmp/hair_mask.png', hair_mask_img * 255)

    hairline_shape, hairline_y = analyze_hairline(hair_mask_img, rgb)

    print("Detected Hairline Shape:", hairline_shape)

    try:
        a_h, b_h, c_h, d_h = extract_metrics(parsing, hairline_y)
        print(f"[DEBUG] BiSeNet metrics (NOT used for shape) → a={a_h}, b={b_h}, c={c_h}, d={d_h}")
    except Exception as e:
        print("[WARN] extract_metrics failed:", e)

    base_shape, romanian_label, earring_tip, base_rule = classify_face_type(
        forehead_width=a,
        face_height=b,
        cheekbone_width=c,
        jaw_width=jaw_width,
    )

    print("[DEBUG] Base face shape:", base_shape)

    face_shape = base_shape  # start from base

    if hairline_shape in ["V-shape", "Rounded"]:
        if base_shape == "Diamond Face":
            face_shape = "Heart-Shaped Face"
            romanian_label = "Inimă / Sangvin – Venus"
            earring_tip = "Triangulars, chandeliers, teardrops. Avoid tiny studs."

        elif base_shape in ["Square Face", "Oval Face"]:
            if a > jaw_width and c >= a * 0.9:  # forehead ≥ jaw, strong cheeks
                face_shape = "Heart-Shaped Face"
                romanian_label = "Inimă / Sangvin – Venus"
                earring_tip = "Triangulars, chandeliers, teardrops. Avoid tiny studs."

    print("[DEBUG] Final face shape after hairline refinement:", face_shape)

    return {
        "primary_shape": face_shape,
        "classification": {
            "romanian_label": romanian_label,
            "earring_tip": earring_tip,
        },
        "hairline_shape": hairline_shape,
        "measurements": measurements,
        "face_landmarks": landmarks.tolist()
    }

@router.post("/forehead")
async def analyze_forehead_route(measurements: Measurements):
    feature_shapes = {}        
    psychological_traits = {} 

    feature_shapes["forehead"], psychological_traits["forehead"] = classify_forehead_traits(measurements)
    
    return {
        "forehead": feature_shapes["forehead"]["label"],
        "justification": feature_shapes["forehead"]["justification"],
        "measurements": measurements.dict(),
        "traits": psychological_traits["forehead"]
    }



@router.post("/eyes")
async def analyze_eyes_route(data: FaceLandmarkRequest):
    landmarks = data.face_landmarks
    feature_shapes = {}        
    psychological_traits = {} 

    feature_shapes["eyes"], psychological_traits["eyes"] = classify_eye_traits(landmarks)
    
    return {
        "eyes": feature_shapes["eyes"]["label"],
        "justification": feature_shapes["eyes"]["justification"],
        "traits": psychological_traits["eyes"]
    }


@router.post("/full", response_model=FullFaceAnalysis)
async def analyze_full_face_route(raw: str = Body(..., media_type="text/plain")):
    """
    Unified endpoint that returns face shape + forehead + eyes + debug info in one response.

    Expects the same payload as /face/shape:
    {
      "front_image_url": "...",
      "side_image_url": "..." // optional
    }
    """
    data = json.loads(raw)
    images = ImageURLs(**data)
    chat_id = ChatID(**data).ID

    # 🔹 Make sure per-chat tmp dir exists
    base_dir = os.path.join("tmp", str(chat_id))
    os.makedirs(base_dir, exist_ok=True)

    # 1) Load front image
    try:
        front_img, front_path = url_to_image(images.front_image_url, prefix="front_full")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading front image: {e}")

    h, w = front_img.shape[:2]

    # 2) Landmarks via MediaPipe
    face_landmarks, face_score = extract_face_landmarks(front_img, min_face_conf=0.7, min_mesh_conf=0.7)
    if face_landmarks is None:
        raise HTTPException(
            status_code=422,
            detail=f"No reliable face found in front image (score={face_score:.2f}).",
        )

    try:
        landmarks_px = to_pixel_landmarks(face_landmarks, w, h)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Landmark conversion error: {ve}")

    # 3) Hairline via BiSeNet (we need this BEFORE all_measurements)
    net = load_bisenet()
    tensor, rgb = preprocess(front_path)
    hair_mask_img, parsing = hair_mask(net, tensor)

    hair_mask_path = os.path.join(base_dir, "hair_mask_full.png")
    cv2.imwrite(hair_mask_path, hair_mask_img * 255)

    hairline_shape, hairline_y = analyze_hairline(hair_mask_img, rgb)

    # 4) Measurements, ratios, angles (now using hairline_y + hair_mask_img)
    try:
        measurements = all_measurements(
            landmarks_px,
            front_path=front_path,
            hairline_y=hairline_y,
            hair_mask_img=hair_mask_img,
            ID=chat_id,
            draw=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Measurement error: {e}")

    ratios = all_ratios(measurements)
    try:
        angles = all_angles(landmarks_px)
    except Exception as e:
        # Angles are mainly used for jawline analysis; don't fail the whole request if it breaks
        angles = {}
        print("[WARN] all_angles failed in /face/full:", e)

    a = measurements.get("forehead_width")
    b = measurements.get("face_height")
    c = measurements.get("cheekbone_width")
    jaw_width = measurements.get("jaw_width")

    if None in (a, b, c, jaw_width):
        raise HTTPException(
            status_code=500,
            detail="Missing key measurements (forehead_width, face_height, cheekbone_width, jaw_width).",
        )

    # 5) Face shape from your existing classifier
    base_shape, romanian_label, earring_tip, base_shape_rule = classify_face_type(
        forehead_width=a,
        face_height=b,
        cheekbone_width=c,
        jaw_width=jaw_width,
    )

    # 6) Forehead & eyes traits
    measurements_model = Measurements(**measurements)
    forehead_shape, forehead_traits = classify_forehead_traits(measurements_model)

    # For eyes we reuse the pixel landmarks (list-of-[x,y])
    eyes_shape, eye_traits = classify_eye_traits(landmarks_px.tolist())

    # 7) Landmarks overlay image (points-only debug)
    landmarks_image_path = os.path.join(base_dir, "landmarks_full.png")
    draw_landmarks_image(front_img, landmarks_px, landmarks_image_path)

    # 7b) Landmarks WITH ALL indices
    landmarks_numbered_image_path = os.path.join(base_dir, "landmarks_full_numbered.png")
    draw_landmarks_with_indices(front_img, landmarks_px, landmarks_numbered_image_path, scale=1.5)

    # 8) Build unified analysis object
    analysis = build_full_analysis(
        base_dir=base_dir,
        ID=chat_id,
        base_shape=base_shape,
        base_shape_measurements_used={
            "forehead_width": a,
            "face_height": measurements.get("face_height"),
            "cheekbone_width": measurements.get("cheekbone_width"),
            "jaw_width": measurements.get("jaw_width"),
        },
        base_shape_rule=base_shape_rule,
        romanian_label=romanian_label,
        earring_tip=earring_tip,
        hairline_shape=hairline_shape,
        forehead_shape=forehead_shape,
        forehead_traits=forehead_traits,
        eyes_shape=eyes_shape,
        eye_traits=eye_traits,
        measurements={k: float(v) for k, v in measurements.items()},
        ratios=ratios,
        angles=angles,
        landmarks=landmarks_px.tolist(),
        hair_mask_path=hair_mask_path,
        landmarks_image_path=landmarks_image_path,
    )

    return analysis

