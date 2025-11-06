from fastapi import FastAPI, HTTPException
from app.models.schemas import ImageURLs
from app.services.image_loader import url_to_image
from app.services.landmark_extractor import extract_face_landmarks

app = FastAPI(title="Face Morphology API")

@app.post("/analyze-face")
async def analyze_face(images: ImageURLs):
    try:
        front_img = url_to_image(images.front_image_url)
        side_img = url_to_image(images.side_image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    front_landmarks = extract_face_landmarks(front_img)
    if front_landmarks is None:
        raise HTTPException(status_code=422, detail="No face found in front image.")

    side_landmarks = extract_face_landmarks(side_img)
    if side_landmarks is None:
        raise HTTPException(status_code=422, detail="No face found in side image.")

    return {
        "front_landmarks": front_landmarks[:10], 
        "side_landmarks": side_landmarks[:10]
    }
    