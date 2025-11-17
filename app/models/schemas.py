from pydantic import BaseModel, Field
from typing import List

class ImageURLs(BaseModel):
    front_image_url: str
    side_image_url: str

class Measurements(BaseModel):
    forehead_width: float
    jaw_width: float
    cheekbone_width: float
    face_height: float
    nose_width: float
    nose_height: float
    mouth_width: float
    mouth_height: float
    interocular_distance: float
    eye_width_left: float
    eye_width_right: float


class Landmark(BaseModel):
    x: float
    y: float
    z: float | None = None  

class FaceLandmarkRequest(BaseModel):
    face_landmarks: List[List[int]]
