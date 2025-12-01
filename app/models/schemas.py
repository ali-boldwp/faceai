from pydantic import BaseModel, Field
from typing import List, Dict, Optional

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


class TraitEvidence(BaseModel):
    measurements_used: Optional[Dict[str, float | None]] = None
    landmark_indices: List[int] = []
    image_url: Optional[str] = None
    notes: Optional[str] = None


class TraitAnalysis(BaseModel):
    name: str
    present: bool
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    explanation: Optional[str] = None
    evidence: Optional[TraitEvidence] = None
    status: str = Field(
        "computed",
        description="computed | not_supported | manual_only | not_detected"
    )


class SectionAnalysis(BaseModel):
    section: str
    translation: str
    traits: List[TraitAnalysis]


class FullFaceAnalysis(BaseModel):
    primary_shape: str
    romanian_label: str
    earring_tip: Optional[str] = None
    hairline_shape: Optional[str] = None

    sections: List[SectionAnalysis]

    measurements: Dict[str, float]
    ratios: Dict[str, float]
    angles: Dict[str, float]

    landmarks: List[List[int]]
    hair_mask_url: Optional[str] = None
    landmarks_image_url: Optional[str] = None
