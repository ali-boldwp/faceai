from typing import Dict, List, Optional
import os
from app.services.image import image_to_data_url

from app.models.schemas import (
    TraitEvidence,
    TraitAnalysis,
    SectionAnalysis,
    FullFaceAnalysis,
)

from app.face_features.eyebrows import classify_eyebrow_traits


def _build_face_shape_section(
    base_dir: str,
    ID: str,
    base_shape: str,
    romanian_label: str,
    earring_tip: Optional[str],
    hairline_shape: Optional[str],
    base_shape_measurements_used: Dict[str, float],
    base_shape_rule: Optional[str] = None,
    hairline_image_url: Optional[str] = None,
) -> SectionAnalysis:
    traits: List[TraitAnalysis] = []

    print ( base_shape_measurements_used )

    # Primary face shape trait
    traits.append(
        TraitAnalysis(
            name=f"Primary Face Shape: {base_shape}",
            present=True,
            confidence=1.0,
            explanation=(
                f"Face classified as '{base_shape}' based on forehead, "
                f"cheekbone and jaw widths correlated with face height. "
                f"Romanian label: {romanian_label}."
            ),
            evidence=TraitEvidence(
                # measurements_used=[
                #     "forehead_width",
                #     "cheekbone_width",
                #     "jaw_width",
                #     "face_height",
                # ],
                image_url=image_to_data_url( os.path.join(base_dir, "face_type_abcd.png") ),
                # image_url=os.path.join(base_dir, "face_type_abcd.png"),
                measurements_used=base_shape_measurements_used,
                notes=(
                    f"Rule used: {base_shape_rule or 'prototype fallback'}; "
                    "Classification derived from 'Tehnica de citire a feței' "
                    "face-shape rules."
                )
            ),
        )
    )

    # Earring tip from your original logic
    # if earring_tip:
    #     traits.append(
    #         TraitAnalysis(
    #             name="Earring Recommendation",
    #             present=True,
    #             confidence=1.0,
    #             explanation=earring_tip,
    #             evidence=TraitEvidence(notes="Original earring style tip for this face shape."),
    #         )
    #     )

    # Hairline as an influencing trait
    if hairline_shape:
        traits.append(
            TraitAnalysis(
                name=f"Hairline Shape: {hairline_shape}",
                present=True,
                confidence=1.0,
                explanation=(
                    "Hairline shape influences apparent face shape — "
                    "for example, some diamond faces with V-shape or rounded hairline "
                    "are reclassified as heart-shaped."
                ),
                evidence=TraitEvidence(
                    image_url=hairline_image_url,
                    notes="Hairline extracted via BiSeNet segmentation on the front image.",
                ),
            )
        )

    return SectionAnalysis(
        section="Face Shape",
        translation="Forma feței",
        traits=traits,
    )


def _build_eyebrows_section(
    eyebrow_shape_info,
    eyebrow_traits,
) -> SectionAnalysis:
    traits = []

    # Main morphology label (this is what React auto-select will match)
    traits.append(
        TraitAnalysis(
            name=eyebrow_shape_info["label"],
            present=True,
            confidence=1.0,
            explanation=eyebrow_shape_info["justification"],
            evidence=TraitEvidence(
                measurements_used={"interocular_distance": None},
                landmark_indices=[],
                image_url=None,
                notes="Derived from eyebrow and eye landmarks.",
            ),
            status="computed",
        )
    )

    # Psychological / meaning traits
    for t in eyebrow_traits:
        traits.append(
            TraitAnalysis(
                name=t.get("meaning", ""),
                present=True,
                confidence=1.0,
                explanation=t.get("explanation", ""),
                evidence=TraitEvidence(
                    measurements_used={},
                    landmark_indices=[],
                    image_url=None,
                    notes=t.get("source", ""),
                ),
                status="computed",
            )
        )

    return SectionAnalysis(
        section="Eyebrows",
        translation="Sprâncenele",
        traits=traits,
    )

def _build_forehead_section(
    forehead_shape: Dict,
    forehead_traits: List[Dict],
) -> SectionAnalysis:
    traits: List[TraitAnalysis] = []

    # Main morphological label
    traits.append(
        TraitAnalysis(
            name=forehead_shape.get("label", "Forehead"),
            present=True,
            confidence=1.0,
            explanation=forehead_shape.get("justification", ""),
            evidence=TraitEvidence(
                measurements_used={}, # ["forehead_width", "jaw_width"]
                notes="Based on forehead_width / jaw_width ratio.",
            ),
        )
    )

    # Psychological traits from the forehead classifier
    for t in forehead_traits:
        traits.append(
            TraitAnalysis(
                name=t.get("meaning", "Forehead trait"),
                present=True,
                confidence=1.0,
                explanation=t.get("explanation", ""),
                evidence=TraitEvidence(
                    notes=t.get("source", "Tehnica de citire a feței.docx"),
                ),
            )
        )

    return SectionAnalysis(
        section="Forehead",
        translation="Fruntea",
        traits=traits,
    )


def _build_eyes_section(
    eyes_shape: Dict,
    eye_traits: List[Dict],
) -> SectionAnalysis:
    traits: List[TraitAnalysis] = []

    # Main morphological eyes label
    traits.append(
        TraitAnalysis(
            name=eyes_shape.get("label", "Eyes"),
            present=True,
            confidence=1.0,
            explanation=eyes_shape.get("justification", ""),
            evidence=TraitEvidence(
                measurements_used={
                    "interocular_distance": None,
                    "eye_width_left": None,
                    "eye_width_right": None,
                },
                notes="Aspect ratio and tilt computed from eye landmarks.",
            ),
        )
    )

    # Psychological traits derived from eyes
    for t in eye_traits:
        traits.append(
            TraitAnalysis(
                name=t.get("meaning", "Eye trait"),
                present=True,
                confidence=1.0,
                explanation=t.get("explanation", ""),
                evidence=TraitEvidence(
                    notes=t.get("source", "Tehnica de citire a feței.docx"),
                ),
            )
        )

    return SectionAnalysis(
        section="Eyes",
        translation="Ochii",
        traits=traits,
    )



def build_full_analysis(
    base_dir: str,
    ID: str,
    base_shape: str,
    base_shape_measurements_used: Dict[str, float],
    romanian_label: str,
    earring_tip: Optional[str],
    hairline_shape: Optional[str],
    forehead_shape: Dict,
    forehead_traits: List[Dict],
    eyes_shape: Dict,
    eye_traits: List[Dict],
    measurements: Dict[str, float],
    base_shape_rule: Optional[str] = None,
    ratios: Dict[str, float] = None,
    angles: Dict[str, float] = None,
    landmarks: List[List[int]] = None,
    hair_mask_path: Optional[str] = None,
    landmarks_image_path: Optional[str] = None,
) -> FullFaceAnalysis:
    """Assemble a unified FullFaceAnalysis object from all partial classifiers.

    This keeps all your existing measurement and classification logic, but
    returns it in a single structured response that the frontend can use
    for all sections (Face Shape, Forehead, Eyes). Other sections from
    your documents (nose, cheeks, lips, ears, neck, skin/wrinkles) can be
    added later in the same pattern when automatic detectors are available.
    """

    sections: List[SectionAnalysis] = []

    sections.append(
        _build_face_shape_section(
            base_dir=base_dir,
            ID=ID,
            base_shape=base_shape,
            romanian_label=romanian_label,
            earring_tip=earring_tip,
            hairline_shape=hairline_shape,
            hairline_image_url=image_to_data_url( hair_mask_path ),
            base_shape_rule=base_shape_rule,
            base_shape_measurements_used=base_shape_measurements_used
        )
    )

    sections.append(
        _build_forehead_section(
            forehead_shape=forehead_shape,
            forehead_traits=forehead_traits,
        )
    )

    sections.append(
        _build_eyes_section(
            eyes_shape=eyes_shape,
            eye_traits=eye_traits,
        )
    )

    eyebrow_shape_info, eyebrow_traits = classify_eyebrow_traits(landmarks)
    sections.append(_build_eyebrows_section(eyebrow_shape_info, eyebrow_traits))


    return FullFaceAnalysis(
        ID=ID,
        primary_shape=base_shape,
        romanian_label=romanian_label,
        earring_tip=earring_tip,
        hairline_shape=hairline_shape,
        sections=sections,
        measurements=measurements,
        ratios=ratios,
        angles=angles,
        landmarks=landmarks,
        hair_mask_url=hair_mask_path,
        landmarks_image_url=landmarks_image_path,
    )
