from typing import List, Tuple, Optional

from app.models.schemas import Measurements, TraitAnalysis, TraitEvidence


def _safe_get(meas: Measurements, name: str) -> Optional[float]:
    """Robust accessor for Measurements fields."""
    if hasattr(meas, name):
        return getattr(meas, name)
    if hasattr(meas, "dict"):
        return meas.dict().get(name)
    if hasattr(meas, "model_dump"):
        return meas.model_dump().get(name)
    return None


def classify_forehead_traits(measurements: Measurements) -> Tuple[dict, List[TraitAnalysis]]:
    """
    Classify forehead traits using both front and profile information when available.

    Returns
    -------
    forehead_shape : dict
        {
          "label": "...",              # main summary (used by build_full_analysis)
          "height_label": "...",       # optional detail
          "width_label": "...",
          "front_shape_label": "...",
          "profile_label": "..."
        }
    traits : List[TraitAnalysis]
        Individual traits (high/low, wide/narrow, profile shape, etc.).
    """
    traits: List[TraitAnalysis] = []

    # ---------- RAW VALUES ----------
    forehead_width = _safe_get(measurements, "forehead_width")
    cheekbone_width = _safe_get(measurements, "cheekbone_width")
    face_height_front = _safe_get(measurements, "face_height")

    # Profile heights / ratios
    h_tr_g = _safe_get(measurements, "forehead_height_tr_g_profile")
    h_tr_n = _safe_get(measurements, "forehead_height_tr_n_profile")
    h_tr_gn = _safe_get(measurements, "face_height_tr_gn_profile")
    ratio_tr_g = _safe_get(measurements, "forehead_height_ratio_tr_g_profile")
    ratio_tr_n = _safe_get(measurements, "forehead_height_ratio_tr_n_profile")

    # Profile slope
    slope_signed = _safe_get(measurements, "forehead_slope_signed_deg_profile")
    slope_abs = _safe_get(measurements, "forehead_slope_deg_profile")

    # Aspect ratio of the forehead (height vs width)
    forehead_height_for_aspect = None
    if h_tr_g is not None and h_tr_g > 0:
        forehead_height_for_aspect = h_tr_g
    elif face_height_front is not None and face_height_front > 0:
        forehead_height_for_aspect = face_height_front / 3.0

    aspect_ratio = None
    if forehead_width and forehead_width > 0 and forehead_height_for_aspect:
        aspect_ratio = forehead_height_for_aspect / forehead_width

    # Width ratio: forehead vs cheekbones
    width_ratio = None
    if forehead_width and cheekbone_width and cheekbone_width > 0:
        width_ratio = forehead_width / cheekbone_width

    # Height ratio: prefer profile ratio
    height_ratio = None
    if ratio_tr_g is not None and ratio_tr_g > 0:
        height_ratio = ratio_tr_g
    elif face_height_front and face_height_front > 0 and forehead_height_for_aspect:
        height_ratio = forehead_height_for_aspect / face_height_front

    # ---------- DISCRETE CATEGORIES ----------

    # 1) Height: high / low / average
    height_label = None
    height_conf = 0.6
    if height_ratio is not None:
        if height_ratio >= 0.38:
            height_label = "High Forehead / Frunte înaltă"
        elif height_ratio <= 0.30:
            height_label = "Low Forehead / Frunte joasă"
        else:
            height_label = "Average forehead height"
        height_conf = 0.9

    # 2) Width: wide / narrow / balanced
    width_label = None
    width_conf = 0.6
    if width_ratio is not None:
        if width_ratio >= 1.02:
            width_label = "Wide Forehead / Frunte lată"
        elif width_ratio <= 0.88:
            width_label = "Narrow Forehead / Frunte îngustă"
        else:
            width_label = "Balanced forehead width"
        width_conf = 0.9

    # 3) Overall front shape
    front_shape_label = None
    front_shape_conf = 0.6
    if aspect_ratio is not None and width_ratio is not None:
        if 0.9 <= aspect_ratio <= 1.1:
            front_shape_label = "Square Forehead / Frunte pătrată"
        elif aspect_ratio > 1.1 and width_ratio <= 1.0:
            front_shape_label = "Oval Forehead / Frunte ovală"
        elif aspect_ratio < 0.9 and width_ratio >= 0.95:
            front_shape_label = "Round Forehead / Frunte rotundă"
        else:
            front_shape_label = "Balanced forehead shape"
        front_shape_conf = 0.85

    # 4) Profile shape
    profile_label = None
    profile_conf = 0.5
    if slope_abs is not None:
        if slope_abs < 5.0:
            profile_label = "Straight Forehead / Frunte dreaptă"
        else:
            if slope_signed is not None:
                if slope_signed > 0:
                    profile_label = "Bulging Forehead / Frunte bombată"
                else:
                    profile_label = "Sloping Forehead / Frunte înclinată"
            else:
                profile_label = "Sloping forehead in profile"
        profile_conf = 0.85

    # ---------- HELPERS TO BUILD TRAITS ----------

    def _build_measurements_used(keys: list[str]) -> dict:
        """Convert a list of measurement names to {name: value} dict."""
        out = {}
        for key in keys:
            val = _safe_get(measurements, key)
            if val is not None:
                try:
                    out[key] = float(val)
                except (TypeError, ValueError):
                    continue
        return out

    def make_trait(
        name: str,
        present: bool,
        confidence: float,
        explanation: str,
        measurement_keys: list[str],
    ) -> TraitAnalysis:
        return TraitAnalysis(
            name=name,
            present=present,
            confidence=confidence,
            explanation=explanation,
            evidence=TraitEvidence(
                measurements_used=_build_measurements_used(measurement_keys),
                landmark_indices=[],
                image_url=None,
                notes=None,
            ),
            status="computed",
        )

    # ---------- BUILD TRAIT OBJECTS ----------

    # Height trait
    if height_label is not None:
        expl = (
            f"{height_label} based on the relative height of the forehead compared "
            f"to total facial height (ratio ≈ {height_ratio:.2f})."
            if height_ratio is not None
            else f"{height_label} based on visual proportions of the upper third."
        )
        traits.append(
            make_trait(
                name=height_label,
                present=True,
                confidence=height_conf,
                explanation=expl,
                measurement_keys=[
                    "forehead_height_tr_g_profile",
                    "face_height_tr_gn_profile",
                    "face_height",
                ],
            )
        )

    # Width trait
    if width_label is not None:
        expl = (
            f"{width_label} based on the proportion between forehead width and "
            f"cheekbone width (ratio ≈ {width_ratio:.2f})."
            if width_ratio is not None
            else f"{width_label} based on relative width of the upper facial third."
        )
        traits.append(
            make_trait(
                name=width_label,
                present=True,
                confidence=width_conf,
                explanation=expl,
                measurement_keys=[
                    "forehead_width",
                    "cheekbone_width",
                ],
            )
        )

    # Front shape trait
    if front_shape_label is not None:
        expl = (
            f"{front_shape_label} based on the balance between forehead height "
            f"(≈ {forehead_height_for_aspect:.1f} px) and forehead width "
            f"(≈ {forehead_width:.1f} px), aspect ratio ≈ {aspect_ratio:.2f}."
            if forehead_width is not None and forehead_height_for_aspect is not None
            else f"{front_shape_label} estimated from the frontal proportions."
        )
        traits.append(
            make_trait(
                name=front_shape_label,
                present=True,
                confidence=front_shape_conf,
                explanation=expl,
                measurement_keys=[
                    "forehead_height_tr_g_profile",
                    "forehead_width",
                    "face_height",
                ],
            )
        )

    # Profile trait
    if profile_label is not None:
        if slope_abs is not None:
            expl = (
                f"{profile_label} based on the angle between the forehead line "
                f"(Tr–G) and the vertical (≈ {slope_abs:.1f}°)."
            )
        else:
            expl = f"{profile_label} estimated from the profile silhouette."
        traits.append(
            make_trait(
                name=profile_label,
                present=True,
                confidence=profile_conf,
                explanation=expl,
                measurement_keys=[
                    "forehead_slope_signed_deg_profile",
                    "forehead_slope_deg_profile",
                ],
            )
        )

    # ---------- OVERALL FOREHEAD SHAPE LABEL (DICT) ----------

    summary_parts = []
    if front_shape_label:
        summary_parts.append(front_shape_label)
    if height_label and "Average" not in height_label:
        summary_parts.append(height_label)
    if width_label and "Balanced" not in width_label:
        summary_parts.append(width_label)
    if profile_label:
        summary_parts.append(profile_label)

    if summary_parts:
        summary_label = ", ".join(summary_parts)
    else:
        summary_label = "Forehead traits not clearly determined"

    forehead_shape = {
        "label": summary_label,
        "height_label": height_label,
        "width_label": width_label,
        "front_shape_label": front_shape_label,
        "profile_label": profile_label,
        # you can add "romanian_label" here later if you want
    }

    # Optional synthetic summary trait at top
    if traits:
        traits.insert(
            0,
            make_trait(
                name=f"Forehead summary: {summary_label}",
                present=True,
                confidence=0.95,
                explanation=(
                    "Combined interpretation of forehead height, width and profile "
                    "shape based on front and side measurements."
                ),
                measurement_keys=[
                    "forehead_width",
                    "cheekbone_width",
                    "face_height",
                    "forehead_height_tr_g_profile",
                    "face_height_tr_gn_profile",
                    "forehead_slope_deg_profile",
                ],
            ),
        )

    return forehead_shape, traits
