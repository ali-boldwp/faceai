from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from app.services.measurements import all_measurements


def safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if not num or not den or den == 0:
        return default
    return num / den

def between(x: float, lo: float, hi: float) -> bool:
    return lo <= x <= hi

def approx1(x: float, tol: float = 0.12) -> bool:
    """True if x is within ±tol of 1.0 (relative)."""
    return (1.0 - tol) <= x <= (1.0 + tol)

@dataclass
class FaceMetrics:
    a: float; b: float; c: float; d: float

    @property
    def R_hw(self) -> float: 
        return safe_ratio(self.b, self.c)

    @property
    def R_fc(self) -> float:  # forehead-to-cheekbone
        return safe_ratio(self.a, self.c)

    @property
    def R_jc(self) -> float:  # jaw-to-cheekbone
        return safe_ratio(self.d, self.c)

    @property
    def R_fj(self) -> float:  # forehead-to-jaw
        return safe_ratio(self.a, self.d)

    def as_dict(self) -> Dict[str, float]:
        return dict(
            a=self.a, b=self.b, c=self.c, d=self.d,
            R_hw=self.R_hw, R_fc=self.R_fc, R_jc=self.R_jc, R_fj=self.R_fj
        )

def _validate(m: FaceMetrics) -> Optional[str]:
    if any(v is None for v in (m.a, m.b, m.c, m.d)):
        return "Missing key facial dimensions (forehead, height, cheekbone, or jaw)."
    if any(v <= 0 for v in (m.a, m.b, m.c, m.d)):
        return "Non-positive measurement encountered. Ensure all lengths are > 0."
    return None


PRIMARY_SHAPES = [
    "Round Face", "Oval Face", "Oblong (Long) Face", "Triangular Face",
    "Heart-Shaped Face", "Square Face", "Rectangular Face", "Diamond Face",
    "Upward Trapezoid Face", "Downward Trapezoid Face"
]
ATTR_WIDE = "Wide Face"
ATTR_NARROW = "Narrow Face"

def classify_face_shape(
    landmarks_or_measurements,
    measurements: Optional[Dict[str, float]] = None,
    treat_width_attributes_as_primary: bool = False
) -> Tuple[str, str, Dict[str, float], List[str]]:
    """
    Returns: (primary_shape, justification, debug_ratios, attributes)
    attributes may include: 'Wide Face', 'Narrow Face'
    Set `treat_width_attributes_as_primary=True` to output those as primary instead.
    """
    # Ingest measurements
    if measurements is None:
        # Assume raw landmarks were passed
        mvals = all_measurements(landmarks_or_measurements)
        a = mvals.get("forehead_width")
        b = mvals.get("face_height")
        c = mvals.get("cheekbone_width")
        d = mvals.get("jaw_width")
    else:
        # Assume dict with numeric values
        a = measurements.get("forehead_width") or measurements.get("a")
        b = measurements.get("face_height") or measurements.get("b")
        c = measurements.get("cheekbone_width") or measurements.get("c")
        d = measurements.get("jaw_width") or measurements.get("d")

    met = FaceMetrics(a, b, c, d)
    err = _validate(met)
    if err:
        return "Undefined", err, met.as_dict(), []

    R_hw, R_fc, R_jc, R_fj = met.R_hw, met.R_fc, met.R_jc, met.R_fj
    debug = met.as_dict()

    # Width attributes (relative width vs height via cheekbone proxy)
    attributes: List[str] = []
    # Wide if width close to or exceeding height; Narrow if quite long
    if R_hw <= 1.02:      # b/c <= 1.02 → “wide” impression
        attributes.append(ATTR_WIDE)
    elif R_hw >= 1.45:    # b/c >= 1.45 → “narrow/long” impression
        attributes.append(ATTR_NARROW)


    if (R_fc <= 0.90) and (R_jc <= 0.90) and (R_hw >= 1.05):
        shape = "Diamond Face"
        why = f"Cheekbones dominate (a/c={R_fc:.2f}≤0.90, d/c={R_jc:.2f}≤0.90) with slight length (b/c={R_hw:.2f}≥1.05)."

    elif (R_fj >= 1.15) and (R_fc >= 1.00) and (R_jc <= 0.92) and (R_hw >= 1.05):
        shape = "Heart-Shaped Face"
        why = f"Forehead > jaw (a/d={R_fj:.2f}≥1.15), jaw tapers (d/c={R_jc:.2f}≤0.92), modest length (b/c={R_hw:.2f})."

    elif (R_fj <= 0.87) and (R_jc >= 1.00) and (R_hw >= 1.02):
        shape = "Triangular Face"
        why = f"Jaw wider than forehead (a/d={R_fj:.2f}≤0.87) and ≳ cheekbones (d/c={R_jc:.2f}≥1.00)."

    elif (R_fj >= 1.05) and (R_fj < 1.15) and between(R_fc, 0.95, 1.15) and between(R_jc, 0.85, 1.05):
        shape = "Upward Trapezoid Face"
        why = f"Forehead slightly wider than jaw (a/d={R_fj:.2f} in 1.05–1.15) with near-parallel sides (a,c,d similar)."

    elif (R_fj <= 0.95) and (R_fj > 0.87) and between(R_fc, 0.85, 1.05) and between(R_jc, 0.95, 1.15):
        shape = "Downward Trapezoid Face"
        why = f"Jaw slightly wider than forehead (a/d={R_fj:.2f} in 0.87–0.95) with near-parallel sides."

    # Square: widths ~ equal; not long
    elif between(R_hw, 1.00, 1.15) and approx1(R_fc, 0.12) and approx1(R_jc, 0.12):
        shape = "Square Face"
        why = f"Similar widths (a≈c≈d) and not long (b/c={R_hw:.2f} in 1.00–1.15)."

    # Rectangular: widths ~ equal; somewhat long
    elif between(R_hw, 1.15, 1.35) and approx1(R_fc, 0.12) and approx1(R_jc, 0.12):
        shape = "Rectangular Face"
        why = f"Similar widths (a≈c≈d) but longer (b/c={R_hw:.2f} in 1.15–1.35)."

    # Oblong (Long): widths ~ equal; distinctly long
    elif (R_hw > 1.35) and approx1(R_fc, 0.12) and approx1(R_jc, 0.12):
        shape = "Oblong (Long) Face"
        why = f"Similar widths (a≈c≈d) with pronounced length (b/c={R_hw:.2f} > 1.35)."

    # Round: height ≈ width, forehead & jaw slightly soft
    elif between(R_hw, 0.95, 1.10) and between(R_fc, 0.90, 1.00) and between(R_jc, 0.90, 1.00):
        shape = "Round Face"
        why = f"Height close to width (b/c={R_hw:.2f}), softer forehead/jaw (a/c={R_fc:.2f}, d/c={R_jc:.2f})."

    # Oval: longer than round; cheekbones widest; forehead & jaw mildly narrower
    elif between(R_hw, 1.20, 1.50) and between(R_fc, 0.85, 0.98) and between(R_jc, 0.85, 0.98):
        shape = "Oval Face"
        why = f"Longer (b/c={R_hw:.2f}) with cheekbones subtly widest (a/c={R_fc:.2f}, d/c={R_jc:.2f} < 1)."

    else:
        # Fallback using nearest family by R_hw
        if R_hw >= 1.30:
            shape = "Rectangular Face"
            why = f"Fallback to rectangular family: length noticeable (b/c={R_hw:.2f})."
        elif R_hw <= 1.05:
            shape = "Square Face"
            why = f"Fallback to square/round family: length minimal (b/c={R_hw:.2f})."
        else:
            shape = "Oval Face"
            why = f"Fallback to oval family: balanced length (b/c={R_hw:.2f})."

    # Optionally elevate width attribute to primary
    if treat_width_attributes_as_primary and attributes:
        # If both wide & narrow somehow (shouldn't happen), keep primary shape
        if len(attributes) == 1:
            return attributes[0], f"Width attribute dominates ({attributes[0]}). {why}", debug, []

    justification = f"{why} | ratios: R_hw={R_hw:.3f}, R_fc={R_fc:.3f}, R_jc={R_jc:.3f}, R_fj={R_fj:.3f}"
    return shape, justification, debug, attributes

# --- compatibility shim (keeps your old API/labels) ---------------------------

def classify_face_type(a: float, b: float, c: float, d: float):
    """Return (shape, ro_label, earring_tip) using your Romanian labels."""
    primary, _, _, _ = classify_face_shape(
        {}, measurements={"a": a, "b": b, "c": c, "d": d}
    )

    ro_label = {
        "Round Face": "Rotundă / Limfatic – Lună",
        "Oval Face": "Ovală / Sangvin – Soare",
        "Oblong (Long) Face": "Alungită / Limfatic – Neptun",
        "Triangular Face": "Triunghiulară / Nervos – Mercur",
        "Heart-Shaped Face": "Inimă / Sangvin – Venus",
        "Square Face": "Pătrată / Coleric – Pământ",
        "Rectangular Face": "Dreptunghiulară / Sangvin – Marte",
        "Diamond Face": "Diamant",
        "Upward Trapezoid Face": "Trapez inversat (baza sus)",
        "Downward Trapezoid Face": "Trapez (baza jos)",
    }.get(primary, "Necunoscut / A se verifica")

    earring_tip = {
        "Round Face": "Long earrings, rectangles, long ovals. Avoid hoops.",
        "Oval Face": "Any style of earrings will go.",
        "Oblong (Long) Face": "Hoops, round studs, classy rounded shapes.",
        "Triangular Face": "Large ovals, small circles, curved bottoms.",
        "Heart-Shaped Face": "Triangulars, chandeliers, teardrops. Avoid tiny studs.",
        "Square Face": "Long teardrops, oversized hoops, chandeliers.",
        "Rectangular Face": "Small studs, round buttons, teardrops, hoops.",
        "Diamond Face": "Small hoops, studs, small drop earrings.",
        "Upward Trapezoid Face": "Teardrops, chandeliers; soften the forehead width.",
        "Downward Trapezoid Face": "Rounded bottoms, soft curves; balance the jaw.",
    }.get(primary, "No recommendation.")

    return primary, ro_label, earring_tip
