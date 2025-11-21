from typing import Dict, List, Tuple


def classify_forehead_traits(measurements) -> Tuple[Dict, List[Dict]]:
    """
    Classify forehead width vs jaw width.

    Returns:
      - a dict: {"label": ..., "justification": ...}
      - a list of psychological trait dicts
    """
    fw = float(getattr(measurements, "forehead_width", 0.0))
    jw = float(getattr(measurements, "jaw_width", 0.0)) or 1e-6

    ratio = fw / jw

    traits: List[Dict] = []

    # Very wide forehead
    if ratio >= 1.25:
        label = "Very Wide Forehead"
        justification = (
            f"Forehead is markedly wider than jaw (ratio={ratio:.2f})."
        )
        traits.append({
            "meaning": "Highly imaginative, strategic",
            "explanation": (
                "A very wide forehead is linked to intense mental activity, "
                "vision and capacity for planning ahead."
            ),
            "source": "Tehnica de citire a feței.docx – Frunte foarte largă",
        })

    # Wide forehead
    elif 1.10 <= ratio < 1.25:
        label = "Wide Forehead"
        justification = (
            f"Forehead is visibly wider than jaw (ratio={ratio:.2f})."
        )
        traits.append({
            "meaning": "Broad-minded, visionary",
            "explanation": (
                "A wide forehead is associated with imagination, curiosity "
                "and openness to new ideas."
            ),
            "source": "Tehnica de citire a feței.docx – Frunte largă",
        })

    # Balanced forehead
    elif 0.90 <= ratio < 1.10:
        label = "Average Forehead"
        justification = (
            f"Forehead and jaw are proportionate (ratio={ratio:.2f})."
        )
        traits.append({
            "meaning": "Balanced thinker",
            "explanation": (
                "Proportions indicate a balance between abstract thinking "
                "and practical sense."
            ),
            "source": "Tehnica de citire a feței.docx – Frunte echilibrată",
        })

    # Slightly narrow forehead
    elif 0.80 <= ratio < 0.90:
        label = "Slightly Narrow Forehead"
        justification = (
            f"Forehead is slightly narrower than jaw (ratio={ratio:.2f})."
        )
        traits.append({
            "meaning": "Pragmatic, concrete thinker",
            "explanation": (
                "A slightly narrower forehead suggests a person who prefers "
                "practical, concrete approaches."
            ),
            "source": "Tehnica de citire a feței.docx – Frunte ușor îngustă",
        })

    # Narrow forehead  (ratio < 0.80)
    else:
        label = "Narrow Forehead"
        justification = (
            f"Forehead is noticeably narrower than jaw (ratio={ratio:.2f})."
        )
        traits.append({
            "meaning": "Focused, action-oriented",
            "explanation": (
                "O frunte semnificativ mai îngustă decât maxilarul este legată "
                "de orientarea spre acțiune și nevoia de rezultate rapide, "
                "uneori cu răbdare redusă."
            ),
            "source": "Tehnica de citire a feței.docx – Frunte îngustă",
        })

    return {"label": label, "justification": justification}, traits
