def classify_forehead_traits(measurements):
    fw = measurements.forehead_width
    jw = measurements.jaw_width

    ratio = fw / max(jw, 1e-6)

    if ratio >= 1.15:
        label = "Wide Forehead"
        justification = (
            f"Forehead is visibly wider than jaw (ratio={ratio:.2f})."
        )
        traits = [{
            "meaning": "Broad-minded, visionary",
            "explanation": "A wide forehead is associated with imagination and strategic thinking.",
            "source": "Tehnica de citire a feței.docx - Frunte largă"
        }]

    elif ratio <= 0.80:
        label = "Narrow Forehead"
        justification = (
            f"Forehead is significantly narrower than jaw (ratio={ratio:.2f})."
        )
        traits = [{
            "meaning": "Detail-oriented, analytical",
            "explanation": "A narrow forehead suggests precision, logic and practical mindset.",
            "source": "Tehnica de citire a feței.docx - Frunte îngustă"
        }]

    else:
        label = "Average Forehead"
        justification = (
            f"Forehead and jaw are proportionate (ratio={ratio:.2f})."
        )
        traits = [{
            "meaning": "Balanced thinker",
            "explanation": "Proportions indicate a balanced mind between logic and creativity.",
            "source": "Tehnica de citire a feței.docx - Frunte echilibrată"
        }]

    return {"label": label, "justification": justification}, traits
