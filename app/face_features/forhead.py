def classify_forehead_traits(measurements):
    fw = measurements.forehead_width
    jw = measurements.jaw_width

    if fw > 1.1 * jw:
        label = "Wide Forehead"
        justification = f"Forehead is significantly wider than jaw (forehead: {fw:.1f}, jaw: {jw:.1f})."
        traits = [{
            "meaning": "Broad-minded",
            "explanation": "A wide forehead suggests visionary thinking and openness to new ideas.",
            "source": "Tehnica de citire a feței.docx"
        }]
    elif fw < 0.9 * jw:
        label = "Narrow Forehead"
        justification = f"Forehead is narrower than jaw (forehead: {fw:.1f}, jaw: {jw:.1f})."
        traits = [{
            "meaning": "Detail-oriented",
            "explanation": "A narrow forehead indicates a practical, methodical thinker.",
            "source": "Tehnica de citire a feței.docx"
        }]
    else:
        label = "Average Forehead"
        justification = f"Forehead and jaw widths are nearly equal (forehead: {fw:.1f}, jaw: {jw:.1f})."
        traits = [{
            "meaning": "Balanced thinker",
            "explanation": "Forehead proportions indicate a balance between imagination and logic.",
            "source": "Tehnica de citire a feței.docx"
        }]

    return {"label": label, "justification": justification}, traits
