from math import sqrt

def approx(x, y, tol=0.1):
    if x is None or y is None:
        return False
    denom = max(abs(x), abs(y), 1e-6)
    return abs(x - y) / denom <= tol


def classify_face_type(
    forehead_width: float,
    face_height: float,
    cheekbone_width: float,
    jaw_width: float,
):
    a = forehead_width
    b = face_height
    c = cheekbone_width
    d = jaw_width

    print(f"[DEBUG] classify_face_type → a={a}, b={b}, c={c}, d={d}")

    # -------------------------
    # 1) Rule-based classification
    # -------------------------

    # Square
    if approx(a, c, 0.12) and approx(c, d, 0.12) and abs(b - c) < 0.4 * c:
        rule = "Square: a≈c≈d and |b−c| < 0.4·c"
        return ("Square Face", "Pătrată / Coleric – Pământ",
                "Long teardrops, oversized hoops, chandeliers.",
                rule)

    # Rectangular
    if approx(a, c, 0.12) and approx(c, d, 0.12) and b > c * 1.15:
        rule = "Rectangular: a≈c≈d and b > 1.15·c"
        return ("Rectangular Face", "Dreptunghiulară / Sangvin – Marte",
                "Small studs, round buttons, teardrops, hoops.",
                rule)

    # Diamond
    if c > a * 1.25 and c > d * 1.1 and b >= c * 0.9:
        rule = "Diamond: c > 1.25·a and c > 1.1·d and b ≥ 0.9·c"
        return ("Diamond Face", "Diamant",
                "Small hoops, studs, small drop earrings.",
                rule)

    # Heart
    if a >= c * 0.95 and a >= d * 1.05 and c >= d * 1.05:
        rule = "Heart: a ≥ 0.95·c and a ≥ 1.05·d and c ≥ 1.05·d"
        return ("Heart-Shaped Face", "Inimă / Sangvin – Venus",
                "Triangulars, chandeliers, teardrops. Avoid tiny studs.",
                rule)

    # Round – closer to original: c ≈ b; a ≈ d; b,c > a,d
    if (
        approx(c, b, 0.12)      # cheekbone_width ≈ face_height
        and approx(a, d, 0.12)  # forehead_width ≈ jaw_width
        and b > a and b > d     # height > forehead & jaw
        and c > a and c > d     # cheekbones > forehead & jaw
    ):
        rule = "Round: c≈b and a≈d and b,c > a,d"
        return ("Round Face", "Rotundă / Limfatic – Lună",
                "Long earrings, rectangles, long ovals. Avoid hoops.",
                rule)

    # Oval
    if b > c * 1.2 and a < c and d < c:
        rule = "Oval: b > 1.2·c and a < c and d < c"
        return ("Oval Face", "Ovală / Sangvin – Soare",
                "Any style of earrings will go.",
                rule)

    # Oblong (Long)
    if b > a * 1.4 and approx(a, c, 0.1) and approx(c, d, 0.1):
        rule = "Oblong: b > 1.4·a and a≈c≈d"
        return ("Oblong (Long) Face", "Alungită / Limfatic – Neptun",
                "Hoops, round studs, classy rounded shapes.",
                rule)

    # Triangular – stricter: jaw clearly dominant
    if a and c:
        r_jaw_to_forehead = d / a
        r_jaw_to_cheek = d / c
        if r_jaw_to_forehead > 1.10 and r_jaw_to_cheek > 1.05 and a <= c * 0.97:
            rule = ("Triangular: d/a > 1.10 and d/c > 1.05 "
                    "and a ≤ 0.97·c")
            return ("Triangular Face", "Triunghiulară / Nervos – Mercur",
                    "Large ovals, small circles, curved bottoms.",
                    rule)

    # -------------------------
    # 2) Fallback: nearest prototype
    # -------------------------

    if not (a and b and c and d):
        rule = "Fallback: missing measurements → default Oval"
        return ("Oval Face", "Ovală / Sangvin – Soare",
                "Any style of earrings will go.",
                rule)

    # Normalize by cheekbone width as main reference
    rf = a / c      # forehead / cheekbones
    rj = d / c      # jaw / cheekbones
    rh = b / c      # height / cheekbones

    # Very rough "ideal" ratios for each shape
    prototypes = {
        "Square Face":        (1.0, 1.0, 1.0),
        "Rectangular Face":   (1.0, 1.0, 1.3),
        "Round Face":         (0.9, 0.9, 1.0),
        "Oval Face":          (0.9, 0.9, 1.35),
        "Oblong (Long) Face": (1.0, 1.0, 1.5),
        "Heart-Shaped Face":  (1.1, 0.9, 1.3),
        "Diamond Face":       (0.8, 0.9, 1.3),
        "Triangular Face":    (0.9, 1.1, 1.2),
    }

    labels = {
        "Square Face":        ("Pătrată / Coleric – Pământ",
                               "Long teardrops, oversized hoops, chandeliers."),
        "Rectangular Face":   ("Dreptunghiulară / Sangvin – Marte",
                               "Small studs, round buttons, teardrops, hoops."),
        "Round Face":         ("Rotundă / Limfatic – Lună",
                               "Long earrings, rectangles, long ovals. Avoid hoops."),
        "Oval Face":          ("Ovală / Sangvin – Soare",
                               "Any style of earrings will go."),
        "Oblong (Long) Face": ("Alungită / Limfatic – Neptun",
                               "Hoops, round studs, classy rounded shapes."),
        "Heart-Shaped Face":  ("Inimă / Sangvin – Venus",
                               "Triangulars, chandeliers, teardrops. Avoid tiny studs."),
        "Diamond Face":       ("Diamant",
                               "Small hoops, studs, small drop earrings."),
        "Triangular Face":    ("Triunghiulară / Nervos – Mercur",
                               "Large ovals, small circles, curved bottoms."),
    }

    best_shape = None
    best_dist = 1e9

    for shape, (pf, pj, ph) in prototypes.items():
        dist = sqrt((rf - pf) ** 2 + (rj - pj) ** 2 + (rh - ph) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_shape = shape

    romanian_label, earring_tip = labels[best_shape]
    rule = (f"Fallback prototype NN → {best_shape} "
            f"(rf={rf:.2f}, rj={rj:.2f}, rh={rh:.2f}, dist={best_dist:.3f})")

    print(f"[DEBUG] Fallback classifier → {best_shape} (dist={best_dist:.3f}, ratios=rf={rf:.2f}, rj={rj:.2f}, rh={rh:.2f})")
    return best_shape, romanian_label, earring_tip, rule
