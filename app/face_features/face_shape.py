from app.services.measurements import all_measurements

def approx_equal(x, y, tol=0.1):
    return abs(x - y) / max(x, y) < tol if x and y else False

def approx(x, y, tol=0.1):
    return abs(x - y) / max(x, y) < tol if x and y else False

def classify_face_type(a, b, c, d):
    jaw = 2 * d

    print(f"[DEBUG] Shape check → a={a}, c={c}, jaw={jaw}, b={b}")
    print("🚨 Using latest face classification logic")

    # --- Square: all widths are similar, face not too long ---
    # --- Square: all widths similar, not long
    if (
            approx(a, c, 0.2) and
            approx(c, jaw, 0.15) and
            abs(b - c) < 70
    ):
        return ("Square", "Pătrată / Coleric – Pământ", "Long teardrops, oversized hoops, chandeliers.")

    # --- Rectangular: like square, but longer face ---
    if (
        approx(a, c, 0.12) and
        approx(c, jaw, 0.12) and
        b > c * 1.15
    ):
        return ("Rectangular", "Dreptunghiulară / Sangvin – Marte", "Small studs, round buttons, teardrops, hoops.")

    # --- Diamond ---
    if (
        c > a * 1.25 and
        c > jaw * 1.25 and
        b > c * 1.15
    ):
        return ("Diamond", "Diamant", "Small hoops, studs, small drop earrings.")

    # --- Heart ---
    if (
            c > jaw * 1.15 and  # Cheekbones clearly dominant
            b > c * 1.05 and  # Face longer than cheekbones
            a > c * 0.55 and a < c * 0.95  # Balanced forehead
    ):
        return ("Heart", "Inimă / Sangvin – Venus", "Triangulars, chandeliers, teardrops. Avoid tiny studs.")

    # --- Triangle ---
    if jaw > c * 1.1 and c > a:
        return ("Triangle", "Triunghiulară / Nervos – Mercur", "Large ovals, small circles, curved bottoms.")

    # --- Oval ---
    if b > c * 1.3 and a < c and jaw < c:
        return ("Oval", "Ovală / Sangvin – Soare", "Any style of earrings will go.")

    # --- Oblong ---
    if b > a * 1.4 and approx(a, c, 0.1) and approx(c, jaw, 0.1):
        return ("Oblong", "Alungită / Limfatic – Neptun", "Hoops, round studs, classy rounded shapes.")

    # --- Round ---
    if abs(c - b) < 25 and approx(a, jaw, 0.15) and c > a and b > a:
        return ("Round", "Rotundă / Limfatic – Lună", "Long earrings, rectangles, long ovals. Avoid hoops.")

    # --- Fallback ---
    if max(a, c, jaw) - min(a, c, jaw) < 100 and abs(b - c) < 100:
        return ("Square", "Pătrată / Coleric – Pământ", "Long teardrops, oversized hoops, chandeliers.")

    return ("Unknown", "Necunoscut / A se verifica", "No recommendation.")

def classify_face_shape(landmarks, measurements=None):
    if measurements is None:
        measurements = all_measurements(landmarks)

    a = measurements.get("forehead_width")     # a
    b = measurements.get("face_height")        # b
    c = measurements.get("cheekbone_width")    # c
    d = measurements.get("jaw_width")          # d

    if None in (a, b, c, d):
        return "Undefined", "Missing key facial dimensions (forehead, height, cheekbone, or jaw)."

    shape = None
    justification = ""

    if approx_equal(a, b) and approx_equal(a, c) and approx_equal(a, d):
        shape = "Square"
        justification = (
            f"Face is nearly as wide as it is long, and forehead, cheekbone, and jaw widths are similar "
            f"(a≈b≈c≈d: {a:.1f}, {b:.1f}, {c:.1f}, {d:.1f})."
        )
    elif b > a * 1.1 and b > c * 1.1 and b > d * 1.1 and approx_equal(a, c) and approx_equal(a, d):
        shape = "Oblong"
        justification = (
            f"Face height is much greater than its width (b={b:.1f} >> a,c,d), with similar forehead, cheek, jaw widths."
        )
    elif approx_equal(c, b) and b > a * 1.1 and b > d * 1.1 and c > a * 1.1 and c > d * 1.1:
        shape = "Round"
        justification = (
            f"Face is nearly circular: cheekbone width ≈ face height (c≈b≈{b:.1f}), both larger than forehead (a={a:.1f}) and jaw (d={d:.1f})."
        )
    elif a > d:
        if a > d * 1.2:
            shape = "Heart"
            justification = (
                f"Broad forehead (a={a:.1f}) and narrower jaw (d={d:.1f}) indicate a heart-shaped face."
            )
        else:
            shape = "Trapezoid (Base Up)"
            justification = (
                f"Forehead is slightly wider than jaw (a≈{a:.1f}, d≈{d:.1f}), suggesting an inverted trapezoid face."
            )
    else:
        if d > a * 1.2:
            shape = "Triangle"
            justification = (
                f"Jaw is significantly wider than forehead (d={d:.1f} >> a={a:.1f}), indicating a triangular face."
            )
        else:
            shape = "Trapezoid (Base Down)"
            justification = (
                f"Jaw is a bit wider than forehead (a≈{a:.1f}, d≈{d:.1f}), suggesting a trapezoid-shaped face (wider base)."
            )

    if shape is None:
        shape = "Oval"
        justification = "Proportions are balanced without extreme values, classified as Oval."

    return shape, justification
