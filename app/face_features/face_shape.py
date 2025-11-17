from app.services.measurements import all_measurements


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

    if approx(a, c, 0.12) and approx(c, d, 0.12) and abs(b - c) < 0.4 * c:
        return ("Square","Pătrată / Coleric – Pământ","Long teardrops, oversized hoops, chandeliers.")

    if approx(a, c, 0.12) and approx(c, d, 0.12) and b > c * 1.15:
        return ("Rectangular","Dreptunghiulară / Sangvin – Marte","Small studs, round buttons, teardrops, hoops.")

    if c > a * 1.25 and c > d * 1.1:
        return ("Diamond","Diamant","Small hoops, studs, small drop earrings.")

    if a >= c * 0.95 and a >= d * 1.05 and c >= d * 1.05:
        return ("Heart","Inimă / Sangvin – Venus","Triangulars, chandeliers, teardrops. Avoid tiny studs.")

    if d > a * 1.05 and d >= c:  # jaw only needs to be 5% bigger than forehead
        return ("Triangle","Triunghiulară / Nervos – Mercur","Large ovals, small circles, curved bottoms.")

    if b > c * 1.2 and a < c and d < c:
        return ("Oval","Ovală / Sangvin – Soare","Any style of earrings will go.")

    if b > a * 1.4 and approx(a, c, 0.1) and approx(c, d, 0.1):
        return ("Oblong","Alungită / Limfatic – Neptun","Hoops, round studs, classy rounded shapes.")

    if abs(c - b) < 0.15 * b and c > a and c > d * 0.95 and b < c * 1.1:
        return ("Round","Rotundă / Limfatic – Lună","Long earrings, rectangles, long ovals. Avoid hoops.")

    return ("Unknown","Necunoscut / A se verifica","No recommendation.")
