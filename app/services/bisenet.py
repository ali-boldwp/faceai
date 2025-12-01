# app/services/bisenet.py

import sys
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests  # <--- NEW

# -------------------------------------------------------------------
# Auto-setup for face-parsing.PyTorch + BiSeNet weights
# -------------------------------------------------------------------

# Where we will keep the original face-parsing code
FACE_PARSING_DIR = os.path.abspath("./face-parsing.PyTorch")

# Where we will keep the pretrained weights
WEIGHTS_DIR = os.path.join(FACE_PARSING_DIR, "weights")
WEIGHTS_NAME = "79999_iter.pth"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_NAME)

# URLs for original code & weights
MODEL_URL = "https://raw.githubusercontent.com/zllrunning/face-parsing.PyTorch/master/model.py"
RESNET_URL = "https://raw.githubusercontent.com/zllrunning/face-parsing.PyTorch/master/resnet.py"
WEIGHTS_URL = "https://huggingface.co/bes-dev/face_parsing/resolve/main/79999_iter.pth"

def _download_file(url: str, dst_path: str, desc: str) -> None:
    """Download a file if it does not exist."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return

    print(f"[BiSeNet] Downloading {desc} from {url} ...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"[BiSeNet] Downloaded {desc} to {dst_path}")
    except Exception as e:
        # Clean up partial file if something went wrong
        if os.path.exists(dst_path):
            try:
                os.remove(dst_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to download {desc} from {url}: {e}")


def ensure_face_parsing_code() -> None:
    """Ensure face-parsing.PyTorch/model.py and resnet.py are present locally."""
    os.makedirs(FACE_PARSING_DIR, exist_ok=True)
    model_py = os.path.join(FACE_PARSING_DIR, "model.py")
    resnet_py = os.path.join(FACE_PARSING_DIR, "resnet.py")

    _download_file(MODEL_URL, model_py, "face-parsing model.py")
    _download_file(RESNET_URL, resnet_py, "face-parsing resnet.py")


def ensure_bisenet_weights() -> None:
    """Ensure BiSeNet weights 79999_iter.pth are present locally."""
    _download_file(WEIGHTS_URL, WEIGHTS_PATH, "BiSeNet weights (79999_iter.pth)")


# Ensure code exists, then import BiSeNet from it
ensure_face_parsing_code()
sys.path.append(FACE_PARSING_DIR)
from model import BiSeNet  # <-- THIS is the official BiSeNet that matches the weights


def load_bisenet(checkpoint: str = WEIGHTS_PATH, n_classes: int = 19):
    """
    Load BiSeNet with pretrained face-parsing weights.

    - If the face-parsing code is missing, it is downloaded.
    - If the weights file is missing, it is downloaded.
    """
    ensure_bisenet_weights()

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"BiSeNet weights are missing even after download: {checkpoint}"
        )

    net = BiSeNet(n_classes=n_classes)
    print(f"[BiSeNet] Loading weights from {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    net.load_state_dict(state)  # strict=True by default, now architecture matches
    net.eval()
    return net

# -------------------------------------------------------------------
# Keep your existing functions BELOW this point:
# - preprocess(...)
# - hair_mask(...)
# - analyze_hairline(...)
# - etc.
# -------------------------------------------------------------------


# Preprocess image
def preprocess(img_path, size=(512,512)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return to_tensor(img).unsqueeze(0), np.array(img)

# Generate hair mask
def hair_mask(net, tensor):
    with torch.no_grad():
        out = net(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    return (parsing == 17).astype(np.uint8), parsing

# Hairline analysis
def analyze_hairline(mask, rgb_img, out_path='hairline_output.png'):
    h, w = mask.shape

    # 1) Find hair bounding box first
    ys_all, xs_all = np.where(mask)
    if len(xs_all) < 10:
        return "Not enough hair pixels", 0

    y_min, y_max = ys_all.min(), ys_all.max()
    x_min, x_max = xs_all.min(), xs_all.max()

    # 2) Use the LOWER part of the hair bbox as ROI (actual hairline zone)
    #    e.g. bottom 40% of hair bbox
    y1 = int(y_min + 0.6 * (y_max - y_min))
    y2 = y_max
    x1 = int(x_min + 0.1 * (x_max - x_min))
    x2 = int(x_max - 0.1 * (x_max - x_min))

    roi = np.zeros_like(mask, dtype=bool)
    roi[y1:y2, x1:x2] = True

    hairline_region = np.logical_and(mask.astype(bool), roi)
    ys, xs = np.where(hairline_region)
    if len(xs) < 10:
        return "Not enough hairline points", y1

    unique_x = np.unique(xs)

    # 3) For each x, take the BOTTOM-most hair pixel as hairline
    hairline_y = {x: max(ys[xs == x]) for x in unique_x}
    xs_s = np.array(sorted(hairline_y.keys()))
    ys_s = np.array([hairline_y[x] for x in xs_s])

    # Quadratic fit
    coeffs = np.polyfit(xs_s, ys_s, 2)
    a = coeffs[0]
    dy = ys_s.max() - ys_s.min()

    # Center vs sides (better than global dy for V/M)
    mid_x = xs_s.mean()
    mid_idx = np.argmin(np.abs(xs_s - mid_x))
    mid_y = ys_s[mid_idx]
    side_n = max(5, len(xs_s)//8)
    side_mean = np.concatenate([ys_s[:side_n], ys_s[-side_n:]]).mean()
    delta_mid = mid_y - side_mean   # >0 = center LOWER (V), <0 = center HIGHER (M)

    def classify(a, dy, delta_mid):
        # M: center higher, sides lower
        if delta_mid < -8 and dy > 15:
            return "M-shape"
        # V: center lower, strong dip
        if delta_mid > 8 and dy > 15:
            return "V-shape"
        # Rounded vs Flat
        if dy > 10:
            return "Rounded"
        if dy <= 10:
            return "Flat"
        return "Unclassified"

    shape = classify(a, dy, delta_mid)

    # Debug plot
    plt.imshow(rgb_img)
    plt.scatter(xs_s, ys_s, s=2, c='red')
    plt.plot(xs_s, np.poly1d(coeffs)(xs_s), c='blue')
    plt.title(f"Hairline: {shape}")
    plt.axis('off')
    plt.savefig(out_path)
    plt.close()

    return shape, int(ys_s.min())

# Extract measurements
def extract_metrics(parsing_mask, hairline_y):
    LABEL_FACE = 1
    coords = np.column_stack(np.where(parsing_mask == LABEL_FACE))
    top = coords[:, 0].min()
    bottom = coords[:, 0].max()
    left = coords[:, 1].min()
    right = coords[:, 1].max()

    # a = forehead width (approx)
    a = int((right - left) * 0.6)

    # b = face height (hairline → chin)
    b = bottom - hairline_y

    # c = cheekbone width
    c = right - left

    # d = half jaw width approximation
    d = c / 2.5

    return a, b, c, d

# Face shape classification


def approx(x, y, tol=0.1):
    return abs(x - y) / max(x, y) < tol if x and y else False

def get_face_type(a, b, c, d):
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