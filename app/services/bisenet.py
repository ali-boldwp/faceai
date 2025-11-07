import sys
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("./face-parsing.PyTorch"))

from model import BiSeNet

# Load BiSeNet model
def load_bisenet(checkpoint='face-parsing.PyTorch/weights/79999_iter.pth', n_classes=19):
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False))
    net.eval()
    return net

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
    y1, y2 = int(h*0.12), int(h*0.30)
    x1, x2 = int(w*0.25), int(w*0.75)
    roi = np.zeros_like(mask)
    roi[y1:y2, x1:x2] = 1
    hairline = np.logical_and(mask, roi)
    ys, xs = np.where(hairline)
    if len(xs) < 10:
        return "Not enough hairline points", y1
    unique_x = np.unique(xs)
    hairline_y = {x: min(ys[xs == x]) for x in unique_x}
    xs_s, ys_s = np.array(sorted(hairline_y.keys())), np.array([hairline_y[x] for x in sorted(hairline_y)])
    coeffs = np.polyfit(xs_s, ys_s, 2)
    a = coeffs[0]; dy = ys_s.max() - ys_s.min()
    asym = abs(xs_s[:len(xs_s)//2].mean() - xs_s[len(xs_s)//2:].mean())
    def classify(a, asym, dy):
        if a<-0.003 and asym>15: return "M-shape"
        if a<-0.003: return "Square"
        if a>0.003 and dy>20: return "V-shape"
        if a>0.003: return "Rounded"
        if abs(a)<=0.003 and dy<10: return "Flat"
        if dy>35 and asym>20: return "Crown/Irregular"
        return "Unclassified"
    shape = classify(a, asym, dy)
    plt.imshow(rgb_img)
    plt.scatter(xs_s, ys_s, s=2, c='red')
    plt.plot(xs_s, np.poly1d(coeffs)(xs_s), c='blue')
    plt.title(f"Hairline: {shape}")
    plt.axis('off')
    plt.savefig(out_path)
    plt.close()
    return shape, int(min(ys_s))

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