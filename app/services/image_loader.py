import os
import requests
import cv2
import numpy as np
from uuid import uuid4

def url_to_image(url: str, prefix: str = "front"):
    """
    Downloads an image from a URL, decodes it, saves it under ./tmp with a UUID-based filename,
    and returns both the image (OpenCV BGR) and its saved path.
    """
    try:
        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Decode image to OpenCV format
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image.")

        # Ensure tmp directory exists
        os.makedirs("./tmp", exist_ok=True)

        # Build unique filename
        ext = os.path.splitext(os.path.basename(url.split("?")[0]))[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"
        filename = f"{uuid4().hex}_{prefix}{ext}"
        save_path = os.path.join("./tmp", filename)

        # Save image locally
        cv2.imwrite(save_path, image)

        return image, save_path

    except Exception as e:
        raise ValueError(f"Error loading image from URL: {e}")
