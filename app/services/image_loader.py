import requests
import cv2
import numpy as np

def url_to_image(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image.")
        return image
    except Exception as e:
        raise ValueError(f"Error loading image from URL: {e}")
