import cv2
import numpy as np

def draw_landmarks_with_indices(
    image: np.ndarray,
    landmarks_px: np.ndarray,
    out_path: str,
    scale: float = 1.5,
):
    """
    Draw ALL landmarks with their indices.

    - image: BGR image
    - landmarks_px: (N, 2) or (N, 3) array of pixel coords
    - out_path: file path for saving
    - scale: upscales image & coords to make numbers readable
    """
    h, w = image.shape[:2]

    # optionally upscale for readability
    if scale != 1.0:
        debug = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        pts = landmarks_px.copy().astype(np.float32)
        pts[:, 0] *= scale
        pts[:, 1] *= scale
    else:
        debug = image.copy()
        pts = landmarks_px

    for idx, pt in enumerate(pts):
        x = int(pt[0])
        y = int(pt[1])

        # point
        cv2.circle(debug, (x, y), 2, (0, 255, 0), -1)

        # index text (for ALL landmarks)
        cv2.putText(
            debug,
            str(idx),
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,            # bump if still too small
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(out_path, debug)
    print(f"[DEBUG] Numbered landmarks image saved to {out_path}")
