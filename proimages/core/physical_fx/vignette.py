import numpy as np


def apply_vignette(image: np.ndarray, strength: float = 0.6) -> np.ndarray:
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    max_radius = float(np.hypot(center_x, center_y))

    radius = np.hypot(x - center_x, y - center_y) / max_radius
    falloff = np.cos(radius * strength) ** 4

    return (image * falloff[..., None]).clip(0.0, 1.0)
