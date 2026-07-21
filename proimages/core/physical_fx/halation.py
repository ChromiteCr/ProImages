import numpy as np
from scipy.ndimage import gaussian_filter

from proimages.core.color import luminance

HALATION_TINT = np.array([1.0, 0.45, 0.25], dtype=np.float32)


def apply_halation(image: np.ndarray, threshold: float = 0.75, radius: float = 8.0, intensity: float = 0.35) -> np.ndarray:
    highlight_mask = np.clip((luminance(image) - threshold) / (1.0 - threshold), 0.0, 1.0)
    glow = gaussian_filter(highlight_mask, sigma=radius)
    tinted_glow = glow[..., None] * HALATION_TINT
    return (image + intensity * tinted_glow).clip(0.0, 1.0)
