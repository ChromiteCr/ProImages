import numpy as np


def apply_rolloff(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    smoothstep = image**2 * (3.0 - 2.0 * image)
    return image + strength * (smoothstep - image)
