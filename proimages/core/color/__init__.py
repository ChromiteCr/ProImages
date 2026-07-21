import numpy as np

REC709_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(image: np.ndarray) -> np.ndarray:
    return image @ REC709_LUMA_WEIGHTS
