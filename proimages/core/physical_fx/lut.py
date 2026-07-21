from pathlib import Path

import colour
import numpy as np


def apply_lut(image: np.ndarray, lut_path: str | Path) -> np.ndarray:
    lut = colour.read_LUT(str(lut_path))
    return lut.apply(image).astype(np.float32).clip(0.0, 1.0)
