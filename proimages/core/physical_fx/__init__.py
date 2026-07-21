from pathlib import Path

import numpy as np

from proimages.core.physical_fx.effects import STAGES
from proimages.core.physical_fx.lut import apply_lut


def process(image: np.ndarray, lut_path: str | Path | None = None) -> np.ndarray:
    if lut_path is not None:
        image = apply_lut(image, lut_path)
    for stage in STAGES:
        image = stage(image)
    return image
