import numpy as np

from proimages.core.physical_fx.effects import STAGES


def process(image: np.ndarray) -> np.ndarray:
    for stage in STAGES:
        image = stage(image)
    return image
