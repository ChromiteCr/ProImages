import numpy as np

from proimages.core import denoise, depth_bokeh, hdr, lut_grain

STAGES = (denoise, hdr, depth_bokeh, lut_grain)


def process_image(image: np.ndarray) -> np.ndarray:
    for stage in STAGES:
        image = stage.process(image)
    return image
