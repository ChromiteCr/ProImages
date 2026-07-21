import numpy as np

from proimages.core import denoise, depth_bokeh, hdr, physical_fx

STAGES = (denoise, hdr, depth_bokeh, physical_fx)


def process_image(image: np.ndarray) -> np.ndarray:
    for stage in STAGES:
        image = stage.process(image)
    return image
