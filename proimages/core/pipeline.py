from pathlib import Path

import numpy as np

from proimages.core import denoise, depth_bokeh, hdr, physical_fx

STAGES = (denoise, hdr, depth_bokeh)


def process_image(image: np.ndarray, lut_path: str | Path | None = None) -> np.ndarray:
    for stage in STAGES:
        image = stage.process(image)
    return physical_fx.process(image, lut_path=lut_path)
