from pathlib import Path

import numpy as np

from proimages.core import physical_fx
from proimages.core.options import ProcessOptions


def process_image(
    image: np.ndarray,
    lut_path: str | Path | None = None,
    options: ProcessOptions | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Run a photo through the pipeline and return the result.

    The optional stages (core/denoise -> core/hdr -> core/depth_bokeh, in that order) run only
    when `options` sets their sub-options. None of them is implemented yet, so every photo goes
    straight to the LUT and the physical film effects. `device` ("cuda", "mps" or "cpu") is
    where model-backed stages run; None lets them detect it.
    """
    # A dict from a Python caller is validated, not ignored, although no stage reads the options yet.
    options = ProcessOptions() if options is None else ProcessOptions.model_validate(options)
    return physical_fx.process(image, lut_path=lut_path)
