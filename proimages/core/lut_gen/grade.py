import numpy as np

from proimages.core.lut_gen.cdl import apply_cdl, to_cdl
from proimages.core.lut_gen.params import GradeParams


def apply_grade(rgb: np.ndarray, params: GradeParams) -> np.ndarray:
    """Apply a parametric color grade to any array shaped (..., 3).

    Works on both a photo (H, W, 3) and a LUT identity table (N, N, N, 3), so LUT
    baking and direct image grading stay on one implementation.
    """
    return apply_cdl(rgb, to_cdl(params))
