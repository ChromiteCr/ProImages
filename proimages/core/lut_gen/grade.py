import numpy as np

from proimages.core.color import luminance
from proimages.core.lut_gen.params import GradeParams

TEMPERATURE_GAIN = np.array([1.0, 0.0, -1.0], dtype=np.float32)
TINT_GAIN = np.array([0.5, -1.0, 0.5], dtype=np.float32)

TEMPERATURE_SCALE = 0.25
TINT_SCALE = 0.15
LIFT_SCALE = 0.25
GAMMA_SCALE = 0.5
GAIN_SCALE = 0.35
TONE_SCALE = 0.35
CONTRAST_SCALE = 0.8


def apply_grade(rgb: np.ndarray, params: GradeParams) -> np.ndarray:
    """Apply a parametric color grade to any array shaped (..., 3).

    Works on both a photo (H, W, 3) and a LUT identity table (N, N, N, 3), so LUT
    baking and direct image grading stay on one implementation.
    """
    graded = rgb.astype(np.float32)

    white_balance = 1.0 + params.temperature * TEMPERATURE_SCALE * TEMPERATURE_GAIN
    white_balance = white_balance + params.tint * TINT_SCALE * TINT_GAIN
    graded = graded * white_balance

    lift = np.asarray(params.lift, dtype=np.float32) * LIFT_SCALE
    gain = 1.0 + np.asarray(params.gain, dtype=np.float32) * GAIN_SCALE
    graded = lift + graded * (gain - lift)

    gamma_exponent = 1.0 / (1.0 + np.asarray(params.gamma, dtype=np.float32) * GAMMA_SCALE)
    graded = np.power(graded.clip(0.0, None), gamma_exponent)

    graded = 0.5 + (graded - 0.5) * (1.0 + params.contrast * CONTRAST_SCALE)
    graded = graded + params.tone * TONE_SCALE

    graded = graded.clip(0.0, 1.0)
    luma = luminance(graded)[..., None]
    graded = luma + (graded - luma) * (1.0 + params.saturation)

    return graded.clip(0.0, 1.0).astype(np.float32)
