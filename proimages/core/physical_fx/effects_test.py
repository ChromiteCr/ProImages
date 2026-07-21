import numpy as np

from proimages.core.physical_fx import process
from proimages.core.physical_fx.effects import STAGES


def test_stages_include_the_four_stub_physical_effects():
    names = {stage.__name__ for stage in STAGES}
    assert names == {"apply_grain", "apply_halation", "apply_rolloff", "apply_vignette"}


def test_process_without_lut_only_changes_pixels_via_grain():
    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)

    result = process(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert (result >= 0.0).all() and (result <= 1.0).all()
