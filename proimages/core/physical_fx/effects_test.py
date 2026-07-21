import numpy as np

from proimages.core.physical_fx import process
from proimages.core.physical_fx.effects import STAGES


def test_stages_include_all_five_physical_effects():
    names = {stage.__name__ for stage in STAGES}
    assert names == {"apply_lut", "apply_grain", "apply_halation", "apply_rolloff", "apply_vignette"}


def test_process_is_identity_for_stub_stages():
    image = np.random.default_rng(0).random((8, 8, 3), dtype=np.float32)

    result = process(image)

    assert np.array_equal(result, image)
