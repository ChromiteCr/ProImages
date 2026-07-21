import numpy as np

from proimages.core.physical_fx.rolloff import apply_rolloff


def test_apply_rolloff_leaves_black_white_and_midgray_unchanged():
    image = np.array([0.0, 0.5, 1.0], dtype=np.float32).reshape(1, 3, 1).repeat(3, axis=2)

    result = apply_rolloff(image)

    assert np.allclose(result, image, atol=1e-6)


def test_apply_rolloff_increases_contrast():
    image = np.array([0.25, 0.75], dtype=np.float32).reshape(1, 2, 1).repeat(3, axis=2)

    result = apply_rolloff(image, strength=1.0)

    assert result[0, 0, 0] < 0.25
    assert result[0, 1, 0] > 0.75


def test_apply_rolloff_strength_zero_is_identity():
    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)

    result = apply_rolloff(image, strength=0.0)

    assert np.allclose(result, image)
