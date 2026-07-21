import numpy as np

from proimages.core.physical_fx.grain import apply_grain


def test_apply_grain_changes_pixel_values():
    image = np.full((16, 16, 3), 0.5, dtype=np.float32)

    result = apply_grain(image, rng=np.random.default_rng(0))

    assert not np.array_equal(result, image)
    assert (result >= 0.0).all() and (result <= 1.0).all()


def test_apply_grain_is_reproducible_with_same_seed():
    image = np.full((16, 16, 3), 0.5, dtype=np.float32)

    result_a = apply_grain(image, rng=np.random.default_rng(42))
    result_b = apply_grain(image, rng=np.random.default_rng(42))

    assert np.array_equal(result_a, result_b)


def test_apply_grain_is_near_zero_at_extreme_luminance():
    black = np.zeros((16, 16, 3), dtype=np.float32)
    white = np.ones((16, 16, 3), dtype=np.float32)
    rng_seed = 0

    black_result = apply_grain(black, rng=np.random.default_rng(rng_seed))
    white_result = apply_grain(white, rng=np.random.default_rng(rng_seed))

    assert np.abs(black_result - black).max() < 1e-6
    assert np.abs(white_result - white).max() < 1e-6
