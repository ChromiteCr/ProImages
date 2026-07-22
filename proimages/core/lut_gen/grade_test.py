import numpy as np

from proimages.core.lut_gen.grade import apply_grade
from proimages.core.lut_gen.params import GradeParams


def test_default_params_are_identity():
    image = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)

    result = apply_grade(image, GradeParams())

    assert np.allclose(result, image, atol=1e-6)


def test_positive_temperature_warms_the_image():
    image = np.full((4, 4, 3), 0.5, dtype=np.float32)

    result = apply_grade(image, GradeParams(temperature=1.0))

    assert result[0, 0, 0] > 0.5
    assert result[0, 0, 2] < 0.5


def test_full_desaturation_makes_channels_equal():
    image = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)

    result = apply_grade(image, GradeParams(saturation=-1.0))

    assert np.allclose(result[..., 0], result[..., 1], atol=1e-6)
    assert np.allclose(result[..., 1], result[..., 2], atol=1e-6)


def test_output_stays_in_range_for_extreme_params():
    image = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    extreme = GradeParams(
        tone=1.0,
        saturation=1.0,
        temperature=1.0,
        tint=-1.0,
        contrast=1.0,
        lift=(1.0, -1.0, 1.0),
        gamma=(-1.0, 1.0, -1.0),
        gain=(1.0, 1.0, -1.0),
    )

    result = apply_grade(image, extreme)

    assert (result >= 0.0).all() and (result <= 1.0).all()
    assert result.dtype == np.float32


def test_works_on_a_lut_shaped_array():
    table = np.random.default_rng(0).random((4, 4, 4, 3)).astype(np.float32)

    result = apply_grade(table, GradeParams(temperature=0.5))

    assert result.shape == table.shape
