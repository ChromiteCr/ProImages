import numpy as np

from proimages.core.physical_fx.halation import apply_halation


def test_apply_halation_is_identity_below_threshold():
    image = np.full((16, 16, 3), 0.5, dtype=np.float32)

    result = apply_halation(image, threshold=0.75)

    assert np.allclose(result, image)


def test_apply_halation_adds_warm_glow_around_bright_spot():
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[14:18, 14:18, :] = 1.0

    result = apply_halation(image, threshold=0.75, radius=4.0, intensity=0.5)

    glow_region = result[8:12, 8:12, :]
    assert glow_region[..., 0].mean() > glow_region[..., 2].mean()
    assert (result >= 0.0).all() and (result <= 1.0).all()
