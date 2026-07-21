import numpy as np

from proimages.core.physical_fx.vignette import apply_vignette


def test_apply_vignette_leaves_center_unchanged():
    image = np.ones((33, 33, 3), dtype=np.float32)

    result = apply_vignette(image)

    assert np.allclose(result[16, 16], 1.0, atol=1e-5)


def test_apply_vignette_darkens_corners_more_than_center():
    image = np.ones((33, 33, 3), dtype=np.float32)

    result = apply_vignette(image)

    assert result[0, 0, 0] < result[16, 16, 0]
    assert (result <= image).all()
