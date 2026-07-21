import numpy as np

from proimages.core.color import luminance


def test_luminance_of_black_and_white():
    black = np.zeros((4, 4, 3), dtype=np.float32)
    white = np.ones((4, 4, 3), dtype=np.float32)

    assert np.allclose(luminance(black), 0.0)
    assert np.allclose(luminance(white), 1.0)


def test_luminance_weighs_green_the_most():
    red = np.zeros((1, 1, 3), dtype=np.float32)
    red[..., 0] = 1.0
    green = np.zeros((1, 1, 3), dtype=np.float32)
    green[..., 1] = 1.0

    assert luminance(green)[0, 0] > luminance(red)[0, 0]
