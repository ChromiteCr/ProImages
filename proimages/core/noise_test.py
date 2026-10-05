import numpy as np
import pytest

from proimages.core.noise import estimate_noise

SIZE = 512
# independent per-channel noise of std s has luma std s * sqrt(0.2126^2 + 0.7152^2 + 0.0722^2)
INDEPENDENT_RGB_FACTOR = float(np.sqrt(0.2126**2 + 0.7152**2 + 0.0722**2))


def gray_plus_shared_noise(sigma: float, seed: int = 0, size: int = SIZE) -> np.ndarray:
    """Flat mid gray plus the SAME Gaussian noise on all three channels, so the luma noise is exactly sigma."""
    noise = np.random.default_rng(seed).normal(0.0, sigma, (size, size, 1))
    return (0.5 + np.repeat(noise, 3, axis=2)).astype(np.float32)


def gray_to_rgb(plane: np.ndarray) -> np.ndarray:
    return np.repeat(plane[..., None], 3, axis=2).astype(np.float32)


def structured_plane(size: int = SIZE) -> np.ndarray:
    """A noise-free gradient with a sine texture and a few step edges (two rectangles and a disc)."""
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    plane = 0.15 + 0.4 * x + 0.1 * y
    plane += 0.05 * np.sin(2 * np.pi * 6 * x) * np.sin(2 * np.pi * 6 * y)
    plane[80:200, 100:260] += 0.2
    plane[300:420, 40:200] -= 0.1
    plane[(x - 0.7) ** 2 + (y - 0.65) ** 2 < 0.15**2] += 0.1
    return plane


@pytest.mark.parametrize("sigma", [0.005, 0.01, 0.02, 0.05])
def test_gaussian_luma_noise_is_estimated_within_five_percent(sigma):
    assert estimate_noise(gray_plus_shared_noise(sigma)) == pytest.approx(sigma, rel=0.05)


def test_independent_channel_noise_appears_as_0_749_times_sigma_in_luma():
    sigma = 0.02
    image = (0.5 + np.random.default_rng(1).normal(0.0, sigma, (SIZE, SIZE, 3))).astype(np.float32)

    assert INDEPENDENT_RGB_FACTOR == pytest.approx(0.749, abs=1e-3)
    assert estimate_noise(image) == pytest.approx(INDEPENDENT_RGB_FACTOR * sigma, rel=0.05)


@pytest.mark.parametrize("channel, weight", [(0, 0.2126), (1, 0.7152), (2, 0.0722)])
def test_noise_in_a_single_channel_is_scaled_by_its_rec709_weight(channel, weight):
    sigma = 0.05
    image = np.full((SIZE, SIZE, 3), 0.5, dtype=np.float32)
    image[..., channel] += np.random.default_rng(2).normal(0.0, sigma, (SIZE, SIZE)).astype(np.float32)

    assert estimate_noise(image) == pytest.approx(weight * sigma, rel=0.05)


def test_estimate_grows_strictly_with_the_noise_added_to_a_structured_image():
    plane = structured_plane()
    noise = np.random.default_rng(3).standard_normal(plane.shape)

    estimates = [estimate_noise(gray_to_rgb(plane + sigma * noise)) for sigma in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)]

    assert all(smaller < larger for smaller, larger in zip(estimates, estimates[1:]))


def test_a_noise_free_smooth_gradient_has_no_noise():
    y, x = np.mgrid[0:SIZE, 0:SIZE] / (SIZE - 1)
    gradient = 0.1 + 0.7 * x + 0.1 * y + 0.1 * x * y

    assert estimate_noise(gray_to_rgb(gradient)) < 1e-3
    assert estimate_noise(gradient.astype(np.float32)) < 1e-3


def test_step_edges_barely_move_the_estimate():
    """Diagonal stripes put a large Haar diagonal coefficient on every edge crossing (axis-aligned edges give
    none), which would wreck a standard-deviation estimate but only nudges the median."""
    sigma = 0.01
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    stripes = 0.2 + 0.5 * (((x + y) // 128) % 2)
    noise = np.random.default_rng(4).normal(0.0, sigma, stripes.shape)

    assert estimate_noise(gray_to_rgb(stripes + noise)) == pytest.approx(sigma, rel=0.10)


def test_a_2d_image_is_used_as_is():
    plane = gray_plus_shared_noise(0.02)[..., 0]

    assert plane.ndim == 2
    assert estimate_noise(plane) == pytest.approx(0.02, rel=0.05)
    assert estimate_noise(plane) == pytest.approx(estimate_noise(gray_to_rgb(plane)), rel=1e-5)


def test_float64_input_is_accepted():
    image = gray_plus_shared_noise(0.02).astype(np.float64)

    assert estimate_noise(image) == pytest.approx(0.02, rel=0.05)
    assert estimate_noise(image[..., 0]) == pytest.approx(0.02, rel=0.05)


def test_odd_sizes_drop_the_last_row_and_column():
    image = gray_plus_shared_noise(0.02, size=513)[:511]
    plane = image[..., 0]

    assert image.shape == (511, 513, 3)
    assert estimate_noise(image) == pytest.approx(0.02, rel=0.05)
    assert estimate_noise(plane) == estimate_noise(plane[:510, :512])


def test_a_2x2_image_gives_its_single_haar_coefficient_over_0_6745():
    plane = np.array([[0.25, 0.5], [0.75, 0.125]], dtype=np.float32)

    # HH = (0.25 - 0.5 - 0.75 + 0.125) / 2 = -0.4375, and the median of one |HH| is 0.4375
    assert estimate_noise(plane) == pytest.approx(0.4375 / 0.6745, rel=1e-6)
    assert estimate_noise(gray_to_rgb(plane)) == pytest.approx(0.4375 / 0.6745, rel=1e-5)


def test_the_last_odd_row_and_column_are_ignored():
    plane = np.array([[0.25, 0.5, 9.0], [0.75, 0.125, 9.0], [9.0, 9.0, 9.0]], dtype=np.float32)

    assert estimate_noise(plane) == pytest.approx(0.4375 / 0.6745, rel=1e-6)


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 5, 3), (5, 1, 3), (0, 4, 3), (1, 1), (1, 6), (6, 1)])
def test_images_smaller_than_2x2_have_no_noise(shape):
    result = estimate_noise(np.full(shape, 0.5, dtype=np.float32))

    assert result == 0.0
    assert type(result) is float


def test_the_estimate_is_a_python_float():
    assert type(estimate_noise(gray_plus_shared_noise(0.01, size=64))) is float


@pytest.mark.parametrize("shape", [(8,), (2, 4, 4, 3)])
def test_images_that_are_not_hw_or_hwc_raise_value_error(shape):
    with pytest.raises(ValueError, match="image"):
        estimate_noise(np.zeros(shape, dtype=np.float32))
