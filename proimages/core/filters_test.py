import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from proimages.core.color import REC709_LUMA_WEIGHTS, luminance
from proimages.core.filters import box, fast_guided_filter, guided_filter, guided_upsample

# (filter taking a guide and a src, factor by which the src is smaller than the guide)
FILTERS = [
    pytest.param(lambda guide, src: guided_filter(guide, src, 2, 0.01), 1, id="guided_filter"),
    pytest.param(lambda guide, src: fast_guided_filter(guide, src, 2, 0.01, 2), 1, id="fast_guided_filter"),
    pytest.param(lambda guide, src: guided_upsample(guide, src, 2, 0.01), 2, id="guided_upsample"),
]

# the same, for the paths that resample: the inputs are box-downsampled and the coefficients bilinearly upsampled
RESAMPLING_FILTERS = [
    pytest.param(lambda guide, src: fast_guided_filter(guide, src, 4, 0.01, 2), 1, id="fast_guided_filter_s2"),
    pytest.param(lambda guide, src: fast_guided_filter(guide, src, 8, 0.01, 4), 1, id="fast_guided_filter_s4"),
    pytest.param(lambda guide, src: guided_upsample(guide, src, 2, 0.01), 2, id="guided_upsample"),
]


def box_down(x: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Box-downsample a float32 (H, W) map to size = (h, w) the way filters.py does, with Pillow's mode "F"."""
    return np.asarray(Image.fromarray(x).resize((size[1], size[0]), Image.Resampling.BOX))


def brute_force_box(x: np.ndarray, r: int) -> np.ndarray:
    """Window mean in float64 straight from the definition: pad by half-sample symmetric
    reflection, then average every (2r+1)x(2r+1) window."""
    padding = [(r, r), (r, r)] + [(0, 0)] * (x.ndim - 2)
    padded = np.pad(np.asarray(x, dtype=np.float64), padding, mode="symmetric")
    return sliding_window_view(padded, (2 * r + 1, 2 * r + 1), axis=(0, 1)).mean(axis=(-2, -1))


def reference_gray_guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Algorithm 1 of He, Sun & Tang (TPAMI 2013) in float64, one src channel at a time."""
    guide = np.asarray(guide, dtype=np.float64)
    channels = np.asarray(src, dtype=np.float64).reshape(*src.shape[:2], -1)
    mean_i = brute_force_box(guide, r)
    var_i = brute_force_box(guide * guide, r) - mean_i * mean_i
    out = np.empty_like(channels)

    for c in range(channels.shape[-1]):
        p = channels[..., c]
        mean_p = brute_force_box(p, r)
        a = (brute_force_box(guide * p, r) - mean_i * mean_p) / (var_i + eps)
        b = mean_p - a * mean_i
        out[..., c] = brute_force_box(a, r) * guide + brute_force_box(b, r)

    return out.reshape(src.shape)


def reference_color_guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Eq. 14-16 of the paper in float64, solving the 3x3 system pixel by pixel."""
    guide = np.asarray(guide, dtype=np.float64)
    channels = np.asarray(src, dtype=np.float64).reshape(*src.shape[:2], -1)
    mean_i = brute_force_box(guide, r)
    sigma = brute_force_box(guide[..., :, None] * guide[..., None, :], r) - mean_i[..., :, None] * mean_i[..., None, :]
    out = np.empty_like(channels)

    for c in range(channels.shape[-1]):
        p = channels[..., c]
        mean_p = brute_force_box(p, r)
        cov_ip = brute_force_box(guide * p[..., None], r) - mean_i * mean_p[..., None]
        a = np.empty_like(guide)
        for y in range(guide.shape[0]):
            for x in range(guide.shape[1]):
                a[y, x] = np.linalg.solve(sigma[y, x] + eps * np.eye(3), cov_ip[y, x])
        b = mean_p - (a * mean_i).sum(axis=-1)
        out[..., c] = (brute_force_box(a, r) * guide).sum(axis=-1) + brute_force_box(b, r)

    return out.reshape(src.shape)


def smooth_field(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """A smooth random field scaled to [0, 1]."""
    field = gaussian_filter(rng.standard_normal((height, width)), 3)
    return (field - field.min()) / (field.max() - field.min())


def step_image(height: int, width: int, edge: int, left: float, right: float) -> np.ndarray:
    """A float32 image that is `left` before column `edge` and `right` from it on."""
    return np.tile(np.where(np.arange(width) < edge, left, right), (height, 1)).astype(np.float32)


def step_height(x: np.ndarray, edge: int) -> float:
    """Mean of the two columns right of the edge minus the mean of the two columns left of it."""
    return float(x[:, edge : edge + 2].mean() - x[:, edge - 2 : edge].mean())


@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("shape", [(9, 11), (9, 11, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_box_matches_a_brute_force_window_mean(dtype, shape: tuple[int, ...], r: int):
    x = np.random.default_rng(0).random(shape).astype(dtype)

    result = box(x, r)

    assert result.dtype == np.float32 and result.shape == x.shape
    assert np.allclose(result, brute_force_box(x, r), rtol=0, atol=1e-6)


def test_box_reflects_by_half_samples_at_the_border():
    x = np.zeros((5, 5), dtype=np.float32)
    x[0, 0] = 9.0

    # 'd c b a | a b c d': the corner sample fills 4 of the 9 window cells; a mirror border would give 1
    assert box(x, 1)[0, 0] == pytest.approx(4.0)


def test_box_keeps_channels_independent():
    rng = np.random.default_rng(1)
    x = np.zeros((9, 11, 3), dtype=np.float32)
    x[..., 1] = rng.random((9, 11))

    result = box(x, 2)

    assert np.array_equal(result[..., 0], np.zeros((9, 11))) and np.array_equal(result[..., 2], np.zeros((9, 11)))
    assert np.allclose(result[..., 1], brute_force_box(x[..., 1], 2), rtol=0, atol=1e-6)


@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("src_shape", [(24, 28), (24, 28, 3)])
def test_guided_filter_with_a_gray_guide_matches_the_paper(src_shape: tuple[int, ...], r: int):
    rng = np.random.default_rng(2)
    guide = rng.random((24, 28)).astype(np.float32)
    src = rng.random(src_shape).astype(np.float32)

    result = guided_filter(guide, src, r, eps=0.01)

    assert np.allclose(result, reference_gray_guided_filter(guide, src, r, 0.01), rtol=0, atol=1e-5)


def test_guided_filter_with_a_gray_guide_is_accurate_on_log_domain_values():
    # A dark log2 guide (about -9..-8) steering a map that spans -9..-1. Without removing the means first,
    # var = box(I*I) - box(I)^2 cancels in float32 and the result is off by ~7e-5; with it, by ~8e-7.
    # (A bound of 1e-3 would not notice the difference.)
    rng = np.random.default_rng(3)
    field = smooth_field(rng, 24, 32)
    guide = (-8.9 + field + 0.03 * rng.standard_normal(field.shape)).astype(np.float32)
    src = (-9.0 + 8.0 * field + 0.02 * rng.standard_normal(field.shape)).clip(-9.0, -1.0).astype(np.float32)

    result = guided_filter(guide, src, 3, eps=0.01)

    assert np.allclose(result, reference_gray_guided_filter(guide, src, 3, 0.01), rtol=0, atol=1e-5)


@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("src_shape", [(16, 20), (16, 20, 2)])
def test_guided_filter_with_a_color_guide_matches_the_paper(src_shape: tuple[int, ...], r: int):
    rng = np.random.default_rng(4)
    guide = rng.random((16, 20, 3)).astype(np.float32)
    src = rng.random(src_shape).astype(np.float32)

    result = guided_filter(guide, src, r, eps=1e-3)

    assert np.allclose(result, reference_color_guided_filter(guide, src, r, 1e-3), rtol=0, atol=1e-5)


def test_guided_filter_with_a_color_guide_is_accurate_on_log_domain_values():
    # Nearly proportional dark log2 channels: the covariance is close to singular and the means are large.
    # Without removing the means first the result is off by ~2e-4; with it, by ~6e-7.
    rng = np.random.default_rng(5)
    field = smooth_field(rng, 16, 20)
    guide = (-8.9 + 0.3 * field)[..., None] * [1.0, 0.98, 1.02] + 0.01 * rng.standard_normal((16, 20, 3))
    src = np.stack([-9.0 + 8.0 * field, -1.0 - 8.0 * field], axis=-1) + 0.02 * rng.standard_normal((16, 20, 2))
    guide, src = guide.astype(np.float32), src.clip(-9.0, -1.0).astype(np.float32)

    result = guided_filter(guide, src, 3, eps=0.01)

    assert np.allclose(result, reference_color_guided_filter(guide, src, 3, 0.01), rtol=0, atol=1e-5)


def test_guided_filter_with_a_nearly_neutral_color_guide_is_accurate_at_a_small_eps():
    # R = G = B plus noise of 0.001 per channel leaves Sigma nearly singular. At eps=1e-6 a float32 cofactor
    # inverse is off by 0.66 on this guide; the float64 one by 1.3e-4.
    rng = np.random.default_rng(18)
    texture = smooth_field(rng, 48, 64)
    guide = (texture[..., None] + 0.001 * rng.standard_normal((48, 64, 3))).astype(np.float32)
    src = rng.random((48, 64)).astype(np.float32)

    result = guided_filter(guide, src, 3, eps=1e-6)

    assert np.allclose(result, reference_color_guided_filter(guide, src, 3, 1e-6), rtol=0, atol=1e-3)


def test_guided_filter_keeps_a_noisy_step_sharp_while_smoothing_the_noise():
    height, width, edge, r = 96, 192, 96, 4
    rng = np.random.default_rng(6)
    clean = step_image(height, width, edge, 0.2, 0.8)
    image = (clean + rng.normal(0.0, 0.03, clean.shape)).astype(np.float32)
    flat = np.abs(np.arange(width) - edge) > 2 * r + 4

    smoothed = guided_filter(image, image, r, eps=1e-3)
    blurred = box(image, r)

    assert step_height(smoothed, edge) > 0.5
    assert step_height(blurred, edge) < 0.25  # a plain box of the same radius smears the same step
    assert (image - clean)[:, flat].std() >= 2.0 * (smoothed - clean)[:, flat].std()


def test_a_color_guide_sees_an_edge_that_the_luminance_cannot():
    height, width, edge = 40, 64, 32
    rng = np.random.default_rng(7)
    red = np.array([0.2 / REC709_LUMA_WEIGHTS[0], 0.0, 0.0])
    green = np.array([0.0, 0.2 / REC709_LUMA_WEIGHTS[1], 0.0])
    image = np.where(np.arange(width)[None, :, None] < edge, red, green) + rng.normal(0.0, 0.01, (height, width, 3))
    image = image.astype(np.float32)
    depth = step_image(height, width, edge, 0.25, 0.75)
    gray = luminance(image)
    assert abs(gray[:, edge:].mean() - gray[:, :edge].mean()) < 0.01  # same luminance on both sides

    color_guided = guided_filter(image, depth, 4, eps=1e-3)
    luminance_guided = guided_filter(gray, depth, 4, eps=1e-3)

    assert step_height(color_guided, edge) > 0.45  # the true step is 0.5
    assert step_height(luminance_guided, edge) < 0.2


@pytest.mark.parametrize("guide_shape", [(20, 24), (20, 24, 3)])
def test_fast_guided_filter_with_s_1_is_the_guided_filter(guide_shape: tuple[int, ...]):
    rng = np.random.default_rng(8)
    guide = rng.random(guide_shape).astype(np.float32)
    src = rng.random((20, 24, 2)).astype(np.float32)

    fast = fast_guided_filter(guide, src, 3, 0.01, s=1)

    assert np.allclose(fast, guided_filter(guide, src, 3, 0.01), rtol=0, atol=1e-6)


@pytest.mark.parametrize("shape", [(96, 128), (101, 133)])
def test_fast_guided_filter_with_s_4_stays_close_to_the_exact_filter_on_a_smooth_image(shape: tuple[int, int]):
    rng = np.random.default_rng(9)
    guide = smooth_field(rng, *shape).astype(np.float32)
    src = (guide + rng.normal(0.0, 0.02, guide.shape)).astype(np.float32)

    fast = fast_guided_filter(guide, src, 8, 0.01, s=4)
    exact = guided_filter(guide, src, 8, 0.01)

    assert np.abs(fast - exact).mean() < 0.01
    assert np.abs(fast - exact).mean() < 0.25 * np.abs(src - exact).mean()  # small next to what the filter does


def test_fast_guided_filter_gives_a_ramp_back_without_block_seams():
    # q = a*I + b with a ramp as both guide and src is the ramp again, provided the coefficients are
    # interpolated; replicating them per block would leave a seam of about 4e-3 at every block border.
    ramp = np.tile(np.linspace(0.0, 1.0, 128, dtype=np.float32), (64, 1))

    result = fast_guided_filter(ramp, ramp, 8, 1e-3, s=4)

    assert np.abs(result - ramp)[:, 24:-24].max() < 1e-4  # the border band is excluded: reflection bends the ramp


def test_fast_guided_filter_averages_a_pattern_finer_than_the_subsampling_grid():
    # Each 2x2 box of a pixel-level checkerboard averages to 0.5; decimating instead would alias it to 0 or 1
    checkerboard = (np.indices((16, 20)).sum(axis=0) % 2).astype(np.float32)

    result = fast_guided_filter(checkerboard, checkerboard, 2, 0.01, s=2)

    assert np.allclose(result, 0.5, rtol=0, atol=1e-6)


def test_fast_guided_filter_survives_a_factor_larger_than_the_image():
    rng = np.random.default_rng(10)
    guide, src = rng.random((6, 8)), rng.random((6, 8, 2))

    result = fast_guided_filter(guide, src, 4, 0.01, s=16)

    # the coefficient grid is 1x1, which has no variance to fit: a = 0 and each channel comes back as its mean
    assert result.shape == (6, 8, 2) and np.isfinite(result).all()
    assert np.allclose(result, src.mean(axis=(0, 1)), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "height, width, r, low_size, low_r",
    [
        (103, 79, 6, (26, 20), 2),  # 25.75, 19.75 and 1.5 round up; flooring them gives 25, 19 and 1
        (101, 73, 5, (25, 18), 1),  # 25.25, 18.25 and 1.25 round down; ceiling them gives 26, 19 and 2
    ],
)
def test_fast_guided_filter_rounds_the_low_resolution_size_and_radius(height, width, r, low_size, low_r):
    rng = np.random.default_rng(20)
    guide = rng.random((height, width)).astype(np.float32)
    src = rng.random((height, width)).astype(np.float32)

    fast = fast_guided_filter(guide, src, r, 0.01, s=4)
    by_hand = guided_upsample(guide, box_down(src, low_size), low_r, 0.01)

    assert np.allclose(fast, by_hand, rtol=0, atol=1e-6)


def test_guided_upsample_snaps_a_blurry_depth_edge_to_the_guide_edge():
    height, width, edge, factor = 96, 128, 64, 8
    rng = np.random.default_rng(11)
    truth = step_image(height, width, edge, 0.2, 0.8)
    low = truth.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))
    left, right = np.array([0.30, 0.35, 0.40]), np.array([0.70, 0.60, 0.50])
    guide = np.where(np.arange(width)[None, :, None] < edge, left, right) + rng.normal(0.0, 0.02, (height, width, 3))
    guide = guide.astype(np.float32)

    result = guided_upsample(guide, low, 2, eps=1e-3)
    bilinear = zoom(low, factor, order=1, mode="nearest", grid_mode=True)

    band = slice(edge - factor, edge + factor)
    assert result.shape == guide.shape[:2] and bilinear.shape == truth.shape
    assert np.abs(result - truth)[:, band].mean() <= 0.5 * np.abs(bilinear - truth)[:, band].mean()


@pytest.mark.parametrize("guide_shape", [(20, 24), (20, 24, 3)])
def test_guided_upsample_at_equal_size_is_the_guided_filter(guide_shape: tuple[int, ...]):
    rng = np.random.default_rng(12)
    guide = rng.random(guide_shape).astype(np.float32)
    src = rng.random((20, 24)).astype(np.float32)

    result = guided_upsample(guide, src, 2, 0.01)

    assert np.allclose(result, guided_filter(guide, src, 2, 0.01), rtol=0, atol=1e-6)


@pytest.mark.parametrize("channels", [2, 3])
@pytest.mark.parametrize("guide_extra", [(), (3,)])
@pytest.mark.parametrize("apply, scale", RESAMPLING_FILTERS)
def test_each_src_channel_is_filtered_independently_of_the_others(apply, scale: int, guide_extra, channels: int):
    rng = np.random.default_rng(19)
    height, width = 32 // scale, 40 // scale
    # channels that differ in content and in mean, so a swap or a mix-up shows in either
    maps = [
        rng.random((height, width)),
        1.5 + smooth_field(rng, height, width),
        -1.0 + 0.1 * rng.random((height, width)),
    ]
    src = np.stack(maps[:channels], axis=-1).astype(np.float32)
    guide = rng.random((32, 40) + guide_extra).astype(np.float32)

    result = apply(guide, src)

    for c in range(channels):
        assert np.allclose(result[..., c], apply(guide, src[..., c]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("src_extra", [(), (1,), (2,)])
@pytest.mark.parametrize("guide_extra", [(), (3,)])
@pytest.mark.parametrize("apply, scale", FILTERS)
def test_filters_return_float32_shaped_like_src(apply, scale: int, guide_extra, src_extra, dtype):
    rng = np.random.default_rng(13)
    guide = rng.random((16, 20) + guide_extra).astype(dtype)
    src = rng.random((16 // scale, 20 // scale) + src_extra).astype(dtype)

    result = apply(guide, src)

    assert result.dtype == np.float32
    assert result.shape == (16, 20) + src_extra
    assert np.isfinite(result).all()


def test_filters_take_numpy_scalars_for_r_eps_and_s():
    rng = np.random.default_rng(16)
    guide, src = rng.random((16, 20, 3)).astype(np.float32), rng.random((16, 20)).astype(np.float32)
    r, eps, s = np.int64(2), np.float64(0.01), np.int64(2)

    for result, expected in [
        (guided_filter(guide, src, r, eps), guided_filter(guide, src, 2, 0.01)),
        (fast_guided_filter(guide, src, r, eps, s), fast_guided_filter(guide, src, 2, 0.01, 2)),
        (guided_upsample(guide, src[::2, ::2], r, eps), guided_upsample(guide, src[::2, ::2], 2, 0.01)),
    ]:
        assert result.dtype == np.float32
        assert np.allclose(result, expected, rtol=0, atol=1e-6)


def test_filters_accept_channel_slices_and_other_non_contiguous_views():
    image = np.random.default_rng(17).random((16, 20, 3)).astype(np.float32)
    guide, chroma = image[..., 0], image[..., 1:3]  # views into one array, neither is contiguous

    for view_result, copy_result in [
        (guided_filter(guide, chroma, 2, 0.01), guided_filter(guide.copy(), chroma.copy(), 2, 0.01)),
        (fast_guided_filter(image, chroma, 2, 0.01, 2), fast_guided_filter(image.copy(), chroma.copy(), 2, 0.01, 2)),
        (guided_upsample(image, chroma[::2, ::2], 2, 0.01), guided_upsample(image, chroma[::2, ::2].copy(), 2, 0.01)),
    ]:
        assert view_result.shape == (16, 20, 2)
        assert np.array_equal(view_result, copy_result)


@pytest.mark.parametrize("guide_level, src_level", [(0.3, 0.3), (0.5, -8.0), (-8.0, 0.25)])
@pytest.mark.parametrize("guide_extra", [(), (3,)])
@pytest.mark.parametrize("apply, scale", FILTERS)
def test_a_constant_image_comes_back_constant(apply, scale: int, guide_extra, guide_level: float, src_level: float):
    guide = np.full((16, 20) + guide_extra, guide_level, dtype=np.float32)
    src = np.full((16 // scale, 20 // scale), src_level, dtype=np.float32)

    result = apply(guide, src)

    assert np.isfinite(result).all()
    assert np.allclose(result, src_level, rtol=0, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("apply, scale", FILTERS)
def test_filters_do_not_modify_their_inputs(apply, scale: int, dtype):
    rng = np.random.default_rng(14)
    guide = rng.random((16, 20, 3)).astype(dtype)
    src = rng.random((16 // scale, 20 // scale, 2)).astype(dtype)
    guide_before, src_before = guide.copy(), src.copy()

    apply(guide, src)

    assert guide.dtype == dtype and src.dtype == dtype
    assert np.array_equal(guide, guide_before) and np.array_equal(src, src_before)


def test_box_does_not_modify_its_input():
    x = np.random.default_rng(15).random((9, 11, 3))
    before = x.copy()

    box(x, 2)

    assert np.array_equal(x, before)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"r": 0}, "r must be"),
        ({"r": -2}, "r must be"),
        ({"eps": 0.0}, "eps must be"),
        ({"eps": -0.01}, "eps must be"),
        ({"eps": float("nan")}, "eps must be"),
        ({"guide": np.zeros((16, 20, 2), dtype=np.float32)}, "guide must be"),
        ({"guide": np.zeros((16, 20, 4), dtype=np.float32)}, "guide must be"),
        ({"guide": np.zeros((16, 20, 3, 1), dtype=np.float32)}, "guide must be"),
        ({"guide": np.zeros(16, dtype=np.float32)}, "guide must be"),
        ({"src": np.zeros(16, dtype=np.float32)}, "src must be"),
        ({"src": np.zeros((16, 20, 2, 1), dtype=np.float32)}, "src must be"),
    ],
)
def test_the_filters_reject_invalid_arguments(overrides: dict, message: str):
    arguments = {
        "guide": np.zeros((16, 20), dtype=np.float32),
        "src": np.zeros((16, 20), dtype=np.float32),
        "r": 2,
        "eps": 0.01,
    } | overrides

    with pytest.raises(ValueError, match=message):
        guided_filter(**arguments)
    with pytest.raises(ValueError, match=message):
        fast_guided_filter(**arguments, s=2)
    with pytest.raises(ValueError, match=message):
        guided_upsample(**arguments)


def test_fast_guided_filter_rejects_a_factor_below_one():
    image = np.zeros((16, 20), dtype=np.float32)

    with pytest.raises(ValueError, match="s must be"):
        fast_guided_filter(image, image, 2, 0.01, s=0)


@pytest.mark.parametrize("src_shape", [(16, 21), (15, 20), (8, 10)])
def test_same_resolution_filters_reject_a_src_of_another_size(src_shape: tuple[int, ...]):
    guide, src = np.zeros((16, 20), dtype=np.float32), np.zeros(src_shape, dtype=np.float32)

    with pytest.raises(ValueError, match="same height and width"):
        guided_filter(guide, src, 2, 0.01)
    with pytest.raises(ValueError, match="same height and width"):
        fast_guided_filter(guide, src, 2, 0.01, s=2)


@pytest.mark.parametrize("src_shape", [(17, 20), (16, 21), (32, 40)])
def test_guided_upsample_rejects_a_src_larger_than_the_guide(src_shape: tuple[int, ...]):
    with pytest.raises(ValueError, match="larger than the guide"):
        guided_upsample(np.zeros((16, 20, 3), dtype=np.float32), np.zeros(src_shape, dtype=np.float32), 2, 0.01)


def test_box_rejects_a_radius_below_one():
    with pytest.raises(ValueError, match="r must be"):
        box(np.zeros((8, 8), dtype=np.float32), 0)
