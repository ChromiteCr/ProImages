import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from proimages.core.system.tiling import tiled_apply

SMALL = dict(tile=256, overlap=32, multiple=32)


def random_image(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def identity(tile: np.ndarray) -> np.ndarray:
    return tile


def alternating_offsets(step: float = 0.05):
    """A fn that adds +step on its first call, -step on its second, and so on."""
    calls = 0

    def fn(tile: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return tile + (step if calls % 2 else -step)

    return fn


def largest_neighbour_steps(image: np.ndarray) -> tuple[float, float]:
    """The largest difference between vertically and between horizontally adjacent pixels."""
    return float(np.abs(np.diff(image, axis=0)).max()), float(np.abs(np.diff(image, axis=1)).max())


@pytest.mark.parametrize(
    "shape, kwargs",
    [
        ((700, 1000, 3), SMALL),
        ((700, 1000, 3), dict(tile=256, overlap=0, multiple=32)),
        ((300, 450), SMALL),
        ((300, 450, 4), SMALL),
        ((40, 1000, 3), SMALL),
        ((5, 7, 3), SMALL),
        ((1, 1, 3), SMALL),
        ((1, 9, 3), SMALL),
        ((9, 1, 3), SMALL),
        ((600, 700, 3), {}),
    ],
)
def test_an_identity_fn_reproduces_the_image(shape, kwargs):
    image = random_image(shape)

    result = tiled_apply(identity, image, **kwargs)

    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.abs(result - image).max() < 1e-6


@pytest.mark.parametrize(
    "shape, kwargs, tile_shape, count",
    [
        # padded 764 x 1064, stride 224: ceil((764 - 256) / 224) + 1 = 4 rows, ceil((1064 - 256) / 224) + 1 = 5 columns
        ((700, 1000, 3), SMALL, (256, 256, 3), 4 * 5),
        # no overlap means plain tiles: ceil(700 / 256) = 3 rows, ceil(1000 / 256) = 4 columns
        ((700, 1000, 3), dict(tile=256, overlap=0, multiple=32), (256, 256, 3), 3 * 4),
        # the defaults (512, 64, 64): padded 728 x 828, stride 448, two tiles each way
        ((600, 700, 3), {}, (512, 512, 3), 2 * 2),
        # a small image becomes one smaller tile: padded 69 x 71 rounds up to 96 x 96 ...
        ((5, 7, 3), SMALL, (96, 96, 3), 1),
        ((1, 1, 3), SMALL, (96, 96, 3), 1),
        # ... or 128 x 128 with a coarser multiple
        ((5, 7, 3), dict(tile=256, overlap=32, multiple=64), (128, 128, 3), 1),
        # the tile size is chosen per axis: padded height 104 rounds up to 128, padded width 1064 needs 5 full tiles
        ((40, 1000, 3), SMALL, (128, 256, 3), 5),
    ],
)
def test_tiles_have_sizes_divisible_by_multiple_and_the_grid_has_the_expected_count(shape, kwargs, tile_shape, count):
    shapes = []

    def record(tile: np.ndarray) -> np.ndarray:
        shapes.append(tile.shape)
        return tile

    tiled_apply(record, random_image(shape), **kwargs)

    multiple = kwargs.get("multiple", 64)
    assert len(shapes) == count
    assert set(shapes) == {tile_shape}
    assert all(height % multiple == 0 and width % multiple == 0 for height, width, _ in shapes)


@pytest.mark.parametrize("size", [48, 49, 50, 103, 104, 105, 159, 160, 161, 300])
def test_the_grid_covers_images_whose_padded_size_sits_on_a_tile_boundary(size):
    # tile 64 with overlap 8 has stride 56, so padded extents of 64, 120 and 176 (image sizes 48, 104 and 160) are
    # exact fits and the sizes just beside them need one tile less or more
    shapes = []

    def record(tile: np.ndarray) -> np.ndarray:
        shapes.append(tile.shape)
        return tile

    image = random_image((size, size), seed=6)

    result = tiled_apply(record, image, tile=64, overlap=8, multiple=8)

    tiles_per_axis = 1
    while (tiles_per_axis - 1) * 56 + 64 < size + 16:
        tiles_per_axis += 1
    assert len(shapes) == tiles_per_axis**2
    assert np.abs(result - image).max() < 1e-6


def test_per_tile_offsets_leave_no_visible_seam_where_tiles_overlap():
    # Both grids below are 3 x 3 tiles, so with tiles visited row by row (an odd number per row) every pair of
    # neighbouring tiles gets opposite offsets, 0.1 apart.
    flat = np.full((600, 600, 3), 0.5, dtype=np.float32)

    blended = tiled_apply(alternating_offsets(), flat, **SMALL)
    hard = tiled_apply(alternating_offsets(), flat, tile=256, overlap=0, multiple=32)

    # the same fn without overlap shows the 0.1 step along both axes, so this test can see a seam
    assert largest_neighbour_steps(hard) == pytest.approx((0.1, 0.1), abs=1e-6)
    # with it the 0.1 is spread over 32 px of raised cosine, at most 0.1 * pi / 64 = 0.005 per pixel
    assert max(largest_neighbour_steps(blended)) < 0.01


def test_the_overlap_is_cross_faded_with_a_raised_cosine_ramp():
    # Padded width 60 + 2 * 16 = 92 and stride 64 - 16 = 48 give two tiles that overlap on image columns 32..47. The
    # first tile adds 0 and the second adds 1, so there the output is exactly the second tile's ramp weight (the two
    # ramps sum to 1, so nothing is rescaled); to the left it is 0 and to the right 1.
    image = np.zeros((8, 60), dtype=np.float32)
    offsets = iter([0.0, 1.0])

    result = tiled_apply(lambda tile: tile + next(offsets), image, tile=64, overlap=16, multiple=16)

    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(16) + 0.5) / 16)
    expected = np.concatenate([np.zeros(32), ramp, np.ones(12)])
    assert np.abs(result - expected).max() < 1e-6


def test_a_context_dependent_fn_matches_the_whole_image_filter():
    def blur(tile: np.ndarray) -> np.ndarray:
        return uniform_filter(tile, size=(5, 5, 1), mode="mirror")  # "mirror" is numpy's "reflect"

    image = random_image((300, 350, 3), seed=1)

    error = np.abs(tiled_apply(blur, image, **SMALL) - blur(image))

    # The filter is wrong only within 2 px of a tile edge, where it sees a mirrored border instead of the real
    # neighbours. Those pixels carry the smallest feather weights (0.0006 at the edge, 0.5 * (1 - cos(3 * pi / 64)) =
    # 0.005 one pixel in) and the neighbouring tile supplies the rest, so the error is tiny next to the 0.24 by
    # which the filter changes this image on average.
    assert error.max() < 5e-3
    assert error.mean() < 1e-4


@pytest.mark.parametrize("shape", [(5, 7), (2, 3), (1, 9), (9, 1)])
def test_reflect_padding_gives_tiny_images_the_same_context_as_the_whole_image_filter(shape):
    def blur(tile: np.ndarray) -> np.ndarray:
        return uniform_filter(tile, size=5, mode="mirror")

    image = random_image(shape, seed=2)

    result = tiled_apply(blur, image, **SMALL)

    # the 32 px pad is wider than the image: numpy repeats the reflection, as scipy's "mirror" does, and pads a
    # 1-pixel axis with its own value
    assert np.abs(result - blur(image)).max() < 1e-6


def test_a_channel_changing_fn_gives_an_output_with_its_own_channel_count():
    image = random_image((300, 450, 3), seed=3)

    result = tiled_apply(lambda tile: tile.mean(axis=-1, keepdims=True), image, tile=128, overlap=16, multiple=16)

    assert result.shape == (300, 450, 1)
    assert np.abs(result - image.mean(axis=-1, keepdims=True)).max() < 1e-6


def test_a_2d_image_can_come_back_with_a_channel_axis():
    image = random_image((300, 450), seed=4)

    result = tiled_apply(lambda tile: np.stack([tile, 1.0 - tile], axis=-1), image, **SMALL)

    assert result.shape == (300, 450, 2)
    assert np.abs(result[..., 0] - image).max() < 1e-6
    assert np.abs(result[..., 1] - (1.0 - image)).max() < 1e-6


def test_tiles_and_result_are_float32_even_for_float64_input_and_output():
    dtypes = set()

    def to_float64(tile: np.ndarray) -> np.ndarray:
        dtypes.add(tile.dtype)
        return tile.astype(np.float64)

    result = tiled_apply(to_float64, random_image((100, 120, 3)).astype(np.float64), tile=64, overlap=8, multiple=8)

    assert dtypes == {np.dtype(np.float32)}
    assert result.dtype == np.float32


def test_a_fn_that_modifies_its_tile_in_place_does_not_disturb_the_other_tiles():
    def add_in_place(tile: np.ndarray) -> np.ndarray:
        tile += 0.25
        return tile

    image = random_image((300, 300, 3), seed=5)
    original = image.copy()

    result = tiled_apply(add_in_place, image, tile=128, overlap=16, multiple=16)

    assert np.abs(result - (original + 0.25)).max() < 1e-6
    assert np.array_equal(image, original)


def test_a_wrong_size_output_raises_value_error_naming_the_expected_and_actual_shapes():
    with pytest.raises(ValueError, match=r"\(64, 64\).*\(63, 64, 3\)"):
        tiled_apply(lambda tile: tile[:-1], random_image((100, 100, 3)), tile=64, overlap=8, multiple=8)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(tile=0, overlap=0, multiple=8), "positive"),
        (dict(tile=-64, overlap=0, multiple=8), "positive"),
        (dict(tile=64, overlap=0, multiple=0), "positive"),
        (dict(tile=64, overlap=0, multiple=-8), "positive"),
        (dict(tile=100, overlap=0, multiple=64), "multiple of"),
        (dict(tile=64, overlap=-1, multiple=8), "overlap"),
        (dict(tile=64, overlap=32, multiple=8), "overlap"),
        (dict(tile=64, overlap=40, multiple=8), "overlap"),
    ],
)
def test_invalid_arguments_raise_value_error_before_fn_runs(kwargs, message):
    def must_not_run(tile: np.ndarray) -> np.ndarray:
        raise AssertionError("fn ran")

    with pytest.raises(ValueError, match=message):
        tiled_apply(must_not_run, random_image((20, 20, 3)), **kwargs)


@pytest.mark.parametrize("shape", [(5,), (2, 4, 4, 3), (0, 6, 3), (6, 0)])
def test_images_that_are_not_hw_or_hwc_raise_value_error(shape):
    with pytest.raises(ValueError, match="image"):
        tiled_apply(identity, np.zeros(shape, dtype=np.float32))
