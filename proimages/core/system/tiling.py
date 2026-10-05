from collections.abc import Callable

import numpy as np


def tiled_apply(
    fn: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    tile: int = 512,
    overlap: int = 64,
    multiple: int = 64,
) -> np.ndarray:
    """Run `fn` over overlapping tiles of `image` and feather the results back together.

    `image` is (H, W) or (H, W, C), converted to float32; numpy in and out, so `fn` does any torch conversion
    itself. The whole image is first reflect-padded by `overlap` on every side, plus whatever the bottom and
    right need to complete the tile grid, so border pixels get the same context as interior pixels.

    Tiles are `tile` pixels along each axis and overlap their neighbours by `overlap` pixels (stride
    `tile - overlap`). An axis whose padded extent is smaller than `tile` gets a single smaller tile instead,
    its extent rounded up to a multiple of `multiple`. Every tile passed to `fn`, in row-major order, has
    height and width divisible by `multiple` and is a float32 copy that `fn` may modify in place.

    `fn` must return an array with the tile's height and width (ValueError otherwise); its channel count may
    differ from the input's, e.g. (h, w, 3) -> (h, w, 1). The results are blended with a raised-cosine ramp
    across each overlap (the ramps of adjacent tiles sum to exactly 1) and cropped back to (H, W);
    `overlap=0` means plain non-overlapping tiles.

    Returns float32. Raises ValueError unless tile and multiple are positive, tile is a multiple of `multiple`,
    0 <= overlap < tile / 2, and `image` is (H, W) or (H, W, C) with at least one pixel.
    """
    if tile <= 0 or multiple <= 0:
        raise ValueError(f"tile and multiple must be positive, got tile={tile}, multiple={multiple}")
    if tile % multiple:
        raise ValueError(f"tile must be a multiple of multiple, got tile={tile}, multiple={multiple}")
    if overlap < 0 or 2 * overlap >= tile:
        raise ValueError(f"overlap must be at least 0 and less than half of tile, got overlap={overlap}, tile={tile}")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim not in (2, 3) or 0 in image.shape[:2]:
        raise ValueError(f"image must be (H, W) or (H, W, C) with at least one pixel, got shape {image.shape}")

    height, width = image.shape[:2]
    tile_height, starts_y, padded_height = _axis_layout(height, tile, overlap, multiple)
    tile_width, starts_x, padded_width = _axis_layout(width, tile, overlap, multiple)

    padding = [(overlap, padded_height - height - overlap), (overlap, padded_width - width - overlap)]
    padded = np.pad(image, padding + [(0, 0)] * (image.ndim - 2), mode="reflect")

    weight = np.outer(_feather(tile_height, overlap), _feather(tile_width, overlap))
    total_weight = np.zeros((padded_height, padded_width), dtype=np.float32)
    accumulator = None
    for y in starts_y:
        for x in starts_x:
            window = (slice(y, y + tile_height), slice(x, x + tile_width))
            result = np.asarray(fn(padded[window].copy()), dtype=np.float32)
            if result.shape[:2] != (tile_height, tile_width):
                raise ValueError(
                    f"fn must return an array with the tile's height and width {(tile_height, tile_width)}, "
                    f"got shape {result.shape}"
                )
            if accumulator is None:
                accumulator = np.zeros((padded_height, padded_width, *result.shape[2:]), dtype=np.float32)
            accumulator[window] += result * _expand(weight, result.ndim)
            total_weight[window] += weight

    original = (slice(overlap, overlap + height), slice(overlap, overlap + width))
    return accumulator[original] / _expand(total_weight[original], accumulator.ndim)


def _axis_layout(extent: int, tile: int, overlap: int, multiple: int) -> tuple[int, range, int]:
    """Tile size, tile start offsets and total padded length along an axis of `extent` pixels."""
    padded = extent + 2 * overlap
    size = min(tile, -(-padded // multiple) * multiple)
    stride = size - overlap
    count = 1 + max(0, -(-(padded - size) // stride))
    return size, range(0, count * stride, stride), (count - 1) * stride + size


def _feather(size: int, overlap: int) -> np.ndarray:
    """Weights along a tile: a raised-cosine ramp up over the first `overlap` samples, 1 in the middle and the
    mirrored ramp down over the last `overlap`."""
    weights = np.ones(size)
    if overlap:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(overlap) + 0.5) / overlap)
        weights[:overlap] = ramp
        weights[-overlap:] = ramp[::-1]
    return weights.astype(np.float32)


def _expand(weight: np.ndarray, ndim: int) -> np.ndarray:
    """Give a (h, w) weight trailing singleton axes to broadcast against an `ndim`-dimensional array."""
    return weight.reshape(weight.shape + (1,) * (ndim - 2))
