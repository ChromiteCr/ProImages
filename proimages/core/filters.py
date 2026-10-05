"""Edge-aware smoothing shared by the denoise, HDR and depth stages: a box filter, the guided
filter, its fast (subsampled) variant and guided upsampling."""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1) x (2r+1) window, per channel, of x shaped (H, W) or (H, W, C).

    Borders reflect by half samples (d c b a | a b c d | d c b a), numpy's "symmetric" padding.
    Returns float32.
    """
    _check_radius(r)
    size = (2 * r + 1, 2 * r + 1) + (1,) * (x.ndim - 2)
    return uniform_filter(x, size=size, mode="reflect", output=np.float32)


def _check_radius(r: int) -> None:
    if r < 1:
        raise ValueError(f"r must be >= 1, got {r}")


def _validate(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Check what all three filters share and return guide and src as float32 arrays."""
    _check_radius(r)
    if not eps > 0:  # also rejects NaN
        raise ValueError(f"eps must be > 0, got {eps}")
    guide = np.asarray(guide, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)
    if not (guide.ndim == 2 or (guide.ndim == 3 and guide.shape[2] == 3)):
        raise ValueError(f"guide must be (H, W) or (H, W, 3), got shape {guide.shape}")
    if src.ndim not in (2, 3):
        raise ValueError(f"src must be (H, W) or (H, W, C), got shape {src.shape}")
    return guide, src


def _check_same_size(guide: np.ndarray, src: np.ndarray) -> None:
    if src.shape[:2] != guide.shape[:2]:
        raise ValueError(
            f"guide and src must have the same height and width, got {guide.shape[:2]} and {src.shape[:2]}"
        )


def _resize(x: np.ndarray, size: tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    """Resample the two leading axes of x to size = (H, W), one float32 plane at a time (Pillow mode "F")."""
    if x.shape[:2] == size:
        return x
    height, width = size
    planes = x.reshape(*x.shape[:2], -1)
    resized = [
        np.asarray(Image.fromarray(np.ascontiguousarray(planes[..., k])).resize((width, height), resample))
        for k in range(planes.shape[-1])
    ]
    return np.stack(resized, axis=-1).reshape(height, width, *x.shape[2:])


def _inverse(m: np.ndarray) -> np.ndarray:
    """Inverse of per-pixel symmetric matrices (h, w, G, G) with G in {1, 3}: cofactors over the determinant."""
    if m.shape[-1] == 1:
        return 1.0 / m
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 1], m[..., 1, 2], m[..., 2, 2]
    c00, c01, c02 = d * f - e * e, c * e - b * f, b * e - c * d
    c11, c12, c22 = a * f - c * c, b * c - a * e, a * d - b * b
    det = a * c00 + b * c01 + c * c02
    cofactors = np.stack([c00, c01, c02, c01, c11, c12, c02, c12, c22], axis=-1).reshape(m.shape)
    return cofactors / det[..., None, None]


def _covariance(x: np.ndarray, y: np.ndarray, mean_x: np.ndarray, mean_y: np.ndarray, r: int) -> np.ndarray:
    """Windowed cov(x_i, y_j) as (h, w, X, Y) for x (h, w, X) and y (h, w, Y), given their window means."""
    h, w, nx = x.shape
    ny = y.shape[2]
    products = (x[..., :, None] * y[..., None, :]).reshape(h, w, nx * ny)
    return box(products, r).reshape(h, w, nx, ny) - mean_x[..., :, None] * mean_y[..., None, :]


def _box_coefficients(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Box-averaged (mean_a, mean_b) of the model q = a . I + b, fitted at the resolution of the inputs.

    guide is (h, w, G) with G = 1 or 3 and src is (h, w, C). a = (Sigma + eps U)^-1 cov(I, p) (paper eq. 14-16),
    which for G = 1 is cov(I, p) / (var(I) + eps). Returns mean_a (h, w, C, G) and mean_b (h, w, C).
    """
    h, w, g = guide.shape
    c = src.shape[2]
    mean_i, mean_p = box(guide, r), box(src, r)
    sigma = _covariance(guide, guide, mean_i, mean_i, r) + np.float32(eps) * np.eye(g, dtype=np.float32)
    # float32 cofactors are useless for a near-singular Sigma (a neutral guide at a small eps): invert in float64
    inverse = _inverse(sigma.astype(np.float64)).astype(np.float32)
    # the inverse is symmetric, so cov(I, p) as a row vector times it is the column-vector solve
    a = _covariance(src, guide, mean_p, mean_i, r) @ inverse
    b = mean_p - (a * mean_i[..., None, :]).sum(axis=-1)
    return box(a.reshape(h, w, c * g), r).reshape(h, w, c, g), box(b, r)


def _apply_coefficients(guide: np.ndarray, mean_a: np.ndarray, mean_b: np.ndarray) -> np.ndarray:
    """q = mean_a . I + mean_b at one resolution: guide (H, W, G), mean_a (H, W, C, G), mean_b (H, W, C)."""
    q = mean_b.copy()
    for c in range(q.shape[-1]):
        for g in range(guide.shape[-1]):
            q[..., c] += mean_a[..., c, g] * guide[..., g]
    return q


def _fit_and_apply(guide: np.ndarray, src: np.ndarray, r: int, eps: float, size: tuple[int, int]) -> np.ndarray:
    """Fit the model at `size` with radius r, upsample its coefficients to guide's resolution and apply them.

    guide is float32 (H, W) or (H, W, 3); src is float32 (h, w) or (h, w, C), either at guide's resolution
    (it is then box-downsampled to `size`) or already at `size`.
    """
    full = guide.shape[:2]
    out_shape = full + src.shape[2:]
    guide = guide.reshape(*full, -1)
    src = src.reshape(*src.shape[:2], -1)
    low_guide, low_src = _resize(guide, size, Image.Resampling.BOX), _resize(src, size, Image.Resampling.BOX)

    # Centre both first: var = box(I*I) - box(I)^2 cancels badly in float32 when the values sit far from 0
    # (log2 luminance is about -8). a is shift invariant and b absorbs the shift, so adding src's mean back
    # at the end is exact; the same guide mean must leave the low- and the full-resolution guide.
    guide_mean = low_guide.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    src_mean = low_src.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    mean_a, mean_b = _box_coefficients(low_guide - guide_mean, low_src - src_mean, r, eps)

    mean_a = _resize(mean_a, full, Image.Resampling.BILINEAR)
    mean_b = _resize(mean_b, full, Image.Resampling.BILINEAR)
    return (_apply_coefficients(guide - guide_mean, mean_a, mean_b) + src_mean).reshape(out_shape)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He, Sun & Tang, TPAMI 2013, Algorithm 1) of src steered by guide, at full resolution.

    guide is (H, W) gray or (H, W, 3) color and src is (H, W) or (H, W, C) with the same H and W. r is the
    window radius in pixels and eps the regularisation, in squared units of the guide. A color guide uses
    the 3x3 covariance form (paper eq. 14-16). Returns src's shape, float32. Time and memory grow with the
    pixel count, a color guide most of all: use fast_guided_filter for full-size photos.
    """
    guide, src = _validate(guide, src, r, eps)
    _check_same_size(guide, src)
    return _fit_and_apply(guide, src, r, eps, guide.shape[:2])


def fast_guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float, s: int) -> np.ndarray:
    """Fast guided filter (He & Sun 2015): guided_filter with the coefficients fitted at 1/s resolution.

    guide and src are box-downsampled by s, mean_a and mean_b are computed there with radius
    max(1, round(r / s)), upsampled bilinearly and applied to the full-resolution guide. r is measured in
    full-resolution pixels; s=1 is guided_filter. Shapes as in guided_filter; returns src's shape, float32.
    """
    if s < 1:
        raise ValueError(f"s must be >= 1, got {s}")
    guide, src = _validate(guide, src, r, eps)
    _check_same_size(guide, src)
    height, width = guide.shape[:2]
    size = (max(1, round(height / s)), max(1, round(width / s)))
    return _fit_and_apply(guide, src, max(1, round(r / s)), eps, size)


def guided_upsample(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Joint upsampling of a low-resolution map (e.g. depth) to the resolution of a guide image.

    src is (h, w) or (h, w, C); guide is the full-resolution (H, W) or (H, W, 3) image, H >= h and W >= w.
    The guide is box-downsampled to (h, w), mean_a and mean_b are computed there with radius r (in
    low-resolution pixels), upsampled bilinearly and applied to the full-resolution guide.
    Returns (H, W) or (H, W, C), float32.
    """
    guide, src = _validate(guide, src, r, eps)
    if src.shape[0] > guide.shape[0] or src.shape[1] > guide.shape[1]:
        raise ValueError(f"src {src.shape[:2]} is larger than the guide {guide.shape[:2]}")
    return _fit_and_apply(guide, src, r, eps, src.shape[:2])
