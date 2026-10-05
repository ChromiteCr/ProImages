import numpy as np

from proimages.core.color import luminance

# median absolute deviation of a standard normal: the constant of Donoho's noise estimator
_NORMAL_MAD = 0.6745


def estimate_noise(image: np.ndarray) -> float:
    """Robust standard deviation of the luminance noise, in the image's own display-encoded [0, 1] units.

    A relative, luminance-only measure: it is taken on the display-encoded values (not on linear light) and
    sees no chroma noise. An (H, W, 3) image is reduced to Rec.709 luma first, so independent per-channel RGB
    noise of std s appears as luma noise of s * sqrt(0.2126**2 + 0.7152**2 + 0.0722**2), about 0.749 * s. An
    (H, W) image is used as is.

    This is Donoho's median-absolute-deviation estimator on the finest Haar diagonal subband,
    HH = (Y[0::2, 0::2] - Y[0::2, 1::2] - Y[1::2, 0::2] + Y[1::2, 1::2]) / 2, estimated as median(|HH|) / 0.6745.
    For i.i.d. Gaussian noise of std sigma, HH has std sigma, so the estimate is sigma itself. Axis-aligned
    edges and planar gradients leave HH at zero and other structure reaches it only sparsely, which the
    median ignores, so only strongly textured images read noticeably high. An odd last row or column is ignored.

    Flat or clipped areas give HH == 0 and pull the median down: once half of the 2x2 blocks are flat the estimate
    is 0. A caller gating on it should leave such areas out.

    Returns a Python float; an image smaller than 2x2 has no estimate and returns 0.0.
    """
    if image.ndim == 3:
        luma = luminance(image)
    elif image.ndim == 2:
        luma = image
    else:
        raise ValueError(f"image must be (H, W) or (H, W, 3), got shape {image.shape}")

    height, width = luma.shape[0] // 2 * 2, luma.shape[1] // 2 * 2
    if height == 0 or width == 0:
        return 0.0
    luma = luma[:height, :width]

    diagonal = (luma[0::2, 0::2] - luma[0::2, 1::2] - luma[1::2, 0::2] + luma[1::2, 1::2]) / 2
    return float(np.median(np.abs(diagonal))) / _NORMAL_MAD
