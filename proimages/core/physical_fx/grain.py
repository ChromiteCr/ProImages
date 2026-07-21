import numpy as np
from scipy.ndimage import gaussian_filter

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def apply_grain(
    image: np.ndarray,
    intensity: float = 0.04,
    grain_size: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(0.0, 1.0, size=image.shape).astype(np.float32)
    if grain_size > 0:
        noise = gaussian_filter(noise, sigma=(grain_size, grain_size, 0))
        noise /= noise.std() + 1e-8

    luminance = image @ LUMA_WEIGHTS
    midtone_weight = 4.0 * luminance * (1.0 - luminance)

    grain = noise * intensity * midtone_weight[..., None]
    return (image + grain).clip(0.0, 1.0)
