import numpy as np

from proimages.core.color import luminance
from proimages.core.lut_gen.params import GAIN_MAX, GAIN_MIN, GAMMA_MAX, GAMMA_MIN, LIFT_MAX, LIFT_MIN

SHADOW_PERCENTILE = 25.0
HIGHLIGHT_PERCENTILE = 75.0

CHANNEL_PERCENTILES = {"p1": 1.0, "p50": 50.0, "p99": 99.0}
FIT_MIN_RANGE = 1e-3
FIT_BISECTION_STEPS = 40


def _mean_rgb(pixels: np.ndarray) -> list[float]:
    if pixels.size == 0:
        return [0.0, 0.0, 0.0]
    return [round(float(v), 4) for v in pixels.mean(axis=0)]


def _fit_gamma(p1: float, p50: float, p99: float) -> float:
    """Solve f(0.5) = p50 for f(x) = ((g - l) * x + l) ^ (1 / gamma), with l and g placed so
    that f(0) = p1 and f(1) = p99. f(0.5) is the gamma-power mean of p1 and p99, which grows
    with gamma, so a bisection finds the root; gamma is clamped to its bounds otherwise."""
    if p99 - p1 < FIT_MIN_RANGE:
        return 1.0

    def midpoint(gamma: float) -> float:
        return ((p1**gamma + p99**gamma) / 2.0) ** (1.0 / gamma)

    low, high = GAMMA_MIN, GAMMA_MAX
    if midpoint(low) >= p50:
        return low
    if midpoint(high) <= p50:
        return high

    for _ in range(FIT_BISECTION_STEPS):
        mid = (low + high) / 2.0
        if midpoint(mid) < p50:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _fit_lift_gamma_gain(percentiles: np.ndarray) -> dict[str, list[float]]:
    """Fit lift/gamma/gain per channel from a (3 percentiles, 3 channels) array. Rough:
    it assumes the scene was neutral and spanned black to white."""
    fit: dict[str, list[float]] = {"lift": [], "gamma": [], "gain": []}
    for p1, p50, p99 in np.clip(percentiles.T.astype(np.float64), 0.0, 1.0):
        gamma = _fit_gamma(p1, p50, p99)
        lift = float(np.clip(p1**gamma, LIFT_MIN, LIFT_MAX))
        gain = float(np.clip(max(p99**gamma, lift), GAIN_MIN, GAIN_MAX))
        fit["lift"].append(round(lift, 4))
        fit["gamma"].append(round(gamma, 4))
        fit["gain"].append(round(gain, 4))
    return fit


def extract_color_stats(image: np.ndarray) -> dict:
    """Summarize a reference photo's grade objectively, so the model translates
    measurements into style parameters instead of eyeballing color from pixels.

    ``image`` and every RGB value reported are display-encoded sRGB in [0, 1].
    """
    flat = image.reshape(-1, 3).astype(np.float32)
    luma = luminance(flat)

    shadow_cut = float(np.percentile(luma, SHADOW_PERCENTILE))
    highlight_cut = float(np.percentile(luma, HIGHLIGHT_PERCENTILE))

    shadows = flat[luma <= shadow_cut]
    midtones = flat[(luma > shadow_cut) & (luma < highlight_cut)]
    highlights = flat[luma >= highlight_cut]

    channel_means = flat.mean(axis=0)
    channel_max = float(channel_means.max())
    channel_min = float(channel_means.min())
    saturation = (flat.max(axis=1) - flat.min(axis=1)).mean()

    percentiles = np.percentile(flat, list(CHANNEL_PERCENTILES.values()), axis=0)

    return {
        "shadow_mean_rgb": _mean_rgb(shadows),
        "midtone_mean_rgb": _mean_rgb(midtones),
        "highlight_mean_rgb": _mean_rgb(highlights),
        "overall_mean_rgb": [round(float(v), 4) for v in channel_means],
        "mean_luminance": round(float(luma.mean()), 4),
        "luminance_std": round(float(luma.std()), 4),
        "mean_saturation": round(float(saturation), 4),
        "warm_cool_balance": round(float(channel_means[0] - channel_means[2]), 4),
        "tint_balance": round(float((channel_means[0] + channel_means[2]) / 2.0 - channel_means[1]), 4),
        "channel_spread": round(channel_max - channel_min, 4),
        "channel_percentiles": {
            key: [round(float(v), 4) for v in row] for key, row in zip(CHANNEL_PERCENTILES, percentiles)
        },
        "cdl_fit": _fit_lift_gamma_gain(percentiles),
    }
