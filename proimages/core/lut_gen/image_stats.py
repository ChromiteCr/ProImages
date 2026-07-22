import numpy as np

from proimages.core.color import luminance

SHADOW_PERCENTILE = 25.0
HIGHLIGHT_PERCENTILE = 75.0


def _mean_rgb(pixels: np.ndarray) -> list[float]:
    if pixels.size == 0:
        return [0.0, 0.0, 0.0]
    return [round(float(v), 4) for v in pixels.mean(axis=0)]


def extract_color_stats(image: np.ndarray) -> dict:
    """Summarize a reference photo's grade objectively, so the model translates
    measurements into style parameters instead of eyeballing color from pixels."""
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

    return {
        "shadow_mean_rgb": _mean_rgb(shadows),
        "midtone_mean_rgb": _mean_rgb(midtones),
        "highlight_mean_rgb": _mean_rgb(highlights),
        "overall_mean_rgb": [round(float(v), 4) for v in channel_means],
        "mean_luminance": round(float(luma.mean()), 4),
        "luminance_std": round(float(luma.std()), 4),
        "mean_saturation": round(float(saturation), 4),
        "warm_cool_balance": round(float(channel_means[0] - channel_means[2]), 4),
        "green_magenta_balance": round(float(channel_means[1] - (channel_means[0] + channel_means[2]) / 2.0), 4),
        "channel_spread": round(channel_max - channel_min, 4),
    }
