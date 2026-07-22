import numpy as np

from proimages.core.lut_gen.image_stats import extract_color_stats


def test_warm_image_reports_positive_warm_cool_balance():
    warm = np.zeros((16, 16, 3), dtype=np.float32)
    warm[..., 0] = 0.8
    warm[..., 1] = 0.5
    warm[..., 2] = 0.2

    stats = extract_color_stats(warm)

    assert stats["warm_cool_balance"] > 0


def test_cool_image_reports_negative_warm_cool_balance():
    cool = np.zeros((16, 16, 3), dtype=np.float32)
    cool[..., 0] = 0.2
    cool[..., 2] = 0.8

    stats = extract_color_stats(cool)

    assert stats["warm_cool_balance"] < 0


def test_gray_image_reports_near_zero_saturation():
    gray = np.full((16, 16, 3), 0.5, dtype=np.float32)

    stats = extract_color_stats(gray)

    assert stats["mean_saturation"] == 0.0
    assert stats["mean_luminance"] == 0.5


def test_gradient_separates_shadows_from_highlights():
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    image = np.repeat(ramp[None, :, None], 8, axis=0).repeat(3, axis=2)

    stats = extract_color_stats(image)

    assert stats["shadow_mean_rgb"][0] < stats["midtone_mean_rgb"][0]
    assert stats["midtone_mean_rgb"][0] < stats["highlight_mean_rgb"][0]
