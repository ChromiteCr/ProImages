import json

import numpy as np
import pytest

from proimages.core.lut_gen.grade import apply_grade
from proimages.core.lut_gen.image_stats import extract_color_stats
from proimages.core.lut_gen.params import GradeParams


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


def _ramp_image(size: int = 1001) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.repeat(ramp[None, :, None], 4, axis=0).repeat(3, axis=2)


def test_tint_balance_is_positive_for_magenta_and_negative_for_green():
    magenta = np.zeros((16, 16, 3), dtype=np.float32)
    magenta[..., 0] = 0.8
    magenta[..., 1] = 0.2
    magenta[..., 2] = 0.8
    green = np.zeros((16, 16, 3), dtype=np.float32)
    green[..., 1] = 0.8

    assert extract_color_stats(magenta)["tint_balance"] == 0.6
    assert extract_color_stats(green)["tint_balance"] == -0.8


def test_tint_balance_has_the_same_sign_as_the_tint_parameter():
    gray = np.full((16, 16, 3), 0.5, dtype=np.float32)

    assert extract_color_stats(apply_grade(gray, GradeParams(tint=0.5)))["tint_balance"] > 0
    assert extract_color_stats(apply_grade(gray, GradeParams(tint=-0.5)))["tint_balance"] < 0
    assert extract_color_stats(apply_grade(gray, GradeParams(temperature=0.5)))["warm_cool_balance"] > 0
    assert extract_color_stats(gray)["tint_balance"] == 0.0


def test_green_magenta_balance_was_replaced_by_tint_balance():
    stats = extract_color_stats(np.full((4, 4, 3), 0.5, dtype=np.float32))

    assert "green_magenta_balance" not in stats
    assert {"tint_balance", "channel_percentiles", "cdl_fit", "warm_cool_balance", "channel_spread"} <= set(stats)


def test_channel_percentiles_report_p1_p50_p99_per_channel():
    image = _ramp_image()
    image[..., 0] *= 0.5

    percentiles = extract_color_stats(image)["channel_percentiles"]

    assert set(percentiles) == {"p1", "p50", "p99"}
    assert percentiles["p1"] == pytest.approx([0.005, 0.01, 0.01], abs=1e-4)
    assert percentiles["p50"] == pytest.approx([0.25, 0.5, 0.5], abs=1e-4)
    assert percentiles["p99"] == pytest.approx([0.495, 0.99, 0.99], abs=1e-4)


def test_cdl_fit_of_a_plain_ramp_is_nearly_neutral():
    fit = extract_color_stats(_ramp_image())["cdl_fit"]

    assert fit["lift"] == pytest.approx([0.01] * 3, abs=1e-3)
    assert fit["gamma"] == pytest.approx([1.0] * 3, abs=1e-3)
    assert fit["gain"] == pytest.approx([0.99] * 3, abs=1e-3)


def test_cdl_fit_recovers_a_known_grade_roughly():
    truth = GradeParams(lift=(0.05, 0.03, 0.02), gamma=(1.2, 1.0, 0.85), gain=(0.95, 1.0, 0.99))

    fit = extract_color_stats(apply_grade(_ramp_image(), truth))["cdl_fit"]

    assert fit["lift"] == pytest.approx(truth.lift, abs=0.015)
    assert fit["gamma"] == pytest.approx(truth.gamma, abs=0.06)
    assert fit["gain"] == pytest.approx(truth.gain, abs=0.015)


def test_cdl_fit_gamma_follows_the_midtone_brightness():
    bright_mids = apply_grade(_ramp_image(), GradeParams(gamma=(1.6, 1.6, 1.6)))
    dark_mids = apply_grade(_ramp_image(), GradeParams(gamma=(0.6, 0.6, 0.6)))

    assert min(extract_color_stats(bright_mids)["cdl_fit"]["gamma"]) > 1.4
    assert max(extract_color_stats(dark_mids)["cdl_fit"]["gamma"]) < 0.7


def test_cdl_fit_is_always_a_valid_set_of_grade_params():
    rng = np.random.default_rng(0)

    for _ in range(100):
        image = rng.random((16, 16, 3)) ** rng.uniform(0.2, 4.0, 3)
        image = np.clip(image * rng.uniform(0.1, 1.0) + rng.uniform(0.0, 0.4), 0.0, 1.0).astype(np.float32)

        fit = extract_color_stats(image)["cdl_fit"]

        params = GradeParams(**fit)
        assert all(gain >= lift for gain, lift in zip(params.gain, params.lift))


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_cdl_fit_survives_flat_images(value: float):
    fit = extract_color_stats(np.full((8, 8, 3), value, dtype=np.float32))["cdl_fit"]

    GradeParams(**fit)
    assert fit["gamma"] == [1.0, 1.0, 1.0]


def test_stats_are_json_serializable():
    stats = extract_color_stats(_ramp_image())

    assert json.loads(json.dumps(stats)) == stats
