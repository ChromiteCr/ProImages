import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from proimages.core.lut_gen.bake import bake_cube_file
from proimages.core.lut_gen.cdl import (
    CDL,
    MAX_EFFECTIVE_GAMMA,
    PIVOT,
    apply_cdl,
    cc_id_from_name,
    format_cdl,
    to_cc_xml,
    to_cdl,
    white_balance_gains,
)
from proimages.core.lut_gen.grade import apply_grade
from proimages.core.lut_gen.params import GradeParams
from proimages.core.physical_fx.lut import apply_lut

REC709 = np.array([0.2126, 0.7152, 0.0722])
NEUTRAL = GradeParams()


def random_cdl(rng: np.random.Generator) -> CDL:
    return CDL(
        slope=tuple(rng.uniform(0.0, 3.0, 3).tolist()),
        offset=tuple(rng.uniform(-0.5, 0.5, 3).tolist()),
        power=tuple(rng.uniform(0.25, 4.0, 3).tolist()),
        saturation=float(rng.uniform(0.0, 2.5)),
    )


def random_params(rng: np.random.Generator) -> GradeParams:
    lift = rng.uniform(-0.5, 0.5, 3)
    gain = rng.uniform(np.maximum(lift, 0.0), 2.0)
    return GradeParams(
        lift=tuple(lift.tolist()),
        gamma=tuple(rng.uniform(0.5, 2.0, 3).tolist()),
        gain=tuple(gain.tolist()),
        saturation=float(rng.uniform(0.0, 2.0)),
        contrast=float(rng.uniform(0.25, 2.0)),
        tone=float(rng.uniform(-1.0, 1.0)),
        temperature=float(rng.uniform(-1.0, 1.0)),
        tint=float(rng.uniform(-1.0, 1.0)),
    )


def typical_params(rng: np.random.Generator) -> GradeParams:
    """A look inside the typical ranges the model prompt advertises, with an effective gamma <= 1.5."""
    while True:
        params = GradeParams(
            lift=tuple(rng.uniform(-0.08, 0.08, 3).tolist()),
            gamma=tuple(rng.uniform(0.8, 1.25, 3).tolist()),
            gain=tuple(rng.uniform(0.85, 1.2, 3).tolist()),
            saturation=float(rng.uniform(0.7, 1.3)),
            contrast=float(rng.uniform(0.85, 1.25)),
            tone=float(rng.uniform(-0.3, 0.3)),
            temperature=float(rng.uniform(-0.4, 0.4)),
            tint=float(rng.uniform(-0.2, 0.2)),
        )
        if (np.array(params.gamma) * 2.0**params.tone <= 1.5).all():
            return params


def reference_cdl(rgb: np.ndarray, cdl: CDL) -> np.ndarray:
    """ASC CDL v1.2 forward transform, written out independently of apply_cdl, in float64."""
    slope, offset, power = (np.array(values, dtype=np.float64) for values in (cdl.slope, cdl.offset, cdl.power))
    sop = np.clip(rgb * slope + offset, 0.0, 1.0) ** power
    luma = (sop @ REC709)[..., None]
    return np.clip(luma + cdl.saturation * (sop - luma), 0.0, 1.0)


def stepwise_grade(rgb: np.ndarray, params: GradeParams) -> np.ndarray:
    """The documented order, one step at a time, in float64: white balance, lift/gain (Nuke
    Grade form), contrast about the pivot, clamp, power, saturation."""
    raw = 1.0 + 0.25 * params.temperature * np.array([1.0, 0.0, -1.0]) + 0.15 * params.tint * np.array([0.5, -1.0, 0.5])
    graded = rgb * (raw / (raw @ REC709))

    lift, gain = np.array(params.lift), np.array(params.gain)
    graded = (gain - lift) * graded + lift

    graded = (graded - 0.5) * params.contrast + 0.5
    graded = np.clip(graded, 0.0, 1.0)

    effective_gamma = np.minimum(np.array(params.gamma) * 2.0**params.tone, 2.0)
    graded = graded ** (1.0 / effective_gamma)

    luma = (graded @ REC709)[..., None]
    return np.clip(luma + params.saturation * (graded - luma), 0.0, 1.0)


def parse_cc(xml_text: str) -> CDL:
    root = ET.fromstring(xml_text.encode("utf-8"))

    def numbers(path: str) -> tuple[float, ...]:
        return tuple(float(value) for value in root.findtext(path).split())

    return CDL(
        slope=numbers("SOPNode/Slope"),
        offset=numbers("SOPNode/Offset"),
        power=numbers("SOPNode/Power"),
        saturation=float(root.findtext("SatNode/Saturation")),
    )


def test_apply_cdl_matches_the_asc_formula_for_random_cdls():
    rng = np.random.default_rng(0)

    for _ in range(200):
        cdl = random_cdl(rng)
        rgb = rng.uniform(-0.25, 1.25, (64, 3)).astype(np.float32)

        result = apply_cdl(rgb, cdl)

        assert result.dtype == np.float32
        assert np.allclose(result, reference_cdl(rgb.astype(np.float64), cdl), atol=1e-5)


def test_apply_cdl_clamps_before_the_power_and_after_the_saturation():
    cdl = CDL(slope=(2.0, 2.0, 2.0), offset=(0.0, 0.0, 0.0), power=(2.0, 2.0, 2.0), saturation=1.0)

    result = apply_cdl(np.array([[0.75, 0.25, -0.5]], dtype=np.float32), cdl)

    # 1.5 clamps to 1 (not 2.25), 0.5 squares to 0.25, -1.0 clamps to 0
    assert np.allclose(result, [[1.0, 0.25, 0.0]], atol=1e-6)


def test_apply_cdl_saturation_uses_rec709_luma_of_the_sop_output():
    cdl = CDL(slope=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0), power=(1.0, 1.0, 1.0), saturation=0.0)

    result = apply_cdl(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32), cdl)

    assert np.allclose(result[:, 0], [0.2126, 0.7152, 0.0722], atol=1e-6)
    assert np.allclose(result[:, 0:1], result, atol=1e-6)


def test_to_cdl_matches_the_documented_step_by_step_order():
    rng = np.random.default_rng(1)

    for _ in range(300):
        params = random_params(rng)
        rgb = rng.uniform(-0.25, 1.25, (40, 3))
        expected = stepwise_grade(rgb, params)

        assert np.allclose(reference_cdl(rgb, to_cdl(params)), expected, atol=1e-7)
        assert np.allclose(apply_grade(rgb.astype(np.float32), params), expected, atol=5e-6)


def test_to_cdl_folds_the_affine_steps_into_slope_and_offset():
    params = GradeParams(
        lift=(0.1, -0.1, 0.0),
        gain=(0.9, 1.2, 1.0),
        contrast=1.5,
        temperature=0.4,
        tint=-0.2,
        saturation=1.3,
    )
    wb = white_balance_gains(0.4, -0.2)

    cdl = to_cdl(params)

    assert cdl.slope == pytest.approx(tuple(1.5 * (np.array(params.gain) - np.array(params.lift)) * wb))
    assert cdl.offset == pytest.approx(tuple(1.5 * np.array(params.lift) + 0.5 * (1.0 - 1.5)))
    assert cdl.power == (1.0, 1.0, 1.0)
    assert cdl.saturation == 1.3


def test_the_contrast_pivot_and_gamma_cap_are_the_documented_constants():
    assert PIVOT == 0.5
    assert MAX_EFFECTIVE_GAMMA == 2.0


def test_neutral_params_fold_to_the_identity_cdl():
    assert to_cdl(NEUTRAL) == CDL(slope=(1, 1, 1), offset=(0, 0, 0), power=(1, 1, 1), saturation=1)


def test_lift_and_gain_map_black_and_white_to_themselves():
    rng = np.random.default_rng(2)

    for _ in range(50):
        lift = rng.uniform(0.0, 0.4, 3)
        gain = rng.uniform(0.5, 1.0, 3)
        params = GradeParams(lift=tuple(lift.tolist()), gain=tuple(gain.tolist()))

        result = apply_grade(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32), params)

        assert np.allclose(result[0], lift, atol=1e-6)
        assert np.allclose(result[1], gain, atol=1e-6)


def test_lift_below_zero_and_gain_above_one_clip_at_the_ends():
    params = GradeParams(lift=(-0.2, -0.2, -0.2), gain=(1.5, 1.5, 1.5))

    result = apply_grade(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32), params)

    assert np.allclose(result, [[0.0] * 3, [1.0] * 3], atol=1e-6)


def test_lift_gamma_gain_follow_the_nuke_grade_formula():
    rng = np.random.default_rng(3)

    for _ in range(50):
        lift = rng.uniform(0.0, 0.3, 3)
        gain = rng.uniform(lift + 0.3, 1.0)
        gamma = rng.uniform(0.5, 2.0, 3)
        params = GradeParams(lift=tuple(lift.tolist()), gamma=tuple(gamma.tolist()), gain=tuple(gain.tolist()))
        rgb = rng.random((32, 3))

        result = apply_grade(rgb.astype(np.float32), params)

        assert np.allclose(result, ((gain - lift) * rgb + lift) ** (1.0 / gamma), atol=1e-5)


@pytest.mark.parametrize("params", [GradeParams(gamma=(1.5, 1.5, 1.5)), GradeParams(tone=0.5)])
def test_gamma_above_one_and_positive_tone_brighten_midtones_but_not_the_ends(params: GradeParams):
    ramp = np.array([[0.0] * 3, [0.5] * 3, [1.0] * 3], dtype=np.float32)

    result = apply_grade(ramp, params)

    assert np.allclose(result[0], 0.0, atol=1e-6)
    assert np.allclose(result[2], 1.0, atol=1e-6)
    assert (result[1] > 0.5 + 0.05).all()


@pytest.mark.parametrize("params", [GradeParams(gamma=(0.7, 0.7, 0.7)), GradeParams(tone=-0.5)])
def test_gamma_below_one_and_negative_tone_darken_midtones(params: GradeParams):
    result = apply_grade(np.full((1, 3), 0.5, dtype=np.float32), params)

    assert (result < 0.5 - 0.05).all()


def test_gamma_is_the_opposite_direction_of_cdl_power():
    cdl = to_cdl(GradeParams(gamma=(1.25, 0.8, 1.0)))

    assert cdl.power == pytest.approx((0.8, 1.25, 1.0))


def test_tone_scales_gamma_by_a_power_of_two():
    cdl = to_cdl(GradeParams(gamma=(1.0, 1.25, 0.8), tone=0.5))

    assert cdl.power == pytest.approx(tuple(1.0 / (np.array([1.0, 1.25, 0.8]) * 2.0**0.5)))


@pytest.mark.parametrize(
    "temperature, tint",
    [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.37, -0.81)],
)
def test_temperature_and_tint_keep_mid_gray_luminance(temperature: float, tint: float):
    gray = np.full((1, 3), 0.5, dtype=np.float32)

    result = apply_grade(gray, GradeParams(temperature=temperature, tint=tint))

    assert (result @ REC709)[0] == pytest.approx(0.5, abs=1e-4)


def test_white_balance_directions_and_neutral():
    assert np.allclose(white_balance_gains(0.0, 0.0), 1.0)

    warm = white_balance_gains(1.0, 0.0)
    magenta = white_balance_gains(0.0, 1.0)

    assert warm[0] > 1.0 > warm[2]
    assert magenta[1] < 1.0 < magenta[0]
    assert np.isclose(warm @ REC709, 1.0) and np.isclose(magenta @ REC709, 1.0)


@pytest.mark.parametrize(
    "params, expected_power",
    [
        (GradeParams(gamma=(2.0, 2.0, 2.0), tone=1.0), (0.5, 0.5, 0.5)),
        (GradeParams(gamma=(1.5, 0.5, 1.0), tone=1.0), (0.5, 1.0, 0.5)),
        (GradeParams(gamma=(1.2, 1.0, 0.9), tone=0.5), None),
    ],
)
def test_effective_gamma_is_capped_at_two(params: GradeParams, expected_power):
    power = np.array(to_cdl(params).power)

    assert (power >= 0.5 - 1e-12).all()
    if expected_power is not None:
        assert power == pytest.approx(expected_power)


def test_power_never_drops_below_half_for_any_valid_params():
    rng = np.random.default_rng(4)

    assert all(min(to_cdl(random_params(rng)).power) >= 0.5 - 1e-12 for _ in range(500))


def test_cc_xml_round_trips_the_numbers_through_an_xml_parser():
    rng = np.random.default_rng(5)

    for _ in range(50):
        cdl = to_cdl(random_params(rng))

        parsed = parse_cc(to_cc_xml(cdl, "look"))

        assert parsed.slope == pytest.approx(cdl.slope, abs=1e-6)
        assert parsed.offset == pytest.approx(cdl.offset, abs=1e-6)
        assert parsed.power == pytest.approx(cdl.power, abs=1e-6)
        assert parsed.saturation == pytest.approx(cdl.saturation, abs=1e-6)


def test_cc_xml_structure_and_declaration():
    xml_text = to_cc_xml(to_cdl(GradeParams(temperature=0.4)), "golden_hour", "warm")

    assert xml_text.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<ColorCorrection id="golden_hour">')
    root = ET.fromstring(xml_text.encode("utf-8"))
    assert [child.tag for child in root] == ["SOPNode", "SatNode"]
    assert [child.tag for child in root.find("SOPNode")] == ["Description", "Slope", "Offset", "Power"]
    assert root.findtext("SOPNode/Description") == "warm"
    for path in ("SOPNode/Slope", "SOPNode/Offset", "SOPNode/Power"):
        assert all(len(token.split(".")[1]) == 6 for token in root.findtext(path).split())


def test_cc_xml_omits_the_description_when_empty_and_escapes_markup():
    without = ET.fromstring(to_cc_xml(to_cdl(NEUTRAL), "plain").encode("utf-8"))
    tricky = ET.fromstring(to_cc_xml(to_cdl(NEUTRAL), 'id<&"', "a <b> & c").encode("utf-8"))

    assert without.find("SOPNode/Description") is None
    assert tricky.get("id") == 'id<&"'
    assert tricky.findtext("SOPNode/Description") == "a <b> & c"


def test_cc_xml_never_prints_negative_zero():
    cdl = CDL(slope=(1.0, 1.0, 1.0), offset=(-1e-9, -0.0, 0.0), power=(1.0, 1.0, 1.0), saturation=1.0)

    assert "-0.000000" not in to_cc_xml(cdl, "x")
    assert "-0.000000" not in format_cdl(cdl)


def test_cc_id_from_name():
    assert cc_id_from_name("Golden Hour!") == "golden_hour"
    assert cc_id_from_name("  Teal & Orange -- v2 ") == "teal_orange_v2"
    assert cc_id_from_name("暖调 胶片") == "暖调_胶片"
    assert cc_id_from_name("★ ★") == "look"


def test_format_cdl_lists_every_number_with_six_decimals():
    cdl = to_cdl(GradeParams(temperature=0.4, contrast=1.2, saturation=0.8))

    assert format_cdl(cdl).startswith("ASC CDL slope ")
    tokens = format_cdl(cdl).split()
    assert [tokens[i] for i in (2, 6, 10, 14)] == ["slope", "offset", "power", "sat"]
    assert len(tokens) == 16 and all(len(token.split(".")[1]) == 6 for token in tokens[3:6] + tokens[7:10])


@pytest.fixture
def ocio():
    return pytest.importorskip("PyOpenColorIO")


def ocio_apply(ocio, cc_path: Path, rgb: np.ndarray) -> np.ndarray:
    transform = ocio.CDLTransform.CreateFromFile(str(cc_path), "")
    transform.setStyle(ocio.CDL_ASC)
    cpu = ocio.Config.CreateRaw().getProcessor(transform).getDefaultCPUProcessor()
    return np.array([cpu.applyRGB(pixel.tolist()) for pixel in rgb], dtype=np.float32)


def test_ocio_reads_our_cc_file_back_to_the_same_numbers(ocio, tmp_path: Path):
    cdl = to_cdl(GradeParams(temperature=0.4, contrast=1.2, saturation=0.8, gamma=(1.2, 1.0, 0.9)))
    cc_path = tmp_path / "read_back.cc"
    cc_path.write_text(to_cc_xml(cdl, "read_back", "a look"), encoding="utf-8")

    transform = ocio.CDLTransform.CreateFromFile(str(cc_path), "")

    assert transform.getID() == "read_back"
    assert transform.getFirstSOPDescription() == "a look"
    assert transform.getSlope() == pytest.approx(cdl.slope, abs=1e-6)
    assert transform.getOffset() == pytest.approx(cdl.offset, abs=1e-6)
    assert transform.getPower() == pytest.approx(cdl.power, abs=1e-6)
    assert transform.getSat() == pytest.approx(cdl.saturation, abs=1e-6)


def test_ocio_applies_the_asc_cdl_the_same_way_as_apply_cdl(ocio, tmp_path: Path):
    """OCIO's CDL_ASC style (clamp, power, Rec.709-luma saturation, clamp) is the reference. Its
    power is a fast approximation (about 1e-5 relative), hence the looser tolerance below."""
    rng = np.random.default_rng(6)
    cdls = [to_cdl(random_params(rng)) for _ in range(40)] + [random_cdl(rng) for _ in range(20)]

    for index, cdl in enumerate(cdls):
        # OCIO caches parsed files by path, so every CDL gets its own file
        cc_path = tmp_path / f"cdl_{index}.cc"
        cc_text = to_cc_xml(cdl, f"cdl_{index}")
        cc_path.write_text(cc_text, encoding="utf-8")
        rgb = rng.uniform(-0.25, 1.25, (100, 3)).astype(np.float32)

        ours = apply_cdl(rgb, parse_cc(cc_text))

        assert np.allclose(ocio_apply(ocio, cc_path, rgb), ours, atol=2e-4)


def test_ocio_agrees_tightly_when_the_power_is_one(ocio, tmp_path: Path):
    rng = np.random.default_rng(7)

    for index in range(30):
        cdl = CDL(
            slope=tuple(rng.uniform(0.0, 3.0, 3).tolist()),
            offset=tuple(rng.uniform(-0.5, 0.5, 3).tolist()),
            power=(1.0, 1.0, 1.0),
            saturation=float(rng.uniform(0.0, 2.5)),
        )
        cc_path = tmp_path / f"linear_{index}.cc"
        cc_text = to_cc_xml(cdl, f"linear_{index}")
        cc_path.write_text(cc_text, encoding="utf-8")
        rgb = rng.uniform(-0.25, 1.25, (100, 3)).astype(np.float32)

        assert np.allclose(ocio_apply(ocio, cc_path, rgb), apply_cdl(rgb, parse_cc(cc_text)), atol=1e-6)


def lut_error(params: GradeParams, tmp_path: Path) -> float:
    """Largest difference between a baked 33^3 LUT (written to disk, read back through the
    project's apply_lut) and the exact grade, over random pixels plus extra ones in the
    darkest LUT cell, where a power below 1 is sampled worst."""
    rng = np.random.default_rng(8)
    pixels = np.concatenate([rng.random((3000, 1, 3)), rng.random((1500, 1, 3)) / 32.0]).astype(np.float32)
    lut_path = tmp_path / "budget.cube"
    bake_cube_file(params, lut_path, size=33)

    return float(np.abs(apply_lut(pixels, lut_path) - apply_grade(pixels, params)).max())


def test_lut_error_budget_for_gamma_and_tone_with_effective_gamma_up_to_one_and_a_half(tmp_path: Path):
    rng = np.random.default_rng(9)
    checked = 0

    while checked < 20:
        params = GradeParams(gamma=tuple(rng.uniform(0.5, 2.0, 3).tolist()), tone=float(rng.uniform(-1.0, 1.0)))
        if (np.array(params.gamma) * 2.0**params.tone > 1.5).any():
            continue
        checked += 1

        assert lut_error(params, tmp_path) <= 4 / 255


def test_lut_error_budget_for_typical_looks(tmp_path: Path):
    """The 33^3 trilinear LUT cannot follow the clamp corners and the steep power near black of a
    strongly graded look exactly; typical looks stay within a few percent at worst (the mean
    error is far smaller) and a LUT with swapped axes or channels would miss by several times more."""
    rng = np.random.default_rng(10)

    for _ in range(20):
        assert lut_error(typical_params(rng), tmp_path) <= 12 / 255


def test_the_gamma_cap_bounds_the_lut_error_even_at_the_range_extremes(tmp_path: Path):
    params = GradeParams(gamma=(2.0, 2.0, 2.0), tone=1.0)

    assert to_cdl(params).power == pytest.approx((0.5, 0.5, 0.5))
    assert lut_error(params, tmp_path) <= 11.5 / 255
