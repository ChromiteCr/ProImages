import subprocess
import sys
from pathlib import Path

import colour
import numpy as np
import pytest

from proimages.core.color.transfer import linear_to_srgb, srgb_to_linear

DECODE_THRESHOLD = 0.04045
ENCODE_THRESHOLD = 0.0031308
UNIT_RAMP = np.linspace(0.0, 1.0, 100001)


def test_encoded_values_survive_a_round_trip_through_linear_light():
    assert np.abs(linear_to_srgb(srgb_to_linear(UNIT_RAMP)) - UNIT_RAMP).max() < 1e-6


def test_linear_values_survive_a_round_trip_through_the_encoded_values():
    assert np.abs(srgb_to_linear(linear_to_srgb(UNIT_RAMP)) - UNIT_RAMP).max() < 1e-6


def test_srgb_to_linear_matches_colour_science():
    expected = colour.cctf_decoding(UNIT_RAMP, function="sRGB")

    assert np.abs(srgb_to_linear(UNIT_RAMP) - expected).max() < 1e-6


def test_linear_to_srgb_matches_colour_science():
    expected = colour.cctf_encoding(UNIT_RAMP, function="sRGB")

    assert np.abs(linear_to_srgb(UNIT_RAMP) - expected).max() < 1e-6


def test_known_values():
    assert srgb_to_linear(np.float32(0.5)) == pytest.approx(0.21404114, abs=1e-7)
    assert linear_to_srgb(np.float32(0.5)) == pytest.approx(0.73535698, abs=1e-7)
    assert np.allclose(srgb_to_linear(np.array([0.0, 1.0])), [0.0, 1.0], atol=1e-6)
    assert np.allclose(linear_to_srgb(np.array([0.0, 1.0])), [0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize("transfer", [srgb_to_linear, linear_to_srgb])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_output_is_float32_for_float32_and_float64_input(transfer, dtype):
    values = np.linspace(0.0, 1.0, 24, dtype=dtype).reshape(2, 4, 3)

    result = transfer(values)

    assert result.dtype == np.float32
    assert result.shape == values.shape


@pytest.mark.parametrize("transfer", [srgb_to_linear, linear_to_srgb])
@pytest.mark.parametrize("shape", [(), (7,), (4, 5, 3), (2, 3, 4, 3)])
def test_applies_element_wise_to_any_shape(transfer, shape):
    values = np.asarray(np.random.default_rng(0).random(shape), dtype=np.float32)

    result = transfer(values)

    assert result.shape == shape
    assert np.array_equal(result.ravel(), transfer(values.ravel()))


@pytest.mark.parametrize("transfer", [srgb_to_linear, linear_to_srgb])
def test_input_is_left_unchanged(transfer):
    values = np.linspace(-0.5, 1.5, 101, dtype=np.float32)
    snapshot = values.copy()

    transfer(values)

    assert np.array_equal(values, snapshot)


@pytest.mark.parametrize(
    "transfer, threshold, linear_slope",
    [(srgb_to_linear, DECODE_THRESHOLD, 1 / 12.92), (linear_to_srgb, ENCODE_THRESHOLD, 12.92)],
)
def test_the_linear_and_power_segments_meet_at_the_thresholds(transfer, threshold, linear_slope):
    edge = np.float32(threshold)
    neighbours = [np.nextafter(edge, np.float32(-1.0)), edge, np.nextafter(edge, np.float32(2.0))]

    below, at, above = transfer(np.array(neighbours, dtype=np.float32))

    # the two segments meet at the threshold: the float below, the threshold itself and the float above all agree,
    # so there is no jump
    assert at == pytest.approx(threshold * linear_slope, abs=1e-7)
    assert below == pytest.approx(at, abs=1e-7)
    assert above == pytest.approx(at, abs=1e-7)


def test_inputs_below_zero_and_above_one_follow_their_segments_without_clipping_or_warnings():
    below = np.array([-1e6, -100.0, -2.0, -1.0, -0.5, -DECODE_THRESHOLD, -1e-6], dtype=np.float32)
    above = np.array([1.0001, 1.5, 2.0, 10.0, 100.0, 1e4], dtype=np.float32)

    # a power of a negative base would be NaN with a RuntimeWarning, so any such evaluation raises here
    with np.errstate(all="raise"):
        decoded = srgb_to_linear(below), srgb_to_linear(above)
        encoded = linear_to_srgb(below), linear_to_srgb(above)

    # float32 against a float64 reference: numpy's float32 power loses about 1e-6 relative at the largest input here
    # (1e4 decodes to 3.5e9), so 1e-5 is the tolerance; a wrong constant or segment would be off by 1e-3 or more
    below, above = below.astype(np.float64), above.astype(np.float64)
    assert all(np.isfinite(result).all() for result in decoded + encoded)
    assert np.allclose(decoded[0], below / 12.92, rtol=1e-5, atol=0.0)
    assert np.allclose(decoded[1], ((above + 0.055) / 1.055) ** 2.4, rtol=1e-5, atol=0.0)
    assert np.allclose(encoded[0], below * 12.92, rtol=1e-5, atol=0.0)
    assert np.allclose(encoded[1], 1.055 * above ** (1 / 2.4) - 0.055, rtol=1e-5, atol=0.0)


def test_huge_finite_inputs_give_finite_results_without_warnings_while_the_result_fits_float32():
    # the branch that is not selected must not overflow either: 12.92 * 3.4e38 does, but the encode of 3.4e38 is 1.2e16
    decode_inputs = np.array([-3.4e38, -1e30, 1e15], dtype=np.float32)
    encode_inputs = np.array([-2.6e37, 1e30, 3.4e38], dtype=np.float32)

    with np.errstate(all="raise"):
        decoded = srgb_to_linear(decode_inputs)
        encoded = linear_to_srgb(encode_inputs)

    assert np.isfinite(decoded).all() and np.isfinite(encoded).all()
    assert np.allclose(decoded[:2], decode_inputs[:2] / 12.92, rtol=1e-5)
    assert encoded[0] == pytest.approx(-2.6e37 * 12.92, rel=1e-5)
    assert encoded[2] == pytest.approx(1.055 * 3.4e38 ** (1 / 2.4) - 0.055, rel=1e-5)


def test_the_module_does_not_import_colour_science():
    """colour returns float64 and is slower, so it is the reference in the tests only."""
    repo_root = Path(__file__).resolve().parents[3]
    probe = "import sys, proimages.core.color.transfer; assert 'colour' not in sys.modules"

    result = subprocess.run([sys.executable, "-c", probe], cwd=repo_root, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
