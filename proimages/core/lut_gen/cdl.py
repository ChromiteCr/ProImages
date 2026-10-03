import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from proimages.core.color import luminance
from proimages.core.lut_gen.params import GradeParams

PIVOT = 0.5
MAX_EFFECTIVE_GAMMA = 2.0

TEMPERATURE_GAIN = np.array([1.0, 0.0, -1.0])
TINT_GAIN = np.array([0.5, -1.0, 0.5])
TEMPERATURE_SCALE = 0.25
TINT_SCALE = 0.15

DEFAULT_CC_ID = "look"
XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>'

RGB = tuple[float, float, float]


class CDL(BaseModel):
    """An ASC CDL v1.2 correction: per-channel slope/offset/power, then saturation."""

    model_config = ConfigDict(frozen=True)

    slope: RGB = Field(description="Per-channel multiplier, >= 0.")
    offset: RGB = Field(description="Per-channel offset added after the slope.")
    power: RGB = Field(description="Per-channel exponent, > 0. The reciprocal of gamma.")
    saturation: float = Field(description="Saturation on Rec.709 luma; 1 is neutral.")


def white_balance_gains(temperature: float, tint: float) -> np.ndarray:
    """Per-channel gains for the temperature and tint sliders, divided by their Rec.709
    luma so a mid-gray keeps its luminance."""
    raw = 1.0 + temperature * TEMPERATURE_SCALE * TEMPERATURE_GAIN + tint * TINT_SCALE * TINT_GAIN
    return raw / luminance(raw)


def to_cdl(params: GradeParams) -> CDL:
    """Fold a grade into one CDL.

    The grade is white balance, lift/gain in Nuke Grade form, contrast about PIVOT,
    a clamp to [0, 1], a power, then saturation. Everything before the clamp is
    affine per channel, so it collapses into the slope and offset. The effective
    gamma (gamma * 2**tone) is capped at MAX_EFFECTIVE_GAMMA: a power below 0.5
    cannot be sampled accurately into a LUT near black.
    """
    lift = np.asarray(params.lift, dtype=np.float64)
    gain = np.asarray(params.gain, dtype=np.float64)
    gamma = np.asarray(params.gamma, dtype=np.float64)
    contrast = params.contrast

    slope = contrast * (gain - lift) * white_balance_gains(params.temperature, params.tint)
    offset = contrast * lift + PIVOT * (1.0 - contrast)
    power = 1.0 / np.minimum(gamma * 2.0**params.tone, MAX_EFFECTIVE_GAMMA)

    return CDL(
        slope=tuple(slope.tolist()),
        offset=tuple(offset.tolist()),
        power=tuple(power.tolist()),
        saturation=params.saturation,
    )


def apply_cdl(rgb: np.ndarray, cdl: CDL) -> np.ndarray:
    """Apply the ASC CDL v1.2 forward transform to any array shaped (..., 3)."""
    x = np.asarray(rgb, dtype=np.float32)
    slope = np.asarray(cdl.slope, dtype=np.float32)
    offset = np.asarray(cdl.offset, dtype=np.float32)
    power = np.asarray(cdl.power, dtype=np.float32)

    sop = np.power(np.clip(x * slope + offset, 0.0, 1.0), power)
    luma = luminance(sop)[..., None]
    graded = luma + cdl.saturation * (sop - luma)
    return np.clip(graded, 0.0, 1.0).astype(np.float32)


def _format_values(values: Iterable[float]) -> str:
    # round first, and add 0.0 so a tiny negative never prints as "-0.000000"
    return " ".join(f"{round(value, 6) + 0.0:.6f}" for value in values)


def format_cdl(cdl: CDL) -> str:
    return (
        f"ASC CDL slope {_format_values(cdl.slope)} offset {_format_values(cdl.offset)} "
        f"power {_format_values(cdl.power)} sat {_format_values([cdl.saturation])}"
    )


def cc_id_from_name(name: str) -> str:
    """Turn a look name into a .cc id: lowercase, each run of non-alphanumerics an underscore."""
    return "_".join(re.findall(r"[^\W_]+", name.lower())) or DEFAULT_CC_ID


def to_cc_xml(cdl: CDL, cc_id: str, description: str = "") -> str:
    """Write a CDL as an ASC ColorCorrection (.cc) XML document."""
    root = ET.Element("ColorCorrection", id=cc_id)

    sop = ET.SubElement(root, "SOPNode")
    if description:
        ET.SubElement(sop, "Description").text = description
    ET.SubElement(sop, "Slope").text = _format_values(cdl.slope)
    ET.SubElement(sop, "Offset").text = _format_values(cdl.offset)
    ET.SubElement(sop, "Power").text = _format_values(cdl.power)

    sat = ET.SubElement(root, "SatNode")
    ET.SubElement(sat, "Saturation").text = _format_values([cdl.saturation])

    ET.indent(root)
    return f"{XML_DECLARATION}\n{ET.tostring(root, encoding='unicode')}\n"
