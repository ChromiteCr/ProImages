import re
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SCHEMA_VERSION = 2

LIFT_MIN, LIFT_MAX = -0.5, 0.5
GAMMA_MIN, GAMMA_MAX = 0.5, 2.0
GAIN_MIN, GAIN_MAX = 0.0, 2.0

NAME_MAX_LENGTH = 80
DESCRIPTION_MAX_LENGTH = 400
DEFAULT_NAME = "Untitled"

# Double quotes, plus what XML 1.0 cannot carry: control characters other than
# whitespace, surrogates and U+FFFE/U+FFFF.
_DROPPED_CHARS = re.compile('["\x00-\x08\x0e-\x1b\x7f-\x84\x86-\x9f\ud800-\udfff\ufffe\uffff]')

UnitRange = Annotated[float, Field(ge=-1.0, le=1.0)]
LiftRGB = tuple[
    Annotated[float, Field(ge=LIFT_MIN, le=LIFT_MAX)],
    Annotated[float, Field(ge=LIFT_MIN, le=LIFT_MAX)],
    Annotated[float, Field(ge=LIFT_MIN, le=LIFT_MAX)],
]
GammaRGB = tuple[
    Annotated[float, Field(ge=GAMMA_MIN, le=GAMMA_MAX)],
    Annotated[float, Field(ge=GAMMA_MIN, le=GAMMA_MAX)],
    Annotated[float, Field(ge=GAMMA_MIN, le=GAMMA_MAX)],
]
GainRGB = tuple[
    Annotated[float, Field(ge=GAIN_MIN, le=GAIN_MAX)],
    Annotated[float, Field(ge=GAIN_MIN, le=GAIN_MAX)],
    Annotated[float, Field(ge=GAIN_MIN, le=GAIN_MAX)],
]


def clean_text(value: str, max_length: int) -> str:
    """Make free text safe for a .cube TITLE or comment line and for XML: a single
    line, no double quotes, at most ``max_length`` characters."""
    text = " ".join(_DROPPED_CHARS.sub("", value).split())
    return text[:max_length].rstrip()


class GradeParams(BaseModel):
    """Color grading parameters, schema v2.

    The controls are industry-standard lift/gamma/gain with their standard neutral
    values (lift 0, gamma 1, gain 1); the whole grade folds exactly into one ASC CDL
    v1.2 correction (see ``cdl.py``). The grading pad maps onto three fields: X axis
    is ``saturation - 1``, Y axis is ``tone`` and the slider is ``temperature``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(
        default=SCHEMA_VERSION,
        description="Must be 2. Version 1 (neutral lift/gamma/gain of 0) is no longer supported.",
    )

    tone: UnitRange = Field(
        default=0.0,
        description=(
            "Overall brightness, -1 to 1; 0 is neutral. Scales the midtone gamma by 2**tone, "
            "so black and white stay fixed. Pad Y axis."
        ),
    )
    saturation: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description=(
            "ASC CDL saturation on Rec.709 luma, 0 to 2; 1 is neutral, 0 is fully gray. "
            "Pad X axis is saturation - 1."
        ),
    )
    temperature: UnitRange = Field(
        default=0.0,
        description="White balance, -1 (cool) to 1 (warm); 0 is neutral. Keeps mid-gray luma. Pad slider.",
    )

    tint: UnitRange = Field(
        default=0.0,
        description="White balance, -1 (green) to 1 (magenta); 0 is neutral. Keeps mid-gray luma.",
    )
    contrast: float = Field(
        default=1.0,
        ge=0.25,
        le=2.0,
        description="Contrast around a fixed pivot of 0.5, 0.25 to 2; 1 is neutral, below 1 flattens.",
    )

    lift: LiftRGB = Field(
        default=(0.0, 0.0, 0.0),
        description=(
            "Per-channel [r, g, b] output black level, each -0.5 to 0.5; 0 is neutral. "
            "Equal values shift shadow brightness, unequal values tint the shadows."
        ),
    )
    gamma: GammaRGB = Field(
        default=(1.0, 1.0, 1.0),
        description=(
            "Per-channel [r, g, b] midtone control, each 0.5 to 2; 1 is neutral. Above 1 brightens "
            "the midtones (ASC CDL power is 1/gamma, the opposite direction). Equal values shift "
            "midtone brightness, unequal values tint the midtones."
        ),
    )
    gain: GainRGB = Field(
        default=(1.0, 1.0, 1.0),
        description=(
            "Per-channel [r, g, b] output white level, each 0 to 2; 1 is neutral, and each channel "
            "must be >= the same channel of lift. Equal values shift highlight brightness, unequal "
            "values tint the highlights."
        ),
    )

    name: str = Field(
        default=DEFAULT_NAME,
        description=(
            "Short name for this look. Whitespace is collapsed, double quotes are removed and "
            f"it is truncated to {NAME_MAX_LENGTH} characters."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "What this look is going for. Cleaned like name and truncated to "
            f"{DESCRIPTION_MAX_LENGTH} characters."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_other_schema_versions(cls, data: Any) -> Any:
        version = data.get("schema_version", SCHEMA_VERSION) if isinstance(data, dict) else SCHEMA_VERSION
        if version == 1:
            raise ValueError(
                "GradeParams schema v1 is no longer supported; "
                "lift/gamma/gain now use standard neutral values 0/1/1 — see README"
            )
        if version != SCHEMA_VERSION:
            raise ValueError(f"GradeParams schema_version must be {SCHEMA_VERSION}, got {version!r}")
        return data

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        return clean_text(value, NAME_MAX_LENGTH) or DEFAULT_NAME

    @field_validator("description")
    @classmethod
    def _clean_description(cls, value: str) -> str:
        return clean_text(value, DESCRIPTION_MAX_LENGTH)

    @model_validator(mode="after")
    def _gain_not_below_lift(self) -> Self:
        if any(gain < lift for gain, lift in zip(self.gain, self.lift)):
            raise ValueError("gain must be >= lift in every channel")
        return self


def format_validation_error(exc: ValidationError, max_length: int = 300) -> str:
    """One short line for a ValidationError, safe to show to a user or send back to a model."""
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts)[:max_length]
