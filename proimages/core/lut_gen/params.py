from typing import Annotated

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

UnitRange = Annotated[float, Field(ge=-1.0, le=1.0)]
RGBOffset = tuple[UnitRange, UnitRange, UnitRange]


class GradeParams(BaseModel):
    """Color grading parameters. This is the public contract shared with the UI repo.

    The first three fields map directly onto the iPhone-style grading pad:
    ``saturation`` is the X axis (gray to pure), ``tone`` is the Y axis (dark to
    bright), and ``temperature`` is the slider below it (cool to warm).
    """

    schema_version: int = SCHEMA_VERSION

    tone: UnitRange = Field(default=0.0, description="Overall brightness; pad Y axis")
    saturation: UnitRange = Field(default=0.0, description="Gray to pure; pad X axis")
    temperature: UnitRange = Field(default=0.0, description="Cool to warm; pad slider")

    tint: UnitRange = Field(default=0.0, description="Green to magenta")
    contrast: UnitRange = Field(default=0.0)

    lift: RGBOffset = Field(default=(0.0, 0.0, 0.0), description="Shadow color offset")
    gamma: RGBOffset = Field(default=(0.0, 0.0, 0.0), description="Midtone color offset")
    gain: RGBOffset = Field(default=(0.0, 0.0, 0.0), description="Highlight color offset")

    name: str = Field(default="Untitled", description="Short name for this look")
    description: str = Field(default="", description="What this look is going for")
