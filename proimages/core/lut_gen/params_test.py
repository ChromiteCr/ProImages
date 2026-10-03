from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from proimages.core.lut_gen.bake import bake_cube_file, bake_cube_text
from proimages.core.lut_gen.params import (
    DESCRIPTION_MAX_LENGTH,
    NAME_MAX_LENGTH,
    SCHEMA_VERSION,
    GradeParams,
    clean_text,
    format_validation_error,
)
from proimages.core.physical_fx.lut import apply_lut


def test_defaults_are_the_standard_neutral_values():
    params = GradeParams()

    assert params.schema_version == SCHEMA_VERSION == 2
    assert params.lift == (0.0, 0.0, 0.0)
    assert params.gamma == (1.0, 1.0, 1.0)
    assert params.gain == (1.0, 1.0, 1.0)
    assert (params.saturation, params.contrast) == (1.0, 1.0)
    assert (params.tone, params.temperature, params.tint) == (0.0, 0.0, 0.0)


def test_every_field_is_documented_for_openapi():
    properties = GradeParams.model_json_schema()["properties"]

    assert set(properties) == set(GradeParams.model_fields)
    assert all(prop.get("description") for prop in properties.values())


def test_extra_keys_are_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GradeParams.model_validate({"vibrance": 0.5})


def test_schema_v1_is_rejected_with_a_migration_message():
    with pytest.raises(ValidationError, match="schema v1 is no longer supported") as raised:
        GradeParams.model_validate({"schema_version": 1, "gamma": [0.0, 0.0, 0.0]})

    assert "0/1/1" in str(raised.value)


@pytest.mark.parametrize("version", [0, 3, 99, "2", None])
def test_any_other_schema_version_is_rejected(version):
    with pytest.raises(ValidationError, match=f"schema_version must be 2, got {version!r}"):
        GradeParams.model_validate({"schema_version": version})


def test_schema_v2_is_accepted_explicitly():
    assert GradeParams.model_validate({"schema_version": 2}).schema_version == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("lift", (0.0, 0.0, 0.51)),
        ("lift", (-0.51, 0.0, 0.0)),
        ("gamma", (0.49, 1.0, 1.0)),
        ("gamma", (1.0, 2.01, 1.0)),
        ("gain", (1.0, -0.01, 1.0)),
        ("gain", (1.0, 1.0, 2.01)),
        ("saturation", -0.01),
        ("saturation", 2.01),
        ("contrast", 0.24),
        ("contrast", 2.01),
        ("tone", 1.01),
        ("temperature", -1.01),
        ("tint", 1.01),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value):
    with pytest.raises(ValidationError):
        GradeParams.model_validate({field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("lift", (-0.5, 0.5, 0.0)),
        ("gamma", (0.5, 2.0, 1.0)),
        ("gain", (0.5, 2.0, 0.5)),
        ("saturation", 0.0),
        ("saturation", 2.0),
        ("contrast", 0.25),
        ("contrast", 2.0),
        ("tone", -1.0),
        ("temperature", 1.0),
    ],
)
def test_range_boundaries_are_accepted(field: str, value):
    GradeParams.model_validate({field: value})


def test_gain_below_lift_is_rejected():
    with pytest.raises(ValidationError, match="gain must be >= lift"):
        GradeParams(lift=(0.3, 0.0, 0.0), gain=(0.2, 1.0, 1.0))


def test_gain_equal_to_lift_is_allowed():
    GradeParams(lift=(0.3, 0.0, 0.0), gain=(0.3, 1.0, 1.0))


def test_name_and_description_lose_newlines_quotes_and_extra_spaces():
    params = GradeParams(name='  My "Great"\nLook\r\n\t2  ', description='line one\nline "two"\n\n  end ')

    assert params.name == "My Great Look 2"
    assert params.description == "line one line two end"


def test_quote_removal_does_not_leave_double_spaces():
    assert GradeParams(name='a " b').name == "a b"


def test_characters_xml_cannot_carry_are_dropped():
    params = GradeParams(name="Ab\x00c\x1bd", description="x\ud800y\x7fz\ufffe\uffffw")

    assert params.name == "Abcd"
    assert params.description == "xyzw"


def test_overlong_name_and_description_are_truncated_not_rejected():
    params = GradeParams(name="n" * 500, description="word " * 400)

    assert len(params.name) == NAME_MAX_LENGTH
    assert 0 < len(params.description) <= DESCRIPTION_MAX_LENGTH
    assert not params.description.endswith(" ")


def test_a_name_that_cleans_to_nothing_falls_back_to_the_default():
    assert GradeParams(name='"" \n ').name == "Untitled"


def test_clean_text_is_idempotent():
    once = clean_text('  a "b"\n c ' * 50, 80)

    assert clean_text(once, 80) == once


def test_a_hostile_name_and_description_still_bake_a_well_formed_cube(tmp_path: Path):
    params = GradeParams(
        name='Evil "Title"\nLUT_3D_SIZE 99\n1 2 3',
        description='second line\n# not a comment "quoted"\nTITLE "again"',
        temperature=0.3,
        contrast=1.2,
    )
    size = 5

    text = bake_cube_text(params, size=size)
    lines = text.splitlines()

    assert lines[0].startswith("TITLE ") and lines[0].count('"') == 2
    assert sum(line.startswith("LUT_3D_SIZE") for line in lines) == 1
    assert f"LUT_3D_SIZE {size}" in lines
    header_end = lines.index(f"LUT_3D_SIZE {size}")
    assert all(line.startswith("#") for line in lines[1:header_end])
    rows = lines[header_end + 1 :]
    assert len(rows) == size**3
    assert all(len(row.split()) == 3 for row in rows)

    lut_path = tmp_path / "hostile.cube"
    bake_cube_file(params, lut_path, size=size)
    image = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
    assert apply_lut(image, lut_path).shape == image.shape


def test_format_validation_error_is_short_and_omits_input_values():
    with pytest.raises(ValidationError) as raised:
        GradeParams.model_validate({"temperature": 12345.678, "bogus": "sk-secret-value", "lift": [0, 0, 9]})

    text = format_validation_error(raised.value)

    assert "temperature" in text and "bogus" in text and "lift.2" in text
    assert "12345.678" not in text and "sk-secret-value" not in text
    assert "\n" not in text and len(text) <= 300


def test_format_validation_error_shows_model_level_messages_without_the_prefix():
    with pytest.raises(ValidationError) as raised:
        GradeParams(lift=(0.3, 0.0, 0.0), gain=(0.2, 1.0, 1.0))

    assert format_validation_error(raised.value) == "gain must be >= lift in every channel"
