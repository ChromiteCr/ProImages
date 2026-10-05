from pathlib import Path

import pytest
from pydantic import DirectoryPath, FilePath, ValidationError, create_model

from proimages.core.options import ProcessOptions

# A property name is path-like when one of its underscore-separated tokens is a path word ("lut_path" is,
# "direction" is not). The formats are what pydantic puts in the JSON schema of Path, FilePath and DirectoryPath.
PATH_WORDS = {"path", "paths", "file", "files", "filename", "dir", "directory", "folder"}
PATH_FORMATS = {"path", "file-path", "directory-path"}


def test_default_options_turn_no_optional_stage_on():
    assert ProcessOptions().model_dump() == {}
    assert ProcessOptions.model_validate_json("{}") == ProcessOptions()


@pytest.mark.parametrize("payload", ['{"denoise": {}}', '{"hdr": true}', '{"typo": 1}'])
def test_options_for_stages_that_do_not_exist_are_rejected(payload: str):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProcessOptions.model_validate_json(payload)


def _properties(schema: dict) -> list[tuple[str, dict]]:
    """(name, schema) of every property, including those of the models under $defs."""
    found = list(schema.get("properties", {}).items())
    for definition in schema.get("$defs", {}).values():
        found += _properties(definition)
    return found


def _formats(node) -> set[str]:
    """Every "format" anywhere inside a schema node: an optional or list field nests it in anyOf or items."""
    if isinstance(node, dict):
        found = {node["format"]} if isinstance(node.get("format"), str) else set()
        return found.union(*(_formats(value) for value in node.values()))
    if isinstance(node, list):
        return set().union(*(_formats(item) for item in node))
    return set()


def _path_like_properties(schema: dict) -> list[str]:
    return [
        name
        for name, prop in _properties(schema)
        if PATH_WORDS & set(name.lower().split("_")) or PATH_FORMATS & _formats(prop)
    ]


def test_options_hold_no_file_paths():
    # The API takes these options as JSON from any client, so a path-like field would let a
    # request point the server at arbitrary files. File inputs travel as uploads instead.
    assert not _path_like_properties(ProcessOptions.model_json_schema())


@pytest.mark.parametrize(
    "name, annotation, path_like",
    [
        ("lut_path", str, True),
        ("output_dir", str, True),
        ("Files", list[str], True),
        ("weights", Path | None, True),
        ("sources", list[Path], True),
        ("model", FilePath, True),
        ("cache", DirectoryPath, True),
        ("direction", str, False),
        ("profile", str, False),
        ("blend", float, False),
    ],
)
def test_the_path_check_flags_path_like_fields_and_spares_look_alikes(name, annotation, path_like: bool):
    # ProcessOptions has no fields yet, so the check above says nothing until this one shows it can fail.
    probe = create_model("Probe", **{name: (annotation, None)})

    assert bool(_path_like_properties(probe.model_json_schema())) is path_like


def test_the_path_check_looks_inside_nested_models():
    outer = create_model("Outer", nested=(create_model("Inner", cache_dir=(str, None)), None))

    assert _path_like_properties(outer.model_json_schema()) == ["cache_dir"]
