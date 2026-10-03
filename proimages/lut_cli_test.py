import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from proimages import lut_cli
from proimages.core.lut_gen.params import GradeParams

TEAL_ORANGE = {
    "schema_version": 2,
    "name": "Teal Orange",
    "description": "teal shadows, orange highlights",
    "lift": [-0.02, 0.01, 0.04],
    "gamma": [1.0, 1.0, 1.0],
    "gain": [1.1, 1.0, 0.88],
    "saturation": 1.15,
    "contrast": 1.15,
    "tone": -0.05,
    "temperature": 0.1,
    "tint": 0.0,
}


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr("sys.argv", ["proimages-lut", *argv])
    lut_cli.main()


def _write_params(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "params.json"
    path.write_text(json.dumps(payload))
    return path


def test_params_file_bakes_a_cube_and_a_cc_file(tmp_path: Path, monkeypatch, capsys):
    params_path = _write_params(tmp_path, TEAL_ORANGE)
    cube_path, cc_path = tmp_path / "look.cube", tmp_path / "look.cc"

    _run(monkeypatch, "--params", str(params_path), "-o", str(cube_path), "--cdl-out", str(cc_path), "--size", "5")

    cube = cube_path.read_text()
    assert 'TITLE "Teal Orange"' in cube
    assert "LUT_3D_SIZE 5" in cube
    assert "schema_version 2" in cube
    root = ET.fromstring(cc_path.read_bytes())
    assert root.tag == "ColorCorrection" and root.get("id") == "teal_orange"
    assert root.findtext("SOPNode/Description") == "teal shadows, orange highlights"
    assert len(root.findtext("SOPNode/Slope").split()) == 3
    assert f"wrote {cc_path}" in capsys.readouterr().out


def test_cdl_out_is_optional(tmp_path: Path, monkeypatch):
    cube_path = tmp_path / "look.cube"

    _run(monkeypatch, "--params", str(_write_params(tmp_path, TEAL_ORANGE)), "-o", str(cube_path))

    assert cube_path.exists() and not list(tmp_path.glob("*.cc"))


def test_saved_params_include_the_schema_version_and_load_back(tmp_path: Path, monkeypatch):
    saved = tmp_path / "saved.json"
    _run(
        monkeypatch,
        "--params",
        str(_write_params(tmp_path, TEAL_ORANGE)),
        "-o",
        str(tmp_path / "a.cube"),
        "--save-params",
        str(saved),
    )

    assert json.loads(saved.read_text())["schema_version"] == 2
    _run(monkeypatch, "--params", str(saved), "-o", str(tmp_path / "b.cube"))
    assert (tmp_path / "a.cube").read_text() == (tmp_path / "b.cube").read_text()


def test_params_file_without_schema_version_is_refused(tmp_path: Path, monkeypatch):
    payload = {key: value for key, value in TEAL_ORANGE.items() if key != "schema_version"}

    with pytest.raises(SystemExit) as raised:
        _run(monkeypatch, "--params", str(_write_params(tmp_path, payload)), "-o", str(tmp_path / "x.cube"))

    assert "schema_version" in str(raised.value) and "2" in str(raised.value)
    assert not (tmp_path / "x.cube").exists()


def test_v1_params_file_is_refused_with_a_migration_message(tmp_path: Path, monkeypatch):
    payload = {"schema_version": 1, "name": "Old", "gamma": [0.0, 0.0, 0.0]}

    with pytest.raises(SystemExit) as raised:
        _run(monkeypatch, "--params", str(_write_params(tmp_path, payload)), "-o", str(tmp_path / "x.cube"))

    assert "no longer supported" in str(raised.value)


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"temperature": 4.2}, "temperature"),
        ({"vibrance": 0.5}, "vibrance"),
        ({"lift": [0.5, 0.0, 0.0], "gain": [0.4, 1.0, 1.0]}, "gain must be >= lift"),
    ],
)
def test_invalid_params_give_a_short_message_not_a_traceback(tmp_path: Path, monkeypatch, changes, expected):
    payload = TEAL_ORANGE | changes

    with pytest.raises(SystemExit) as raised:
        _run(monkeypatch, "--params", str(_write_params(tmp_path, payload)), "-o", str(tmp_path / "x.cube"))

    message = str(raised.value)
    assert expected in message
    assert "\n" not in message and "Traceback" not in message
    assert not (tmp_path / "x.cube").exists()


def test_unreadable_params_files_give_a_short_message(tmp_path: Path, monkeypatch):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    not_an_object = tmp_path / "list.json"
    not_an_object.write_text("[1, 2]")

    for path in (broken, tmp_path / "missing.json", not_an_object):
        with pytest.raises(SystemExit) as raised:
            _run(monkeypatch, "--params", str(path), "-o", str(tmp_path / "x.cube"))
        assert str(path) in str(raised.value)


@pytest.mark.parametrize("size", ["1", "0", "-3", "65"])
def test_size_outside_the_lut_range_is_a_usage_error(tmp_path: Path, monkeypatch, capsys, size: str):
    with pytest.raises(SystemExit) as raised:
        _run(
            monkeypatch,
            "--params",
            str(_write_params(tmp_path, TEAL_ORANGE)),
            "-o",
            str(tmp_path / "x.cube"),
            "--size",
            size,
        )

    assert raised.value.code == 2
    assert "--size must be between 2 and 64" in capsys.readouterr().err
    assert not (tmp_path / "x.cube").exists()


def test_size_bounds_are_inclusive(tmp_path: Path, monkeypatch):
    for size in (2, 64):
        _run(
            monkeypatch,
            "--params",
            str(_write_params(tmp_path, GradeParams(name="Edge").model_dump())),
            "-o",
            str(tmp_path / f"{size}.cube"),
            "--size",
            str(size),
        )

        assert f"LUT_3D_SIZE {size}" in (tmp_path / f"{size}.cube").read_text()
