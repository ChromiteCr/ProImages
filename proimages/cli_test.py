import sys
from pathlib import Path

import numpy as np
from PIL import Image

from proimages import cli


def _run(monkeypatch, tmp_path: Path, *extra_args: str) -> list[dict]:
    source = tmp_path / "in.png"
    Image.new("RGB", (4, 4), color=(120, 60, 200)).save(source)
    calls = []

    def fake_process_image(image: np.ndarray, **kwargs) -> np.ndarray:
        calls.append(kwargs)
        return image

    monkeypatch.setattr(cli, "process_image", fake_process_image)
    monkeypatch.setattr(sys, "argv", ["proimages", str(source), str(tmp_path / "out.png"), *extra_args])
    cli.main()
    return calls


def test_cli_runs_the_pipeline_on_the_detected_device(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_device", lambda override=None: override or "mps")

    calls = _run(monkeypatch, tmp_path, "--lut", "look.cube")

    assert calls == [{"lut_path": "look.cube", "device": "mps"}]
    assert "using device: mps" in capsys.readouterr().out
    assert (tmp_path / "out.png").exists()


def test_cli_device_flag_overrides_detection(tmp_path: Path, monkeypatch):
    calls = _run(monkeypatch, tmp_path, "--device", "cpu")

    assert calls[0]["device"] == "cpu"
