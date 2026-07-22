import builtins
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from proimages.core.system.io import load_image, save_image


def test_load_image_returns_unit_range_float_rgb(tmp_path: Path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (8, 4), color=(120, 60, 200)).save(path)

    image = load_image(path)

    assert image.shape == (4, 8, 3)
    assert image.dtype == np.float32
    assert (image >= 0.0).all() and (image <= 1.0).all()


def test_save_image_round_trips(tmp_path: Path):
    path = tmp_path / "out.png"
    original = np.full((4, 4, 3), 0.5, dtype=np.float32)

    save_image(original, path)

    assert np.allclose(load_image(path), original, atol=1.0 / 255.0)


def test_raw_without_rawpy_points_at_the_heavy_extra(tmp_path: Path, monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "rawpy":
            raise ImportError("No module named 'rawpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"proimages\[heavy\]"):
        load_image(tmp_path / "photo.dng")
