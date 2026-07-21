from pathlib import Path

import colour
import numpy as np

from proimages.core.pipeline import process_image


def test_process_image_without_lut_stays_in_valid_range():
    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)

    result = process_image(image)

    assert result.shape == image.shape
    assert (result >= 0.0).all() and (result <= 1.0).all()


def test_process_image_applies_lut_when_lut_path_given(tmp_path: Path):
    identity_lut = colour.LUT3D(colour.LUT3D.linear_table(9), name="identity")
    lut_path = tmp_path / "identity.cube"
    colour.write_LUT(identity_lut, str(lut_path))

    image = np.full((8, 8, 3), 0.5, dtype=np.float32)

    result = process_image(image, lut_path=lut_path)

    assert result.shape == image.shape
