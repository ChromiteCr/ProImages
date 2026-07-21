from pathlib import Path

import colour
import numpy as np

from proimages.core.physical_fx.lut import apply_lut


def test_apply_lut_with_identity_lut_leaves_image_unchanged(tmp_path: Path):
    identity_lut = colour.LUT3D(colour.LUT3D.linear_table(9), name="identity")
    lut_path = tmp_path / "identity.cube"
    colour.write_LUT(identity_lut, str(lut_path))

    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)

    result = apply_lut(image, lut_path)

    assert np.allclose(result, image, atol=1e-5)


def test_apply_lut_output_stays_in_valid_range(tmp_path: Path):
    identity_lut = colour.LUT3D(colour.LUT3D.linear_table(9), name="identity")
    lut_path = tmp_path / "identity.cube"
    colour.write_LUT(identity_lut, str(lut_path))

    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)

    result = apply_lut(image, lut_path)

    assert result.dtype == np.float32
    assert (result >= 0.0).all() and (result <= 1.0).all()
