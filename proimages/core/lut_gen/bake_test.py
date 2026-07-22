from pathlib import Path

import numpy as np

from proimages.core.lut_gen.bake import bake_cube_file, bake_cube_text
from proimages.core.lut_gen.params import GradeParams
from proimages.core.physical_fx.lut import apply_lut


def test_baked_lut_is_readable_by_the_projects_own_apply_lut(tmp_path: Path):
    lut_path = tmp_path / "identity.cube"
    bake_cube_file(GradeParams(name="Identity"), lut_path)

    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    result = apply_lut(image, lut_path)

    assert np.allclose(result, image, atol=1e-4)


def test_baked_warm_lut_warms_an_image(tmp_path: Path):
    lut_path = tmp_path / "warm.cube"
    bake_cube_file(GradeParams(name="Warm", temperature=0.8), lut_path)

    result = apply_lut(np.full((4, 4, 3), 0.5, dtype=np.float32), lut_path)

    assert result[0, 0, 0] > 0.5
    assert result[0, 0, 2] < 0.5


def test_bake_cube_text_has_cube_header_and_metadata():
    text = bake_cube_text(GradeParams(name="Peek", description="a test look"), size=2)

    assert 'TITLE "Peek"' in text
    assert "LUT_3D_SIZE 2" in text
    assert "schema_version 1" in text
    assert "a test look" in text


def test_bake_size_controls_entry_count():
    text = bake_cube_text(GradeParams(), size=4)

    entries = [line for line in text.splitlines() if line and not line.startswith(("#", "TITLE", "LUT_3D_SIZE"))]
    assert len(entries) == 4**3
