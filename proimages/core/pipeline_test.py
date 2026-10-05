from pathlib import Path

import colour
import numpy as np
import pytest
from pydantic import ValidationError

from proimages.core import physical_fx
from proimages.core.options import ProcessOptions
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


@pytest.mark.parametrize("options", [None, ProcessOptions(), {}])
def test_default_options_send_the_photo_straight_to_the_physical_effects(options, tmp_path: Path, monkeypatch):
    # With no stage options set, the pipeline is exactly physical_fx: the LUT, then the film effects.
    # Grain draws from np.random.default_rng(), so pin it to compare the two runs bit for bit.
    real_default_rng = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda *args, **kwargs: real_default_rng(7))
    image = real_default_rng(0).random((16, 16, 3)).astype(np.float32)
    brighten = colour.LUT3D(colour.LUT3D.linear_table(9) ** 0.8, name="brighten")
    lut_path = tmp_path / "brighten.cube"
    colour.write_LUT(brighten, str(lut_path))

    expected = physical_fx.process(image, lut_path=lut_path)
    result = process_image(image, lut_path=lut_path, options=options, device="cpu")

    assert np.array_equal(result, expected)


def test_options_given_as_a_dict_are_validated_instead_of_ignored():
    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        process_image(image, options={"denoise": {}})
