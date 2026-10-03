import io

import numpy as np
import pytest
from PIL import Image

from proimages.core.system.io import decode_image

EXIF_ORIENTATION = 0x0112

# How EXIF orientation n maps the stored pixels to the upright image (EXIF 2.3, tag 0x0112).
UPRIGHT = {
    1: lambda a: a,
    2: lambda a: a[:, ::-1],
    3: lambda a: a[::-1, ::-1],
    4: lambda a: a[::-1],
    5: lambda a: a.transpose(1, 0, 2),
    6: lambda a: np.rot90(a, k=-1),
    7: lambda a: a[::-1, ::-1].transpose(1, 0, 2),
    8: lambda a: np.rot90(a, k=1),
}


def _stored_pixels() -> np.ndarray:
    """3 rows x 5 columns of distinct colors, so every flip and rotation is distinguishable."""
    values = np.arange(15, dtype=np.uint8).reshape(3, 5) * 17
    return np.stack([values, 255 - values, np.full_like(values, 128)], axis=-1)


def _png(pixels: np.ndarray, orientation: int | None = None) -> bytes:
    buffer = io.BytesIO()
    img = Image.fromarray(pixels, mode="RGB")
    if orientation is None:
        img.save(buffer, format="PNG")
    else:
        exif = Image.Exif()
        exif[EXIF_ORIENTATION] = orientation
        img.save(buffer, format="PNG", exif=exif)
    return buffer.getvalue()


@pytest.mark.parametrize("orientation", sorted(UPRIGHT))
def test_every_exif_orientation_ends_upright(orientation: int):
    pixels = _stored_pixels()

    image = decode_image(_png(pixels, orientation), "photo.png")

    assert np.array_equal(image, UPRIGHT[orientation](pixels).astype(np.float32) / 255.0)


@pytest.mark.parametrize("orientation", [None, 0, 9])
def test_missing_or_invalid_orientation_leaves_the_pixels_alone(orientation):
    pixels = _stored_pixels()

    image = decode_image(_png(pixels, orientation), "photo.png")

    assert np.array_equal(image, pixels.astype(np.float32) / 255.0)


@pytest.mark.parametrize(("mode", "file_format"), [("RGBA", "PNG"), ("LA", "PNG"), ("L", "PNG"), ("1", "PNG"), ("P", "PNG"), ("CMYK", "JPEG")])
def test_common_pillow_modes_decode_to_float_rgb(mode: str, file_format: str):
    buffer = io.BytesIO()
    extra = {"transparency": 0} if mode == "P" else {}
    Image.new(mode, (4, 3)).save(buffer, format=file_format, **extra)

    image = decode_image(buffer.getvalue(), f"x.{file_format.lower()}")

    assert image.shape == (3, 4, 3)
    assert image.dtype == np.float32
    assert (image >= 0.0).all() and (image <= 1.0).all()


def test_animated_gif_uses_its_first_frame():
    frames = [Image.new("RGB", (4, 4), color) for color in [(255, 0, 0), (0, 0, 255)]]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])

    image = decode_image(buffer.getvalue(), "anim.gif")

    assert image.shape == (4, 4, 3)
    assert image[0, 0, 0] > 0.9 and image[0, 0, 2] < 0.1


@pytest.mark.parametrize("filename", ["PHOTO.PNG", "a.b.png", "noext", "dir/sub/photo.png", None])
def test_decode_image_accepts_any_ordinary_filename(filename):
    assert decode_image(_png(_stored_pixels()), filename).shape == (3, 5, 3)
