import builtins
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from proimages.core.system import io as image_io
from proimages.core.system.io import SRGB_GAMMA, decode_image, encode_png, load_image, save_image

RED = (255, 0, 0)
BLUE = (0, 0, 255)
EXIF_ORIENTATION = 0x0112


def _block_import(monkeypatch, blocked: str) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == blocked:
            raise ImportError(f"No module named '{blocked}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def _rotated_jpeg_bytes() -> bytes:
    """An 8x4 JPEG whose top-left quadrant is red, tagged EXIF orientation 6
    ("rotate 90 degrees clockwise to display")."""
    img = Image.new("RGB", (8, 4), color=BLUE)
    img.paste(RED, (0, 0, 4, 2))
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = 6
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=100, exif=exif)
    return buffer.getvalue()


def _assert_displayed_upright(image: np.ndarray) -> None:
    # Rotated clockwise the image is 4 wide and 8 tall, and the red quadrant sits top right.
    assert image.shape == (8, 4, 3)
    assert image[1, 3, 0] > 0.8 and image[1, 3, 2] < 0.2
    assert image[6, 0, 2] > 0.8 and image[6, 0, 0] < 0.2


class _FakeRaw:
    def __init__(self, source, camera_whitebalance):
        self.source = source
        self.camera_whitebalance = camera_whitebalance
        self.postprocess_kwargs = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def postprocess(self, **kwargs):
        self.postprocess_kwargs = kwargs
        return np.full((2, 3, 3), 32768, dtype=np.uint16)


class _FakeLibRawError(Exception):
    pass


class _FakeLibRawIOError(_FakeLibRawError):
    pass


def _install_fake_rawpy(monkeypatch, camera_whitebalance, fail: bool = False) -> list[_FakeRaw]:
    opened: list[_FakeRaw] = []

    def imread(source):
        if fail:
            raise _FakeLibRawIOError(b"Input/output error")
        raw = _FakeRaw(source, camera_whitebalance)
        opened.append(raw)
        return raw

    fake = types.SimpleNamespace(imread=imread, LibRawError=_FakeLibRawError)
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    return opened


def _dummy_raw(tmp_path: Path, name: str = "photo.dng") -> Path:
    path = tmp_path / name
    path.write_bytes(b"placeholder; the fake rawpy never reads it")
    return path


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


def test_encode_png_round_trips():
    original = np.full((4, 4, 3), 0.25, dtype=np.float32)

    decoded = decode_image(encode_png(original), "out.png")

    assert np.allclose(decoded, original, atol=1.0 / 255.0)


def test_load_image_applies_exif_orientation(tmp_path: Path):
    path = tmp_path / "rotated.jpg"
    path.write_bytes(_rotated_jpeg_bytes())

    _assert_displayed_upright(load_image(path))


def test_decode_image_applies_exif_orientation_with_or_without_a_filename():
    data = _rotated_jpeg_bytes()

    _assert_displayed_upright(decode_image(data, "rotated.jpg"))
    _assert_displayed_upright(decode_image(data))


def test_decode_image_matches_load_image(tmp_path: Path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (6, 3), color=(10, 200, 90)).save(path)

    assert np.array_equal(decode_image(path.read_bytes(), path.name), load_image(path))


def test_raw_without_rawpy_points_at_the_heavy_extra(tmp_path: Path, monkeypatch):
    _block_import(monkeypatch, "rawpy")

    with pytest.raises(ImportError, match=r"proimages\[heavy\]"):
        load_image(tmp_path / "photo.dng")
    with pytest.raises(ImportError, match=r"proimages\[heavy\]"):
        decode_image(b"", "photo.dng")


def test_raw_uses_camera_white_balance_and_the_srgb_curve(tmp_path: Path, monkeypatch):
    opened = _install_fake_rawpy(monkeypatch, camera_whitebalance=[2.1, 1.0, 1.6, 0.0])

    image = load_image(_dummy_raw(tmp_path))

    kwargs = opened[0].postprocess_kwargs
    assert kwargs["use_camera_wb"] is True
    assert kwargs["gamma"] == SRGB_GAMMA == (2.4, 12.92)
    assert kwargs["output_bps"] == 16
    assert image.shape == (2, 3, 3) and image.dtype == np.float32
    assert np.allclose(image, 32768 / 65535)


def test_raw_without_camera_white_balance_falls_back_to_daylight(tmp_path: Path, monkeypatch):
    opened = _install_fake_rawpy(monkeypatch, camera_whitebalance=[0.0, 0.0, 0.0, 0.0])

    load_image(_dummy_raw(tmp_path, "photo.nef"))

    assert opened[0].postprocess_kwargs["use_camera_wb"] is False


def test_decode_image_reads_raw_bytes_as_a_file_object(monkeypatch):
    opened = _install_fake_rawpy(monkeypatch, camera_whitebalance=[2.1, 1.0, 1.6, 0.0])

    decode_image(b"raw bytes", "PHOTO.DNG")

    assert isinstance(opened[0].source, io.BytesIO)
    assert opened[0].source.getvalue() == b"raw bytes"


def test_heif_without_pi_heif_points_at_the_heif_extra(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(image_io, "_heif_registered", False)
    _block_import(monkeypatch, "pi_heif")

    with pytest.raises(ImportError, match=r"proimages\[heif\]"):
        load_image(tmp_path / "photo.heic")
    with pytest.raises(ImportError, match=r"proimages\[heif\]"):
        decode_image(b"", "photo.HEIF")


def test_heif_decoder_is_registered_once_and_only_when_needed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(image_io, "_heif_registered", False)
    calls = []
    monkeypatch.setitem(sys.modules, "pi_heif", types.SimpleNamespace(register_heif_opener=lambda: calls.append(1)))

    load_image(_png(tmp_path))
    assert calls == []

    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "missing.heic")
    assert calls == [1]


def test_decode_image_without_pi_heif_still_reads_ordinary_images(monkeypatch):
    monkeypatch.setattr(image_io, "_heif_registered", False)
    _block_import(monkeypatch, "pi_heif")

    image = decode_image(encode_png(np.zeros((2, 2, 3), dtype=np.float32)))

    assert image.shape == (2, 2, 3)


def _assert_clean_unreadable(exc_info, kind: str) -> None:
    message = str(exc_info.value)
    assert message == f"not a readable {kind} file"
    assert "0x" not in message and "b'" not in message


@pytest.mark.parametrize(("filename", "kind"), [("x.jpg", "jpg"), ("x.PNG", "png"), (None, "image"), ("noext", "image")])
def test_undecodable_bytes_raise_a_clean_value_error(filename, kind):
    with pytest.raises(ValueError) as exc_info:
        decode_image(b"these bytes are not an image", filename)

    _assert_clean_unreadable(exc_info, kind)


def test_truncated_jpeg_raises_a_clean_value_error():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(9, 9, 9)).save(buffer, format="JPEG")

    with pytest.raises(ValueError) as exc_info:
        decode_image(buffer.getvalue()[:300], "cut.jpg")

    _assert_clean_unreadable(exc_info, "jpg")


def test_undecodable_heic_raises_a_clean_value_error(monkeypatch):
    pytest.importorskip("pi_heif")
    monkeypatch.setattr(image_io, "_heif_registered", False)

    with pytest.raises(ValueError) as exc_info:
        decode_image(b"not a heic", "photo.heic")

    _assert_clean_unreadable(exc_info, "heic")


def test_undecodable_raw_raises_a_clean_value_error(monkeypatch):
    _install_fake_rawpy(monkeypatch, camera_whitebalance=[1.0, 1.0, 1.0, 0.0], fail=True)

    with pytest.raises(ValueError) as exc_info:
        decode_image(b"not a raw file", "photo.DNG")

    _assert_clean_unreadable(exc_info, "dng")


def test_missing_raw_path_is_a_file_not_found_error(tmp_path: Path, monkeypatch):
    _install_fake_rawpy(monkeypatch, camera_whitebalance=[1.0, 1.0, 1.0, 0.0])

    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.dng")


def test_missing_ordinary_path_is_still_a_file_not_found_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")


def test_sixteen_bit_grayscale_keeps_its_brightness(tmp_path: Path):
    path = tmp_path / "gray16.png"
    Image.fromarray(np.array([[0, 30000, 65535]], dtype=np.uint16)).save(path)

    image = load_image(path)

    assert image.shape == (1, 3, 3)
    assert np.allclose(image[0, :, 0], [0.0, 30000 / 65535, 1.0], atol=1e-6)
    assert np.array_equal(image[..., 0], image[..., 2])


def _png(tmp_path: Path) -> Path:
    path = tmp_path / "plain.png"
    Image.new("RGB", (2, 2)).save(path)
    return path
