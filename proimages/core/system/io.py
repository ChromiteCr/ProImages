import errno
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

RAW_EXTENSIONS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2"}
HEIF_EXTENSIONS = {".heic", ".heif"}

# LibRaw's (power, toe slope) pair for the sRGB curve; its default is BT.709 (2.222, 4.5).
SRGB_GAMMA = (2.4, 12.92)

# Single-channel modes deeper than 8 bits, with the value that means white. Pillow's
# convert("RGB") clips these at 255, which turns a 16-bit mid-gray into pure white.
HIGH_BIT_GRAY_WHITE = {"I;16": 65535.0, "I;16B": 65535.0, "I;16L": 65535.0, "I;16N": 65535.0, "I": 65535.0, "F": 1.0}

_heif_registered = False


def _import_rawpy(suffix: str):
    try:
        import rawpy
    except ImportError as exc:
        raise ImportError(
            f"Reading {suffix} needs rawpy, which ships in the 'heavy' extra: "
            "pip install 'proimages[heavy]'"
        ) from exc
    return rawpy


def _register_heif(suffix: str | None) -> None:
    """Register the HEIF decoder with Pillow once. When ``suffix`` names a HEIF file a
    missing decoder is an error; otherwise registration is opportunistic."""
    global _heif_registered
    if _heif_registered:
        return
    try:
        from pi_heif import register_heif_opener
    except ImportError as exc:
        if suffix is None:
            return
        raise ImportError(
            f"Reading {suffix} needs pi-heif, which ships in the 'heif' extra: "
            "pip install 'proimages[heif]'"
        ) from exc
    register_heif_opener()
    _heif_registered = True


def _develop_raw(raw) -> np.ndarray:
    """Develop with the white balance recorded at capture (daylight if the file has none)
    and the sRGB curve the rest of the pipeline assumes. LibRaw applies the orientation."""
    use_camera_wb = raw.camera_whitebalance[0] > 0
    rgb = raw.postprocess(use_camera_wb=use_camera_wb, gamma=SRGB_GAMMA, output_bps=16)
    return (rgb.astype(np.float32) / 65535.0).clip(0.0, 1.0)


def _from_pillow(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img)
    white = HIGH_BIT_GRAY_WHITE.get(img.mode)
    if white is not None:
        gray = (np.asarray(img).astype(np.float32) / white).clip(0.0, 1.0)
        return np.repeat(gray[..., None], 3, axis=2)
    return (np.asarray(img.convert("RGB")).astype(np.float32) / 255.0).clip(0.0, 1.0)


def _unreadable(suffix: str) -> ValueError:
    # A clean message: the underlying errors carry object addresses or LibRaw byte strings.
    return ValueError(f"not a readable {suffix.lstrip('.') or 'image'} file")


def _read_raw(rawpy, source, suffix: str) -> np.ndarray:
    try:
        with rawpy.imread(source) as raw:
            return _develop_raw(raw)
    except rawpy.LibRawError as exc:
        raise _unreadable(suffix) from exc


def _read_pillow(source, suffix: str) -> np.ndarray:
    try:
        with Image.open(source) as img:
            return _from_pillow(img)
    except UnidentifiedImageError as exc:
        raise _unreadable(suffix) from exc
    except OSError as exc:
        # A plain OSError is Pillow's "truncated file"; subclasses such as
        # FileNotFoundError and PermissionError are about the path and stay as they are.
        if type(exc) is not OSError:
            raise
        raise _unreadable(suffix) from exc


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return (image.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        rawpy = _import_rawpy(path.suffix)
        if not path.is_file():
            # LibRaw reports a missing path as an I/O error; say what actually happened.
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
        return _read_raw(rawpy, str(path), suffix)

    if suffix in HEIF_EXTENSIONS:
        _register_heif(path.suffix)
    return _read_pillow(path, suffix)


def decode_image(data: bytes, filename: str | None = None) -> np.ndarray:
    """Decode image bytes, e.g. an upload. Only the suffix of ``filename`` is used, to pick
    the RAW or HEIF path; without one, HEIF still decodes if pi-heif is installed.
    Undecodable data raises ValueError("not a readable <kind> file")."""
    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix in RAW_EXTENSIONS:
        return _read_raw(_import_rawpy(suffix), io.BytesIO(data), suffix)

    _register_heif(suffix if suffix in HEIF_EXTENSIONS else None)
    return _read_pillow(io.BytesIO(data), suffix)


def save_image(image: np.ndarray, path: str | Path) -> None:
    Image.fromarray(_to_uint8(image), mode="RGB").save(Path(path))


def encode_png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(_to_uint8(image), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()
