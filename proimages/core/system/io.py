from pathlib import Path

import numpy as np
from PIL import Image

RAW_EXTENSIONS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2"}


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(output_bps=16)
        return (rgb.astype(np.float32) / 65535.0).clip(0.0, 1.0)

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return (np.asarray(rgb).astype(np.float32) / 255.0).clip(0.0, 1.0)


def save_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    encoded = (image.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(path)
