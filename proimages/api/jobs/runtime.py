import asyncio
import io

import numpy as np
from PIL import Image

from proimages.api.jobs.store import JobStore
from proimages.core.pipeline import process_image


def _decode(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        return (np.asarray(rgb).astype(np.float32) / 255.0).clip(0.0, 1.0)


def _encode(image: np.ndarray) -> bytes:
    encoded = (image.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(encoded, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


async def run_job(job_id: str, image_bytes: bytes, job_store: JobStore) -> None:
    await job_store.mark_running(job_id)
    try:
        image = await asyncio.to_thread(_decode, image_bytes)
        result = await asyncio.to_thread(process_image, image)
        result_bytes = await asyncio.to_thread(_encode, result)
        await job_store.mark_completed(job_id, result_bytes)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client via job status
        await job_store.mark_failed(job_id, str(exc))
