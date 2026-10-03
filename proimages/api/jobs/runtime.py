import asyncio
from pathlib import Path

from proimages.api.jobs.store import JobStore
from proimages.core.pipeline import process_image
from proimages.core.system.io import decode_image, encode_png


async def run_job(
    job_id: str,
    image_bytes: bytes,
    job_store: JobStore,
    lut_path: str | Path | None = None,
    filename: str | None = None,
) -> None:
    await job_store.mark_running(job_id)
    try:
        image = await asyncio.to_thread(decode_image, image_bytes, filename)
        result = await asyncio.to_thread(process_image, image, lut_path=lut_path)
        result_bytes = await asyncio.to_thread(encode_png, result)
        await job_store.mark_completed(job_id, result_bytes)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client via job status
        await job_store.mark_failed(job_id, str(exc))
