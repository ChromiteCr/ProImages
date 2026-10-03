import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile

from proimages.api.jobs.models import JobStatus
from proimages.api.jobs.runtime import run_job
from proimages.api.jobs.store import JobStore

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _job_store(request: Request) -> JobStore:
    return request.app.state.job_store


async def _run_and_cleanup(
    job_id: str, image_bytes: bytes, job_store: JobStore, lut_path: Path | None, filename: str | None
) -> None:
    try:
        await run_job(job_id, image_bytes, job_store, lut_path=lut_path, filename=filename)
    finally:
        if lut_path is not None:
            lut_path.unlink(missing_ok=True)


@router.post("")
async def submit_job(request: Request, file: UploadFile, lut: UploadFile | None = None):
    job_store = _job_store(request)
    record = await job_store.create()
    image_bytes = await file.read()

    lut_path: Path | None = None
    if lut is not None:
        lut_bytes = await lut.read()
        with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as tmp:
            tmp.write(lut_bytes)
            lut_path = Path(tmp.name)

    asyncio.create_task(_run_and_cleanup(record.job_id, image_bytes, job_store, lut_path, file.filename))
    return {"job_id": record.job_id, "status": record.status.value}


@router.get("/{job_id}")
async def get_job_status(job_id: str, request: Request):
    record = await _job_store(request).get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": record.job_id, "status": record.status.value, "error": record.error}


@router.get("/{job_id}/result")
async def get_job_result(job_id: str, request: Request):
    record = await _job_store(request).get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    if record.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"job is {record.status.value}")
    return Response(content=record.result, media_type="image/png")
