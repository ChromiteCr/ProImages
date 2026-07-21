import asyncio

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile

from proimages.api.jobs.models import JobStatus
from proimages.api.jobs.runtime import run_job
from proimages.api.jobs.store import JobStore

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _job_store(request: Request) -> JobStore:
    return request.app.state.job_store


@router.post("")
async def submit_job(request: Request, file: UploadFile):
    job_store = _job_store(request)
    record = await job_store.create()
    image_bytes = await file.read()
    asyncio.create_task(run_job(record.job_id, image_bytes, job_store))
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
