import asyncio
import functools
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, Response, UploadFile
from pydantic import Json

from proimages.api.jobs.models import JobStatus
from proimages.api.jobs.runtime import run_job
from proimages.api.jobs.store import JobStore
from proimages.core.options import ProcessOptions

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _job_store(request: Request) -> JobStore:
    return request.app.state.job_store


async def _run_and_cleanup(job: Callable[[], Awaitable[None]], lut_path: Path | None) -> None:
    try:
        await job()
    finally:
        if lut_path is not None:
            lut_path.unlink(missing_ok=True)


@router.post("")
async def submit_job(
    request: Request,
    file: UploadFile,
    lut: UploadFile | None = None,
    options: Json[ProcessOptions] | None = Form(
        default=None, description="ProcessOptions as a JSON string; leave it out for the default pipeline."
    ),
):
    # FastAPI validates `options` before this body runs, so a bad value is a 422 that leaves
    # no job and no temporary LUT file behind.
    state = request.app.state
    image_bytes = await file.read()

    lut_path: Path | None = None
    if lut is not None:
        lut_bytes = await lut.read()
        with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as tmp:
            tmp.write(lut_bytes)
            lut_path = Path(tmp.name)

    # Only now: a failure above must not leave a job that nothing will ever run.
    record = await state.job_store.create()
    job = functools.partial(
        run_job,
        record.job_id,
        image_bytes,
        state.job_store,
        state.job_slots,
        lut_path=lut_path,
        filename=file.filename,
        options=options,
        device=state.device,
    )
    task = asyncio.create_task(_run_and_cleanup(job, lut_path))
    state.job_tasks.add(task)
    task.add_done_callback(state.job_tasks.discard)
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
