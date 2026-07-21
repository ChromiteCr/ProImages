import asyncio
import uuid

from proimages.api.jobs.models import JobRecord, JobStatus


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> JobRecord:
        async with self._lock:
            record = JobRecord(job_id=str(uuid.uuid4()))
            self._jobs[record.job_id] = record
            return record

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def mark_running(self, job_id: str) -> None:
        async with self._lock:
            self._jobs[job_id].status = JobStatus.RUNNING

    async def mark_completed(self, job_id: str, result: bytes) -> None:
        async with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.COMPLETED
            record.result = result

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.FAILED
            record.error = error
