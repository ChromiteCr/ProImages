import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from proimages.api.jobs.store import JobStore
from proimages.gpu_config import detect_device

DEFAULT_MAX_JOBS = 1


def max_concurrent_jobs() -> int:
    """PROIMAGES_MAX_JOBS: how many jobs may process at once; the rest wait as "pending".

    Every running job holds its decoded photo plus each stage's working copies in memory
    (the planned depth-of-field renderer alone needs about 2.6 GB at 8 MP), so the default
    runs one at a time.
    """
    raw = os.environ.get("PROIMAGES_MAX_JOBS", str(DEFAULT_MAX_JOBS))
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"PROIMAGES_MAX_JOBS must be a positive integer, got {raw!r}")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.device = detect_device(override=os.environ.get("PROIMAGES_DEVICE"))
    app.state.job_store = JobStore()
    app.state.job_slots = asyncio.Semaphore(max_concurrent_jobs())
    # The event loop only keeps weak references to tasks; hold queued and running jobs here.
    app.state.job_tasks = set()
    yield
