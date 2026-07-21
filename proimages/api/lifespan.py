import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from proimages.api.jobs.store import JobStore
from proimages.gpu_config import detect_device


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.device = detect_device(override=os.environ.get("PROIMAGES_DEVICE"))
    app.state.job_store = JobStore()
    yield
