from fastapi import FastAPI

from proimages.api.http.routes import router as jobs_router
from proimages.api.lifespan import lifespan

app = FastAPI(title="ProImages API", lifespan=lifespan)
app.include_router(jobs_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
