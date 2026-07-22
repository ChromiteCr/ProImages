import io

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from proimages.core.lut_gen.bake import DEFAULT_LUT_SIZE, bake_cube_text
from proimages.core.lut_gen.params import GradeParams

router = APIRouter(prefix="/v1/luts", tags=["luts"])


class BakeRequest(BaseModel):
    params: GradeParams
    size: int = Field(default=DEFAULT_LUT_SIZE, ge=2, le=64)


class LutResponse(BaseModel):
    params: GradeParams
    cube: str


@router.post("/bake", response_model=LutResponse)
async def bake(request: BakeRequest) -> LutResponse:
    """Bake parameters into a .cube. No model call, no API key -- this is what the
    grading pad calls after the user drags it."""
    return LutResponse(params=request.params, cube=bake_cube_text(request.params, size=request.size))


@router.post("/generate", response_model=LutResponse)
async def generate(
    model: str = Form(...),
    api_key: str = Form(...),
    describe: str | None = Form(default=None),
    base_url: str | None = Form(default=None),
    size: int = Form(default=DEFAULT_LUT_SIZE),
    reference: UploadFile | None = File(default=None),
) -> LutResponse:
    """Ask a model for grading parameters, then bake them.

    The api_key lives only for this request: it is never stored, logged, or echoed
    back. Errors below are deliberately reported without the request body so a key
    cannot leak through an error message.
    """
    if describe is None and reference is None:
        raise HTTPException(status_code=422, detail="provide either 'describe' or 'reference'")

    from proimages.core.lut_gen.llm import params_from_description, params_from_reference_image

    try:
        if describe is not None:
            params = params_from_description(describe, model, api_key, base_url)
        else:
            image_bytes = await reference.read()
            with Image.open(io.BytesIO(image_bytes)) as img:
                image = (np.asarray(img.convert("RGB")).astype(np.float32) / 255.0).clip(0.0, 1.0)
            params = params_from_reference_image(image, model, api_key, base_url)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"model request failed: {type(exc).__name__}") from None

    return LutResponse(params=params, cube=bake_cube_text(params, size=size))
