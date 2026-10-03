import asyncio
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

from proimages.core.lut_gen.bake import DEFAULT_LUT_SIZE, MAX_LUT_SIZE, MIN_LUT_SIZE, bake_cube_text
from proimages.core.lut_gen.cdl import CDL, cc_id_from_name, to_cc_xml, to_cdl
from proimages.core.lut_gen.params import SCHEMA_VERSION, GradeParams
from proimages.core.system.io import decode_image

router = APIRouter(prefix="/v1/luts", tags=["luts"])


class BakeRequest(BaseModel):
    params: GradeParams
    size: int = Field(default=DEFAULT_LUT_SIZE, ge=MIN_LUT_SIZE, le=MAX_LUT_SIZE)

    @model_validator(mode="before")
    @classmethod
    def _require_schema_version(cls, data: Any) -> Any:
        """GradeParams fills a missing schema_version with the current one, which would let an
        old v1 payload through as v2 (its neutral gain of 0 would bake to black), so demand it."""
        params = data.get("params") if isinstance(data, dict) else None
        if isinstance(params, dict) and "schema_version" not in params:
            raise ValueError(
                f"params.schema_version is required and must be {SCHEMA_VERSION}: "
                "lift/gamma/gain now use standard neutral values 0/1/1"
            )
        return data


class LutResponse(BaseModel):
    params: GradeParams
    cube: str = Field(description="The .cube LUT text, a sampled approximation of the grade.")
    cdl: CDL = Field(description="The grade folded into one ASC CDL v1.2 correction (exact).")
    cc: str = Field(description="The same CDL as an ASC ColorCorrection (.cc) XML document.")


def _build_response(params: GradeParams, size: int) -> LutResponse:
    cdl = to_cdl(params)
    return LutResponse(
        params=params,
        cube=bake_cube_text(params, size=size),
        cdl=cdl,
        cc=to_cc_xml(cdl, cc_id_from_name(params.name), params.description),
    )


async def _decode_reference(reference: UploadFile) -> np.ndarray:
    """Decode outside the model-error handler, so an unreadable upload or a missing
    optional decoder is reported as such instead of as a model failure."""
    data = await reference.read()
    try:
        return await asyncio.to_thread(decode_image, data, reference.filename)
    except ImportError as exc:
        # The message names the extra to install and carries no request data.
        raise HTTPException(status_code=501, detail=str(exc)) from None
    except Exception:
        raise HTTPException(status_code=422, detail="reference is not a readable image") from None


@router.post("/bake", response_model=LutResponse)
async def bake(request: BakeRequest) -> LutResponse:
    """Bake parameters into a .cube, plus the exact ASC CDL and its .cc file. No model call,
    no API key -- this is what the grading pad calls after the user drags it."""
    return await asyncio.to_thread(_build_response, request.params, request.size)


@router.post("/generate", response_model=LutResponse)
async def generate(
    model: str = Form(...),
    api_key: str = Form(...),
    describe: str | None = Form(default=None),
    base_url: str | None = Form(default=None),
    size: int = Form(default=DEFAULT_LUT_SIZE, ge=MIN_LUT_SIZE, le=MAX_LUT_SIZE),
    reference: UploadFile | None = File(default=None),
) -> LutResponse:
    """Ask a model for grading parameters, then bake them.

    The api_key lives only for this request: it is never stored, logged, or echoed
    back. Errors below are deliberately reported without the request body so a key
    cannot leak through an error message.
    """
    if describe is None and reference is None:
        raise HTTPException(status_code=422, detail="provide either 'describe' or 'reference'")

    image = await _decode_reference(reference) if describe is None else None

    from proimages.core.lut_gen.llm import params_from_description, params_from_reference_image

    try:
        if image is None:
            params = await asyncio.to_thread(params_from_description, describe, model, api_key, base_url)
        else:
            params = await asyncio.to_thread(params_from_reference_image, image, model, api_key, base_url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"model request failed: {type(exc).__name__}") from None

    return await asyncio.to_thread(_build_response, params, size)
