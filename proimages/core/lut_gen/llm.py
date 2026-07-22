import base64
import io
import json

import numpy as np
from openai import OpenAI
from PIL import Image

from proimages.core.lut_gen.image_stats import extract_color_stats
from proimages.core.lut_gen.params import GradeParams
from proimages.core.lut_gen.prompts import DESCRIPTION_PROMPT, REFERENCE_PROMPT, SYSTEM_PROMPT

REFERENCE_MAX_EDGE = 512


def _client(api_key: str, base_url: str | None) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def _parse_params(content: str) -> GradeParams:
    payload = json.loads(content)
    payload.pop("schema_version", None)
    return GradeParams.model_validate(payload)


def _complete(client: OpenAI, model: str, messages: list[dict]) -> GradeParams:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return _parse_params(response.choices[0].message.content)


def _encode_reference(image: np.ndarray) -> str:
    height, width = image.shape[:2]
    scale = min(1.0, REFERENCE_MAX_EDGE / max(height, width))
    pil = Image.fromarray((image.clip(0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")
    if scale < 1.0:
        pil = pil.resize((max(1, int(width * scale)), max(1, int(height * scale))))

    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def params_from_description(description: str, model: str, api_key: str, base_url: str | None = None) -> GradeParams:
    return _complete(
        _client(api_key, base_url),
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": DESCRIPTION_PROMPT.format(description=description)},
        ],
    )


def params_from_reference_image(
    image: np.ndarray, model: str, api_key: str, base_url: str | None = None
) -> GradeParams:
    stats = json.dumps(extract_color_stats(image), indent=2)
    encoded = _encode_reference(image)

    return _complete(
        _client(api_key, base_url),
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": REFERENCE_PROMPT.format(stats=stats)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            },
        ],
    )
