import asyncio
import builtins
import io
import json
import threading
import time
import xml.etree.ElementTree as ET

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from proimages.api.app import app
from proimages.core.lut_gen import llm
from proimages.core.lut_gen.params import GradeParams

VALID_PAYLOAD = {
    "name": "Summer Film",
    "description": "warm and slightly faded",
    "temperature": 0.5,
    "saturation": 0.8,
}

RED = (255, 0, 0)
BLUE = (0, 0, 255)
EXIF_ORIENTATION = 0x0112


def _sample_jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(120, 60, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _rotated_jpeg_bytes() -> bytes:
    """An 8x4 JPEG whose top-left quadrant is red, tagged EXIF orientation 6
    ("rotate 90 degrees clockwise to display")."""
    img = Image.new("RGB", (8, 4), color=BLUE)
    img.paste(RED, (0, 0, 4, 2))
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = 6
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=100, exif=exif)
    return buffer.getvalue()


def _fake_client(*_args, **_kwargs):
    class FakeCompletions:
        def create(self, **_kwargs):
            message = type("Message", (), {"content": json.dumps(VALID_PAYLOAD)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    return type("FakeClient", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})})


def _bake_body(**params) -> dict:
    return {"params": GradeParams(**params).model_dump(), "size": 4}


def test_bake_returns_a_cube_without_needing_a_key():
    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json={"params": GradeParams(name="Test").model_dump(), "size": 4})

        assert response.status_code == 200
        body = response.json()
        assert 'TITLE "Test"' in body["cube"]
        assert "LUT_3D_SIZE 4" in body["cube"]
        assert body["params"]["schema_version"] == 2


def test_bake_returns_the_cdl_and_a_cc_document():
    params = GradeParams(name="Golden Hour!", description="warm", temperature=0.4, contrast=1.2, gamma=(1.1, 1.0, 0.9))

    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json={"params": params.model_dump(), "size": 4})

    assert response.status_code == 200
    body = response.json()
    assert set(body["cdl"]) == {"slope", "offset", "power", "saturation"}
    assert len(body["cdl"]["slope"]) == 3
    assert body["cc"].startswith('<?xml version="1.0" encoding="UTF-8"?>')

    root = ET.fromstring(body["cc"].encode("utf-8"))
    assert root.tag == "ColorCorrection"
    assert root.get("id") == "golden_hour"
    slope = [float(v) for v in root.findtext("SOPNode/Slope").split()]
    assert slope == pytest.approx(body["cdl"]["slope"], abs=1e-6)
    assert float(root.findtext("SatNode/Saturation")) == pytest.approx(body["cdl"]["saturation"], abs=1e-6)


def test_openapi_documents_the_response_and_the_strict_params_schema():
    with TestClient(app) as client:
        schemas = client.get("/openapi.json").json()["components"]["schemas"]

    assert set(schemas["LutResponse"]["properties"]) == {"params", "cube", "cdl", "cc"}
    assert set(schemas["CDL"]["properties"]) == {"slope", "offset", "power", "saturation"}
    assert schemas["GradeParams"]["additionalProperties"] is False


def test_bake_rejects_out_of_range_params():
    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json={"params": {"schema_version": 2, "temperature": 5.0}})

        assert response.status_code == 422
        assert "temperature" in response.text


def test_bake_requires_schema_version_in_params():
    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json={"params": {"name": "Legacy", "gain": [0, 0, 0]}, "size": 4})

        assert response.status_code == 422
        assert "schema_version" in response.text


def test_bake_rejects_a_v1_payload():
    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json={"params": {"schema_version": 1, "gain": [0, 0, 0]}, "size": 4})

        assert response.status_code == 422
        assert "no longer supported" in response.text


def test_bake_rejects_gain_below_lift_with_422_not_500():
    body = {"params": GradeParams().model_dump() | {"lift": [0.4, 0.0, 0.0], "gain": [0.2, 1.0, 1.0]}, "size": 4}

    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json=body)

        assert response.status_code == 422
        assert "gain" in response.text


def test_bake_rejects_unknown_params_fields():
    body = {"params": GradeParams().model_dump() | {"vibrance": 0.5}, "size": 4}

    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json=body)

        assert response.status_code == 422


@pytest.mark.parametrize("size", [1, 65])
def test_bake_rejects_out_of_range_sizes(size: int):
    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json={"params": GradeParams().model_dump(), "size": size})

        assert response.status_code == 422


def test_generate_from_description(monkeypatch):
    monkeypatch.setattr(llm, "_client", _fake_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/luts/generate",
            data={"model": "gpt-4o", "api_key": "sk-test", "describe": "warm summer film", "size": 4},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["params"]["name"] == "Summer Film"
        assert body["params"]["schema_version"] == 2
        assert "LUT_3D_SIZE 4" in body["cube"]
        assert body["cc"].startswith("<?xml")
        assert body["cdl"]["saturation"] == pytest.approx(0.8)


def test_generate_from_reference_image(monkeypatch):
    monkeypatch.setattr(llm, "_client", _fake_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/luts/generate",
            data={"model": "gpt-4o", "api_key": "sk-test", "size": 4},
            files={"reference": ("ref.jpg", _sample_jpeg_bytes(), "image/jpeg")},
        )

        assert response.status_code == 200
        assert response.json()["params"]["name"] == "Summer Film"


def test_generate_hands_the_model_the_reference_image_upright(monkeypatch):
    received: list[np.ndarray] = []

    def fake_params_from_reference_image(image, *_args):
        received.append(image)
        return GradeParams()

    monkeypatch.setattr(llm, "params_from_reference_image", fake_params_from_reference_image)

    with TestClient(app) as client:
        response = client.post(
            "/v1/luts/generate",
            data={"model": "gpt-4o", "api_key": "sk-test", "size": 4},
            files={"reference": ("ref.jpg", _rotated_jpeg_bytes(), "image/jpeg")},
        )

        assert response.status_code == 200

    (image,) = received
    # Displayed upright the 8x4 upload is rotated clockwise: 4 wide, 8 tall, red quadrant at the top right.
    assert image.shape == (8, 4, 3)
    assert image[1, 3, 0] > 0.8 and image[1, 3, 2] < 0.2
    assert image[6, 0, 2] > 0.8 and image[6, 0, 0] < 0.2


def _explode_if_the_model_is_called(monkeypatch) -> None:
    def exploding_client(*_args, **_kwargs):
        raise AssertionError("the model must not be called for an unreadable reference")

    monkeypatch.setattr(llm, "_client", exploding_client)


def test_generate_reports_a_missing_decoder_as_501_with_the_extra_to_install(monkeypatch):
    _explode_if_the_model_is_called(monkeypatch)
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "rawpy":
            raise ImportError("No module named 'rawpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with TestClient(app) as client:
        response = client.post(
            "/v1/luts/generate",
            data={"model": "gpt-4o", "api_key": "sk-secret-value", "size": 4},
            files={"reference": ("photo.dng", b"not really a raw file", "application/octet-stream")},
        )

    assert response.status_code == 501
    assert "proimages[heavy]" in response.json()["detail"]
    assert "sk-secret-value" not in response.text


def test_generate_rejects_an_unreadable_reference_with_422_not_a_model_failure(monkeypatch):
    _explode_if_the_model_is_called(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/v1/luts/generate",
            data={"model": "gpt-4o", "api_key": "sk-secret-value", "size": 4},
            files={"reference": ("ref.jpg", b"garbage bytes", "image/jpeg")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "reference is not a readable image"
    assert "sk-secret-value" not in response.text


def test_generate_requires_describe_or_reference():
    with TestClient(app) as client:
        response = client.post("/v1/luts/generate", data={"model": "gpt-4o", "api_key": "sk-test"})

        assert response.status_code == 422


@pytest.mark.parametrize("size", [1, 0, 65])
def test_generate_rejects_out_of_range_sizes_before_calling_the_model(monkeypatch, size: int):
    def exploding_client(*_args, **_kwargs):
        raise AssertionError("the model must not be called for an invalid size")

    monkeypatch.setattr(llm, "_client", exploding_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/luts/generate",
            data={"model": "gpt-4o", "api_key": "sk-test", "describe": "anything", "size": size},
        )

        assert response.status_code == 422


def test_generate_error_response_does_not_leak_the_api_key(monkeypatch):
    def exploding_client(*_args, **_kwargs):
        raise RuntimeError("upstream refused key sk-secret-value")

    monkeypatch.setattr(llm, "_client", exploding_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/luts/generate",
            data={"model": "gpt-4o", "api_key": "sk-secret-value", "describe": "anything"},
        )

        assert response.status_code == 502
        assert "sk-secret-value" not in response.text


def test_generate_reports_a_model_reply_that_stays_invalid_as_502_without_leaking(monkeypatch):
    class FakeCompletions:
        def create(self, **_kwargs):
            message = type("Message", (), {"content": json.dumps({"temperature": 4.2, "note": "sk-secret-value"})})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    fake = type("FakeClient", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})})
    monkeypatch.setattr(llm, "_client", lambda *_args: fake)

    with TestClient(app) as client:
        response = client.post(
            "/v1/luts/generate",
            data={"model": "gpt-4o", "api_key": "sk-secret-value", "describe": "anything"},
        )

        assert response.status_code == 502
        assert "sk-secret-value" not in response.text


def test_generate_runs_the_model_call_off_the_event_loop(monkeypatch):
    call_threads: list[int] = []

    class SlowCompletions:
        def create(self, **_kwargs):
            call_threads.append(threading.get_ident())
            time.sleep(0.5)
            message = type("Message", (), {"content": json.dumps(VALID_PAYLOAD)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    slow = type("SlowClient", (), {"chat": type("Chat", (), {"completions": SlowCompletions()})})
    monkeypatch.setattr(llm, "_client", lambda *_args: slow)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            generating = asyncio.create_task(
                client.post(
                    "/v1/luts/generate",
                    data={"model": "gpt-4o", "api_key": "sk-test", "describe": "anything", "size": 4},
                )
            )
            started = time.perf_counter()
            await asyncio.sleep(0.1)
            health = await client.get("/health")
            elapsed = time.perf_counter() - started
            return threading.get_ident(), health, elapsed, await generating

    loop_thread, health, elapsed, generated = asyncio.run(scenario())

    assert generated.status_code == 200
    assert health.status_code == 200
    assert elapsed < 0.4, "the event loop was blocked while the model call ran"
    assert call_threads and loop_thread not in call_threads
