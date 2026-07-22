import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from proimages.api.app import app
from proimages.core.lut_gen import llm
from proimages.core.lut_gen.params import GradeParams

VALID_PAYLOAD = {
    "name": "Summer Film",
    "description": "warm and slightly faded",
    "temperature": 0.5,
    "saturation": -0.2,
}


def _sample_jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(120, 60, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _fake_client(*_args, **_kwargs):
    class FakeCompletions:
        def create(self, **_kwargs):
            message = type("Message", (), {"content": json.dumps(VALID_PAYLOAD)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    return type("FakeClient", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})})


def test_bake_returns_a_cube_without_needing_a_key():
    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json={"params": GradeParams(name="Test").model_dump(), "size": 4})

        assert response.status_code == 200
        body = response.json()
        assert 'TITLE "Test"' in body["cube"]
        assert "LUT_3D_SIZE 4" in body["cube"]
        assert body["params"]["schema_version"] == 1


def test_bake_rejects_out_of_range_params():
    with TestClient(app) as client:
        response = client.post("/v1/luts/bake", json={"params": {"temperature": 5.0}})

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
        assert "LUT_3D_SIZE 4" in body["cube"]


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


def test_generate_requires_describe_or_reference():
    with TestClient(app) as client:
        response = client.post("/v1/luts/generate", data={"model": "gpt-4o", "api_key": "sk-test"})

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
