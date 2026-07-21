import io
import time
from pathlib import Path

import colour
from fastapi.testclient import TestClient
from PIL import Image

from proimages.api.app import app


def _sample_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(120, 60, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def _poll_until_done(client: TestClient, job_id: str) -> dict:
    for _ in range(50):
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in {"completed", "failed"}:
            return status
        time.sleep(0.05)
    raise AssertionError(f"job did not complete in time, last status: {status}")


def test_submit_poll_and_fetch_job_result():
    with TestClient(app) as client:
        response = client.post("/v1/jobs", files={"file": ("sample.png", _sample_png_bytes(), "image/png")})
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        status = _poll_until_done(client, job_id)
        assert status["status"] == "completed"

        result = client.get(f"/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        assert result.headers["content-type"] == "image/png"


def test_submit_with_lut_completes(tmp_path: Path):
    identity_lut = colour.LUT3D(colour.LUT3D.linear_table(9), name="identity")
    lut_path = tmp_path / "identity.cube"
    colour.write_LUT(identity_lut, str(lut_path))

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            files={
                "file": ("sample.png", _sample_png_bytes(), "image/png"),
                "lut": ("identity.cube", lut_path.read_bytes(), "application/octet-stream"),
            },
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        status = _poll_until_done(client, job_id)
        assert status["status"] == "completed"


def test_unknown_job_returns_404():
    with TestClient(app) as client:
        response = client.get("/v1/jobs/does-not-exist")
        assert response.status_code == 404
