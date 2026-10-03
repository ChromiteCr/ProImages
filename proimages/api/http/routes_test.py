import builtins
import io
import time
from pathlib import Path

import colour
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from proimages.api.app import app

RED = (255, 0, 0)
BLUE = (0, 0, 255)
EXIF_ORIENTATION = 0x0112


def _sample_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(120, 60, 200)).save(buffer, format="PNG")
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


def _block_import(monkeypatch, blocked: str) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == blocked:
            raise ImportError(f"No module named '{blocked}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


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


def test_job_result_honours_the_exif_orientation_of_the_upload():
    with TestClient(app) as client:
        response = client.post("/v1/jobs", files={"file": ("photo.jpg", _rotated_jpeg_bytes(), "image/jpeg")})
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        status = _poll_until_done(client, job_id)
        assert status["status"] == "completed", status

        result = client.get(f"/v1/jobs/{job_id}/result")
        assert result.status_code == 200

    png = Image.open(io.BytesIO(result.content))
    # Displayed upright the 8x4 upload is rotated clockwise: 4 wide, 8 tall, red quadrant at the top right.
    assert png.size == (4, 8)
    pixels = np.asarray(png.convert("RGB")).astype(int)
    # Probe away from the red/blue seam only: JPEG chroma subsampling blurs it, and the pipeline adds
    # vignette and random grain on top. These pixels stay far apart regardless (worst case over 2000 runs: 104).
    red_column = pixels[0:3, 3]
    blue_rows = pixels[5:8]
    assert (red_column[:, 0] - red_column[:, 2] > 60).all()
    assert (blue_rows[..., 2] - blue_rows[..., 0] > 60).all()


def test_raw_upload_without_rawpy_fails_the_job_and_points_at_the_heavy_extra(monkeypatch):
    _block_import(monkeypatch, "rawpy")

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs", files={"file": ("photo.dng", b"not decodable without rawpy", "application/octet-stream")}
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        status = _poll_until_done(client, job_id)
        assert status["status"] == "failed"
        assert "proimages[heavy]" in status["error"]

        result = client.get(f"/v1/jobs/{job_id}/result")
        assert result.status_code == 409
        assert "failed" in result.json()["detail"]
