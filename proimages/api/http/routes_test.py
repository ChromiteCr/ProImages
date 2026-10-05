import builtins
import io
import tempfile
import threading
import time
from pathlib import Path

import colour
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from proimages.api.app import app
from proimages.api.jobs import runtime
from proimages.core.options import ProcessOptions

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


def _submit(client: TestClient, **data: str) -> str:
    response = client.post("/v1/jobs", files={"file": ("sample.png", _sample_png_bytes(), "image/png")}, data=data)
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def _wait_for(condition, what: str) -> None:
    for _ in range(100):
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def _gated_pipeline(monkeypatch) -> tuple[threading.Event, dict]:
    """Swap in a pipeline that blocks until released and records how many runs overlapped."""
    release = threading.Event()
    lock = threading.Lock()
    runs = {"active": 0, "peak": 0}

    def fake_process_image(image: np.ndarray, **kwargs) -> np.ndarray:
        with lock:
            runs["active"] += 1
            runs["peak"] = max(runs["peak"], runs["active"])
        release.wait(timeout=10)
        with lock:
            runs["active"] -= 1
        return image

    monkeypatch.setattr(runtime, "process_image", fake_process_image)
    return release, runs


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


def test_the_uploaded_lut_is_removed_whether_the_job_completes_or_fails(tmp_path: Path, monkeypatch):
    identity_lut = colour.LUT3D(colour.LUT3D.linear_table(9), name="identity")
    lut_path = tmp_path / "identity.cube"
    colour.write_LUT(identity_lut, str(lut_path))
    lut = ("identity.cube", lut_path.read_bytes(), "application/octet-stream")
    uploads = tmp_path / "uploads"  # where the API keeps the LUTs it is sent, apart from the one written above
    uploads.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(uploads))

    with TestClient(app) as client:
        completing = client.post(
            "/v1/jobs", files={"file": ("sample.png", _sample_png_bytes(), "image/png"), "lut": lut}
        ).json()["job_id"]
        failing = client.post(
            "/v1/jobs", files={"file": ("garbage.png", b"not an image", "image/png"), "lut": lut}
        ).json()["job_id"]

        assert _poll_until_done(client, completing)["status"] == "completed"
        assert _poll_until_done(client, failing)["status"] == "failed"
        # the file goes right after the job's last status update, so allow it a moment
        _wait_for(lambda: not list(uploads.glob("*.cube")), "the uploaded LUTs to be removed")


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


@pytest.mark.parametrize("options", ["not json", "[]", '{"denoise": {}}'])
def test_invalid_options_are_rejected_before_a_job_is_created(options: str):
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs", files={"file": ("sample.png", _sample_png_bytes(), "image/png")}, data={"options": options}
        )

        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"][:2] == ["body", "options"]
        assert client.app.state.job_store._jobs == {}


def test_a_failure_to_store_the_upload_leaves_no_job_behind(monkeypatch):
    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", disk_full)

    with TestClient(app) as client:
        with pytest.raises(OSError, match="disk full"):
            client.post(
                "/v1/jobs",
                files={
                    "file": ("sample.png", _sample_png_bytes(), "image/png"),
                    "lut": ("identity.cube", b"unread", "application/octet-stream"),
                },
            )

        assert client.app.state.job_store._jobs == {}  # a pending job nobody runs would stay pending forever


def test_jobs_run_on_the_server_device_with_the_submitted_options(monkeypatch):
    monkeypatch.setenv("PROIMAGES_DEVICE", "mps")
    calls = []

    def fake_process_image(image: np.ndarray, **kwargs) -> np.ndarray:
        calls.append(kwargs)
        return image

    monkeypatch.setattr(runtime, "process_image", fake_process_image)

    with TestClient(app) as client:
        job_id = _submit(client, options="{}")
        assert _poll_until_done(client, job_id)["status"] == "completed"

    assert calls == [{"lut_path": None, "options": ProcessOptions(), "device": "mps"}]


def test_jobs_beyond_the_slot_limit_wait_as_pending(monkeypatch):
    monkeypatch.setenv("PROIMAGES_MAX_JOBS", "1")
    release, runs = _gated_pipeline(monkeypatch)

    with TestClient(app) as client:
        try:
            first = _submit(client)
            _wait_for(lambda: runs["active"] == 1, "the first job to start processing")
            second = _submit(client)

            # The first job holds the only slot, so the second stays pending for as long as we look.
            for _ in range(10):
                assert client.get(f"/v1/jobs/{second}").json()["status"] == "pending"
                time.sleep(0.03)
        finally:
            release.set()

        assert _poll_until_done(client, first)["status"] == "completed"
        assert _poll_until_done(client, second)["status"] == "completed"

    assert runs["peak"] == 1


def test_max_jobs_lets_that_many_jobs_run_at_once(monkeypatch):
    monkeypatch.setenv("PROIMAGES_MAX_JOBS", "2")
    release, runs = _gated_pipeline(monkeypatch)

    with TestClient(app) as client:
        try:
            job_ids = [_submit(client) for _ in range(2)]
            _wait_for(lambda: runs["active"] == 2, "both jobs to be processing together")
        finally:
            release.set()

        for job_id in job_ids:
            assert _poll_until_done(client, job_id)["status"] == "completed"
