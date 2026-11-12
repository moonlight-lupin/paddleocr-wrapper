import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeModel


def png_b64():
    ok, buf = cv2.imencode(".png", np.full((20, 40, 3), 255, dtype=np.uint8))
    return base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def client(service):
    return TestClient(service.app)          # no context manager: startup/lifespan not run


def test_ocr_503_until_model_ready(client):
    assert client.post("/ocr", json={"image_b64": png_b64()}).status_code == 503


def test_ocr_json_and_upload_roundtrip(client, fresh_state):
    fresh_state.update(model=FakeModel(["hi"], [0.97]), device="cpu", requested_device="cpu")
    r = client.post("/ocr", json={"image_b64": png_b64(), "mime": "image/png"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"text", "lines", "elapsed_s", "device"} and body["text"] == "hi"
    ok, buf = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))
    r = client.post("/ocr/upload", files={"image": ("a.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200


def test_bad_base64_and_bad_image_are_400(client, fresh_state):
    fresh_state.update(model=FakeModel(), device="cpu", requested_device="cpu")
    assert client.post("/ocr", json={"image_b64": "!!!"}).status_code == 400
    assert client.post("/ocr", json={"image_b64": base64.b64encode(b"not an image").decode()}).status_code == 400


def test_health_warming_then_ok(client, fresh_state):
    assert client.get("/health").json()["status"] == "warming"
    fresh_state.update(model=FakeModel(), device="cpu")
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["engine_chain"] == ["paddle-cpu"]   # CPU boot: no GPU tier listed


def test_gui_and_logs(client):
    assert "PaddleOCR service" in client.get("/").text
    assert "lines" in client.get("/logs").json()


def test_default_rapid_python_is_platform_aware(service):
    import os
    assert service.default_rapid_python("/svc", platform="win32") == os.path.join("/svc", ".venv-rapid", "Scripts", "python.exe")
    assert service.default_rapid_python("/svc", platform="linux") == os.path.join("/svc", ".venv-rapid", "bin", "python")


def test_health_reports_error_when_init_failed(client, fresh_state):
    fresh_state["error"] = "gpu:99: Device id must be less than GPU count"
    h = client.get("/health").json()
    assert h["status"] == "error" and "gpu:99" in h["init_error"]


def test_health_error_not_shown_when_a_later_candidate_succeeded(client, fresh_state):
    fresh_state.update(error="gpu:0: boom", model=FakeModel(), device="cpu")
    assert client.get("/health").json()["status"] == "ok"
