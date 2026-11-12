import base64
import struct
import zlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeModel


@pytest.fixture
def client(service, fresh_state):
    fresh_state.update(model=FakeModel(), device="cpu", requested_device="cpu", max_upload_bytes=2000)
    return TestClient(service.app)


def png(w, h):
    """Minimal PNG whose IHDR claims w x h (pixel data absent: only the header is sniffed)."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))


def jpeg(w, h):
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00" + sof


def real_png(n=20):
    return cv2.imencode(".png", np.zeros((n, n, 3), np.uint8))[1].tobytes()


def test_sniff_dimensions(service):
    assert service.image_dimensions(png(123, 456)) == (123, 456)
    assert service.image_dimensions(jpeg(321, 654)) == (321, 654)
    assert service.image_dimensions(b"not an image") is None


def test_small_upload_ok(client):
    r = client.post("/ocr/upload", files={"image": ("a.png", real_png(), "image/png")})
    assert r.status_code == 200


def test_oversize_multipart_is_413(client):
    r = client.post("/ocr/upload", files={"image": ("big.png", b"x" * 5000, "image/png")})
    assert r.status_code == 413 and "limit" in r.text.lower()


def test_oversize_json_body_is_413(client):
    r = client.post("/ocr", json={"image_b64": base64.b64encode(b"x" * 5000).decode()})
    assert r.status_code == 413


def test_chunked_body_without_content_length_is_413(client):
    def gen():
        for _ in range(10):
            yield b"x" * 1000
    r = client.post("/ocr/upload", content=gen(), headers={"Content-Type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413


def test_decoded_image_over_limit_is_413(service, fresh_state):
    from fastapi import HTTPException
    fresh_state.update(model=FakeModel(), device="cpu", requested_device="cpu", max_upload_bytes=100)
    with pytest.raises(HTTPException) as e:
        service._run_one(real_png(60), None)
    assert e.value.status_code == 413


@pytest.mark.parametrize("data", [png(30000, 30000), jpeg(30000, 30000)])
def test_pixel_bomb_is_413_before_decode(client, fresh_state, data):
    fresh_state["max_upload_bytes"] = 10_000_000
    r = client.post("/ocr/upload", files={"image": ("bomb", data, "image/png")})
    assert r.status_code == 413 and "pixels" in r.text.lower()


def test_parser_validates_limits(service):
    p = service.build_parser()
    assert p.parse_args([]).max_upload_mb == 20 and p.parse_args([]).max_pixels == 40_000_000
    for flag in ("--max-upload-mb", "--max-pixels"):
        with pytest.raises(SystemExit):
            p.parse_args([flag, "0"])
