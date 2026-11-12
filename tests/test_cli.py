import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import ocr_cli  # noqa: E402

RESULT = {"text": "hi", "lines": [{"index": 0, "text": "hi", "confidence": 0.99,
                                    "box": [[0, 0], [4, 0], [4, 4], [0, 4]]}], "elapsed_s": 0.1, "device": "cpu"}


@pytest.fixture
def server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"status": "ok"})

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if seen[-1]["image_b64"] == "Ym9vbQ==":   # b"boom"
                self._send(400, {"detail": "cannot decode image"})
            elif seen[-1]["image_b64"] == "aHVnZQ==":   # b"huge"
                self._send(413, {"detail": {"error": "request body exceeds the 20 MiB limit (--max-upload-mb)"}})
            else:
                self._send(200, RESULT)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def test_single_image_prints_json(server, tmp_path, capsys):
    url, seen = server
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNGdata")
    assert ocr_cli.main([str(img), "--host", url]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["text"] == "hi" and out["file"] == str(img)
    assert seen[0]["mime"] == "image/png"


def test_text_format_prints_only_text(server, tmp_path, capsys):
    url, _ = server
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    assert ocr_cli.main([str(img), "--host", url, "--format", "text"]) == 0
    assert capsys.readouterr().out.strip() == "hi"


def test_multiple_images_json_array_and_partial_failure_exit_code(server, tmp_path, capsys):
    url, _ = server
    good, bad = tmp_path / "g.png", tmp_path / "b.png"
    good.write_bytes(b"ok")
    bad.write_bytes(b"boom")
    code = ocr_cli.main([str(good), str(bad), "--host", url])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and len(out) == 2
    assert out[0]["text"] == "hi" and "error" in out[1]


def test_missing_file_is_json_error_not_traceback(server, capsys):
    url, _ = server
    code = ocr_cli.main(["/nope.png", "--host", url])
    assert code == 1 and "error" in json.loads(capsys.readouterr().out)


def test_unreachable_service_exit_2(tmp_path, capsys):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    assert ocr_cli.main([str(img), "--host", "http://127.0.0.1:9", "--timeout", "2"]) == 2
    assert "unreachable" in capsys.readouterr().err


def test_health_flag(server, capsys):
    url, _ = server
    assert ocr_cli.main(["--health", "--host", url]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_host_from_env(server, tmp_path, capsys, monkeypatch):
    url, _ = server
    monkeypatch.setenv("PADDLE_OCR_URL", url)
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    assert ocr_cli.main([str(img)]) == 0


def test_structured_service_error_is_unwrapped(server, tmp_path, capsys):
    url, _ = server
    img = tmp_path / "h.png"
    img.write_bytes(b"huge")
    assert ocr_cli.main([str(img), "--host", url]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "request body exceeds the 20 MiB limit (--max-upload-mb)"
