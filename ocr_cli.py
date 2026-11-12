#!/usr/bin/env python3
"""Command-line OCR client for a running paddleocr-wrapper service. Stdlib only.

    python ocr_cli.py receipt.png                       # JSON on stdout
    python ocr_cli.py a.png b.jpg --format text         # plain text only
    python ocr_cli.py --health                          # service status JSON
    cat img.png | python ocr_cli.py -                   # image from stdin

Service URL: --host, else $PADDLE_OCR_URL, else http://127.0.0.1:8649.
Output is machine-readable: results go to stdout, diagnostics to stderr.
One image -> one JSON object; several -> a JSON array. Each result carries
the service fields (text, lines[{index,text,confidence,box}], elapsed_s,
device) plus "file"; a failed image is {"file": ..., "error": ...}.
Exit codes: 0 all OK | 1 at least one image failed | 2 service unreachable.
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_HOST = "http://127.0.0.1:8649"
MIMES = {".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}


class Unreachable(Exception):
    pass


def _request(req: urllib.request.Request, timeout: float):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", exc.reason)
        except Exception:
            detail = exc.reason
        if isinstance(detail, dict):   # service errors look like {"error": ..., "detail": ...}
            detail = detail.get("error") or json.dumps(detail)
        return exc.code, {"error": detail if isinstance(detail, str) else json.dumps(detail)}
    except (urllib.error.URLError, OSError) as exc:
        raise Unreachable(str(getattr(exc, "reason", exc))) from exc


def ocr_one(path: str, host: str, timeout: float) -> dict:
    try:
        data = sys.stdin.buffer.read() if path == "-" else open(path, "rb").read()
    except OSError as exc:
        return {"file": path, "error": f"cannot read file: {exc.strerror or exc}"}
    ext = os.path.splitext(path)[1].lower()
    body = json.dumps({"image_b64": base64.b64encode(data).decode(), "mime": MIMES.get(ext, "image/jpeg")}).encode()
    req = urllib.request.Request(f"{host}/ocr", data=body, headers={"Content-Type": "application/json"})
    status, payload = _request(req, timeout)
    if status != 200:
        return {"file": path, "error": payload.get("error", f"HTTP {status}")}
    return {"file": path, **payload}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OCR images via a paddleocr-wrapper service.",
                                 epilog="Exit codes: 0 ok | 1 an image failed | 2 service unreachable.")
    ap.add_argument("images", nargs="*", help="image paths, or '-' for stdin")
    ap.add_argument("--host", default=os.environ.get("PADDLE_OCR_URL", DEFAULT_HOST))
    ap.add_argument("--format", choices=["json", "text"], default="json")
    ap.add_argument("--timeout", type=float, default=120, help="per-request seconds (default 120)")
    ap.add_argument("--health", action="store_true", help="print /health and exit")
    args = ap.parse_args(argv)
    host = args.host.rstrip("/")

    try:
        if args.health:
            status, payload = _request(urllib.request.Request(f"{host}/health"), args.timeout)
            print(json.dumps(payload, ensure_ascii=False))
            return 0 if status == 200 and payload.get("status") == "ok" else 1
        if not args.images:
            ap.error("give at least one image path (or --health)")
        results = [ocr_one(p, host, args.timeout) for p in args.images]
    except Unreachable as exc:
        print(f"error: OCR service unreachable at {host}: {exc}", file=sys.stderr)
        return 2

    if args.format == "text":
        for r in results:
            print(r["text"] if "text" in r else f"[{r['file']}] ERROR: {r['error']}",
                  file=sys.stdout if "text" in r else sys.stderr)
    else:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))
    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
