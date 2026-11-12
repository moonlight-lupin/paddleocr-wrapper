"""Fake tier-2 child speaking the rapid_worker line protocol. MODE env var picks behaviour."""
import json
import os
import sys
import time

mode = os.environ.get("FAKE_MODE", "ok")
if mode == "die_before_ready":
    sys.exit(1)
if mode == "noise":
    for _ in range(5):
        print("some library banner", flush=True)
        print("", flush=True)
print("READY", flush=True)
for raw in sys.stdin:
    if mode == "hang":
        time.sleep(60)
    if mode == "die_on_call":
        sys.exit(1)
    if mode == "ocr_error":
        resp = {"ok": False, "error": "boom"}
    else:
        resp = {"ok": True, "text": "hi", "lines": [
            {"index": 0, "text": "hi", "confidence": float(os.environ.get("FAKE_CONF", "0.99")),
             "box": [[0, 0], [1, 0], [1, 1], [0, 1]]}]}
    print("OCR " + json.dumps(resp), flush=True)
