"""Client example for a paddleocr-wrapper service (base64 JSON).

Usage:
    python client_example.py path/to/image.png [http://host:8649]
"""
import base64
import json
import sys
import urllib.request


def ocr(path: str, host: str) -> dict:
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    req = urllib.request.Request(
        f"{host}/ocr",
        data=json.dumps({"image_b64": b64, "mime": mime}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    host = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8649"
    d = ocr(sys.argv[1], host)
    print(d["text"])
    print(f"\n[{len(d['lines'])} lines, {d['elapsed_s']}s, device {d['device']}]")
    for ln in d["lines"]:
        if ln["box"]:
            xs = [pt[0] for pt in ln["box"]]
            ys = [pt[1] for pt in ln["box"]]
            print(f"  #{ln['index']} conf={ln['confidence']:.3f} "
                  f"bbox=({min(xs):.0f},{min(ys):.0f})-({max(xs):.0f},{max(ys):.0f}) {ln['text']}")