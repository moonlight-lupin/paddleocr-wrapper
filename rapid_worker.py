# Tier-2 OCR worker for paddle-ocr-service. Runs RapidOCR PP-OCRv6 medium (onnxruntime).
# Protocol: prints "READY" when loaded; reads one JSON line {"image": "<b64>"} per request;
# replies one line "OCR {json}". Any other stdout line is ignored by the parent.
import base64
import json
import sys
import time

import cv2
import numpy as np
from rapidocr import RapidOCR, ModelType

ocr = RapidOCR(params={"Det.model_type": ModelType.MEDIUM, "Rec.model_type": ModelType.MEDIUM})
print("READY", flush=True)


def boxes_of(r, n):
    b = getattr(r, "boxes", None)
    if b is None:
        b = getattr(r, "dt_polys", None)
    out = []
    if b is not None:
        for arr in list(b)[:n]:
            try:
                out.append([[float(x), float(y)] for x, y in list(arr)])
            except Exception:
                out.append(None)
    while len(out) < n:
        out.append(None)
    return out


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(req["image"]), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            raise ValueError("cannot decode image bytes")
        t0 = time.time()
        r = ocr(img)
        elapsed = time.time() - t0
        texts = list(r.txts) if getattr(r, "txts", None) else []
        scores = [round(float(s), 4) for s in r.scores] if getattr(r, "scores", None) else []
        boxes = boxes_of(r, len(texts))
        lines = [
            {"index": i, "text": str(t), "confidence": scores[i] if i < len(scores) else None, "box": boxes[i]}
            for i, t in enumerate(texts)
        ]
        resp = {"ok": True, "lines": lines, "text": "\n".join(texts), "elapsed_s": round(elapsed, 3)}
    except Exception as exc:  # request-level failure; stay alive for the next line
        resp = {"ok": False, "error": str(exc)}
    print("OCR " + json.dumps(resp, ensure_ascii=True), flush=True)