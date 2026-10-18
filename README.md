# paddleocr-wrapper

A single-file FastAPI wrapper around [PaddleOCR 3.x](https://github.com/PaddlePaddle/PaddleOCR)
that turns a machine with a GPU (or CPU only) into a LAN OCR service.

- **One service, one port.** `paddle_service.py` does everything: HTTP API, browser GUI, logs.
- **CPU and GPU builds.** `--device auto` probes the host: paddle CUDA build → driver-visible GPU →
  a real prewarm inference. Any failure (no GPU, old driver, too little VRAM, incompatible
  runtime) falls back to CPU automatically, and `/health` reports the exact reason.
- **Tiered engine chain with a confidence gate.** `/health` reports the chain
  `[paddle-gpu, rapidocr-cpu, paddle-cpu]`: tier 1 is GPU paddle; on a GPU runtime
  failure tier 2 is a rapidocr CPU child process (its own `.venv-rapid`, ONNX Runtime);
  if any tier-2 returned line lands under the confidence gate (`--conf-gate`, default
  0.70 — a MISSING confidence also gates) the request is re-run on tier 3 paddle CPU
  for accuracy. The GPU model and its per-request retry and recovery stay untouched.
- **Box coordinates.** Every detected line returns its text, confidence, and the 4-point
  polygon in source-image pixels — usable for receipts, invoices, and coordinate extraction.
- **Built-in GUI** at `/`: drag-and-drop upload, canvas preview with box overlay,
  click a line to pin its box, click a box to jump to its line, live log viewer,
  health chips, and an API cheatsheet. Interactive Swagger at `/docs`.
- **Model overrides.** Run any named det/rec release (`--det-model`, `--rec-model`)
  without touching the default packaged models. Newer releases auto-download on first start.

## Quick start

### CPU-only host

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe paddlepaddle paddleocr fastapi "uvicorn[standard]" python-multipart
.venv\Scripts\python.exe paddle_service.py --port 8649 --device auto
```

> Windows/Linux differ only in the venv `bin`/`Scripts` path.
> **Known issue (paddle 3.3.1 CPU):** the CPU build crashes inside `predict()` with
> `(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support` when oneDNN is used.
> `enable_mkldnn=False` (the default in this service) avoids it.

### GPU host (CUDA 12.x)

GPU wheels are published on paddle's own index, not PyPI:

```bash
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu/Scripts/python.exe \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    paddlepaddle-gpu==3.3.1
uv pip install --python .venv-gpu/Scripts/python.exe \
    paddleocr==3.7.0 fastapi "uvicorn[standard]" python-multipart
.venv-gpu/Scripts/python.exe paddle_service.py --port 8649 --device auto
```

The cu126 wheel bundles the CUDA runtime and cuDNN as pip dependencies
(`nvidia-*-cu12`) — no NVIDIA toolkit install needed. Match the bundled cuDNN:
the 3.3.1 wheel expects cuDNN 9.9, so if the resolved wheel is older:

```bash
uv pip install --python .venv-gpu/Scripts/python.exe nvidia-cudnn-cu12==9.9.0.52
```

If paddleocr pulls `paddlepaddle` (CPU) in on top of the GPU wheel, install the
rest of the stack from a pinned freeze that omits it:

```bash
uv pip freeze --python <cpu-venv>/Scripts/python.exe | grep -v paddlepaddle > req.txt
uv pip install --python .venv-gpu/Scripts/python.exe -r req.txt
```

> Note: older `uv` releases (<=0.11.17) have no `--no-install-package` flag.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, device, versions, `fallback_active`, `fallback_reason`, `engine_chain`, `conf_gate`, `rapid_worker` counters |
| `POST /ocr` | JSON body `{"image_b64": "...", "mime": "image/png"}` |
| `POST /ocr/upload` | multipart file upload |
| `GET /` | browser GUI |
| `GET /docs`, `/redoc` | interactive API docs |
| `GET /logs` | recent log lines (ring buffer of 300) |

Result shape (both endpoints):

```json
{
  "text": "merged line text...",
  "lines": [
    {"index": 0, "text": "first line", "confidence": 0.99,
     "box": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]}
  ],
  "elapsed_s": 0.12,
  "device": "gpu:0"
}
```

`box` is the polygon in source-image pixel coordinates. Client example:
see [`client_example.py`](client_example.py).

## Measured performance (RTX 4070 Ti, PP-OCRv6 medium, 2026-10)

| Image | CPU | GPU | speedup | identical output |
|---|---|---|---|---|
| English prose, 800x300 | 5.08s | 0.06s | 83x | byte-identical |
| Chinese prose | 3.79s | 0.04s | 89x | byte-identical |
| Receipt-style layout | 6.44s | 0.07s | 97x | byte-identical |

GPU runtime VRAM cost: ~1 GiB. Accuracy: EN prose, simplified Chinese, and
receipt-style layouts all read back exact, ZH confidence ~0.9999.

## Notes

- The default models are the paddleocr package defaults (PP-OCRv6 medium det+rec
  at paddleocr 3.7.0); they download automatically to `~/.paddlex` on first start.
- GUI boxes render from the returned polygons, which handles skewed/rotated text.
- If a GPU predict dies at runtime (e.g. VRAM got taken), the request retries once
  on CPU instead of failing; the tier-2 rapidocr child spawns on demand for the
  next attempt, with one 80 s call budget and a transport-only circuit breaker.
- After VRAM frees up on a host that fell back to CPU at startup, restart the
  service to return to GPU.
- Tier-2 setup: create `.venv-rapid` beside the script with `rapidocr` (ONNX,
  medium det+rec) + `onnxruntime` + opencv-headless, and put `rapid_worker.py`
  in the same directory. Missing either disables tier 2 with a startup log ("tier
  2 disabled"); `--no-rapid` skips it, `--force-tier` overrides the chain for
  testing. The review record (BRIEF, per-round reviewer logs, FIXES) is in this
  repository; LAN locations are documented as `<PC-LAN>`.

## License

[MIT](LICENSE)