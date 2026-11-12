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
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv\Scripts\python.exe paddle_service.py --port 8649 --device auto
```

> Windows/Linux differ only in the venv `bin`/`Scripts` path.
> **Known issue (paddle 3.3.1 CPU):** the CPU build crashes inside `predict()` with
> `(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support` when oneDNN is used.
> `enable_mkldnn=False` (the default in this service) avoids it.

### GPU host (CUDA 12.x)

```bash
uv venv .venv-gpu --python 3.12
# Windows / CPython 3.12: hash-locked, exactly the verified set (recommended)
uv pip install --python .venv-gpu/Scripts/python.exe --require-hashes \
    --index-strategy unsafe-best-match -r requirements-gpu.lock
# Other platforms: resolve from the pinned direct deps (then generate your own lock, see below)
# uv pip install --python .venv-gpu/bin/python --index-strategy unsafe-best-match -r requirements-gpu.txt
.venv-gpu/Scripts/python.exe paddle_service.py --port 8649 --device auto
```

`--index-strategy unsafe-best-match` is required: Paddle's index also lists names such as `paddleocr`
and `numpy`, and uv's default (`first-index`) then refuses to fall back to PyPI for them. Install
into a fresh venv (see "Reproducible installs").

#### Where the GPU wheel comes from

`paddlepaddle-gpu` 3.x is **not on PyPI** (PyPI stops at 2.6.2), and Hugging Face does not host
Python wheels. The only official source is Paddle's own index
(`https://www.paddlepaddle.org.cn/packages/stable/cu126/`), which `requirements-gpu.txt` adds as an
*extra* index. Everything else installs from PyPI. Mitigations, strongest last:

- The exact pin (`==3.3.1`) means PyPI can never supply a different build under that name.
- Generate a hashed lock (below) so every artefact, including the Paddle wheel, is hash-verified.
- For a locked-down host: download the wheel once on a trusted machine, check its hash, and install
  from your own folder or internal index (`uv pip install --no-index --find-links wheelhouse/ ...`).
- If you must avoid Paddle's server entirely, the alternative is the ONNX route: RapidOCR (tier 2)
  with `onnxruntime-gpu`, both on PyPI. It is not wired up for GPU in this repo today (tier 2 runs on CPU).

**Models** are separate from the wheel. Since paddleocr 3.0.2 they download from Hugging Face by
default (Baidu BOS only if `PADDLE_PDX_MODEL_SOURCE` is set to BOS), into `~/.paddlex`. To avoid
network access at runtime, copy a populated `~/.paddlex` to the target host.

The cu126 wheel bundles the CUDA runtime and cuDNN as pip dependencies
(`nvidia-*-cu12`) — no NVIDIA toolkit install needed. The 3.3.1 wheel targets cuDNN 9.9,
but a resolved 9.5 also worked in testing (RTX 4070 Ti, 2026-11). Only if you hit a cuDNN
load error at startup:

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

### Reproducible installs

The `requirements*.txt` files pin the *direct* dependencies to the versions verified on real
hardware, but transitive packages (numpy, opencv, paddlex's tree, ...) still float. To freeze
everything, generate a hashed lock **on the target host** (wheels are platform-specific) and commit it:

```bash
uv pip compile requirements-gpu.txt --python-version 3.12 --index-strategy unsafe-best-match \
    --emit-index-url --generate-hashes -o requirements-gpu.lock
uv pip install --require-hashes --index-strategy unsafe-best-match -r requirements-gpu.lock
```

Both flags are needed with the GPU file (tested with uv 0.11.17): Paddle's index also lists names
such as `paddleocr` and `numpy`, and uv's default `first-index` strategy then refuses to look at PyPI
for them. `--emit-index-url` writes both index URLs into the lock so the install line can find
`paddlepaddle-gpu`. `unsafe-best-match` relaxes uv's dependency-confusion guard, but every artefact is
still pinned by version and sha256, so a different build cannot be substituted.

Install the lock into a **fresh** venv. Syncing it over an environment that already has
`opencv-python-headless` removes that package's files from the shared `cv2/` folder and leaves
`cv2` broken (`no attribute 'imencode'`). The committed `requirements-gpu.lock` was generated for
Windows / CPython 3.12 (`--python-platform windows`); regenerate it on Linux.

Regenerate deliberately when upgrading, then re-run `pytest` and the GPU brief.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status (`ok` / `warming` / `error`), device, versions, `fallback_active`, `fallback_reason`, `engine_chain`, `conf_gate`, `rapid_worker` counters |
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

`box` is the polygon in source-image pixel coordinates.

## CLI (for scripts and agents)

[`ocr_cli.py`](ocr_cli.py) is a stdlib-only client for a running service (no paddle needed on the
calling machine). Point it at the service with `--host` or `PADDLE_OCR_URL`
(default `http://127.0.0.1:8649`).

```bash
python ocr_cli.py receipt.png                  # JSON object on stdout
python ocr_cli.py a.png b.jpg                  # JSON array, one entry per image
python ocr_cli.py a.png --format text          # plain text only
cat scan.png | python ocr_cli.py -             # image from stdin
python ocr_cli.py --health                     # service status JSON
```

Results go to stdout, diagnostics to stderr. A failed image is `{"file": ..., "error": ...}`.
Exit codes: `0` all OK, `1` at least one image failed, `2` service unreachable.

## Request limits

Requests are bounded so one huge upload cannot tie up or crash the service. Over a limit the
service answers **413** with a JSON `detail.error`:

| Flag | Default | Guards |
|---|---|---|
| `--max-upload-mb` | 20 | request body size, checked from `Content-Length` and while streaming (chunked bodies too). Base64 JSON bodies carry ~33% overhead over the raw image |
| `--max-pixels` | 40,000,000 | width x height read from the PNG/JPEG header before decoding (decompression bombs: a tiny file that expands to gigabytes) |

The pixel check covers PNG and JPEG; other formats (WebP, BMP, ...) are only size-limited. OCR runs
one image at a time (`CHAIN_LOCK`), so concurrent large requests queue rather than multiply memory.

## Measured performance (RTX 4070 Ti, PP-OCRv6 medium, 2026-10)

GPU is dozens of times faster than CPU. A later re-run on a shared desktop GPU with a small
three-line image measured ~0.085 s vs ~3.4 s (about 40x); see
[`docs/GPU_TEST_REPORT.md`](docs/GPU_TEST_REPORT.md). The table below is the original, larger-image benchmark.


| Image | CPU | GPU | speedup | identical output |
|---|---|---|---|---|
| English prose, 800x300 | 5.08s | 0.06s | 83x | byte-identical |
| Chinese prose | 3.79s | 0.04s | 89x | byte-identical |
| Receipt-style layout | 6.44s | 0.07s | 97x | byte-identical |

GPU runtime VRAM cost: roughly 0.5-1 GiB. Accuracy: EN prose, simplified Chinese, and
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
  testing. The review record (briefs, per-round reviewer logs, FIXES) is in
  [`docs/review-record/`](docs/review-record/); LAN locations are documented as `<PC-LAN>`.
- On Linux/macOS the tier-2 interpreter defaults to `.venv-rapid/bin/python`
  (`Scripts/python.exe` on Windows); override with `--rapid-worker`.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

Tests stub paddle/rapidocr, so they run on any machine. `scripts/smoke/` holds live smoke
scripts that need a running service and local sample images.

```
paddle_service.py   service (API, GUI, engine chain)
rapid_worker.py     tier-2 child process (runs in .venv-rapid)
ocr_cli.py          CLI client
tests/              pytest suite
scripts/smoke/      live smoke scripts
docs/review-record/ tier-2 review history
```

## License

[MIT](LICENSE)