# Brief: verify the GPU path (for a local Claude Code on the GPU host)

**Goal:** confirm that `paddle_service.py` runs OCR on the GPU (`device: gpu:0`), is much faster than
CPU, returns the same text, and degrades cleanly when the GPU is unavailable.
**Constraint:** test only. Do not edit code. Report findings; do not fix them.

## 0. Prereqs
- NVIDIA GPU with a working driver (`nvidia-smi` must work).
- Python 3.12 venv (`uv venv .venv-gpu --python 3.12`; other Python versions may lack wheels) with the GPU stack. Follow "GPU host (CUDA 12.x)" in `README.md`
  (`paddlepaddle-gpu==3.3.1` from the cu126 index, `paddleocr==3.7.0`, `fastapi`, `uvicorn[standard]`,
  `python-multipart`, matching cuDNN).
- Skip tier 2 unless asked: always pass `--no-rapid`.

Quick check that paddle sees the GPU:
```bash
python -c "import paddle; print(paddle.__version__, paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
```
Expect `True` and a count of at least 1. If not, stop and report this output (wrong wheel or driver).

## 1. Unit tests (no GPU needed, 30 s)
```bash
pip install -r requirements-dev.txt && pytest
```
Expect all green.

## 2. Make a test image
```bash
python - <<'PY'
import cv2, numpy as np
img = np.full((300, 800, 3), 255, np.uint8)
for i, t in enumerate(["Invoice 2026-1042", "Total due: 1,284.50 USD", "Thank you for your business"]):
    cv2.putText(img, t, (30, 80 + i * 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
cv2.imwrite("gpu_test.png", img)
PY
```
(Optional: also test one real Chinese or receipt image you have; note the file name.)

## 3. GPU path
```bash
python paddle_service.py --port 8649 --device auto --no-rapid
```
First start may download models to `~/.paddlex`. In another shell:
```bash
python ocr_cli.py --health
python ocr_cli.py gpu_test.png          # run it 3 times; the first call includes warm-up
python ocr_cli.py gpu_test.png --format text
```

| # | Check | Pass |
|---|---|---|
| 1 | `/health` | `status: ok`, `device: gpu:0`, `fallback_active: false`, `paddle_compiled_with_cuda: true`, `engine_chain` starts with `paddle-gpu` |
| 2 | OCR result | `device: gpu:0`; the three lines read back correctly; every line has 4-point `box` and `confidence` above 0.9 |
| 3 | Speed | warm `elapsed_s` under 0.5 s (README measured about 0.05 s on a 4070 Ti) |
| 4 | GPU is really used | total `memory.used` in `nvidia-smi` rises by roughly 0.5-1 GiB once the service is up (on Windows/WDDM per-process VRAM shows `[N/A]`, so compare totals before and after) |
| 5 | Logs | `GET /logs` shows `initialized device=gpu:0 ... fallback_active=False` and no errors |

## 4. Compare against CPU
Stop the service, then:
```bash
python paddle_service.py --port 8649 --device cpu --no-rapid
python ocr_cli.py gpu_test.png          # note device, elapsed_s, text
```
Pass: `device: cpu`, same text as the GPU run, GPU at least 10x faster (typically 40x or more).

## 5. Degradation (each: start service, run `ocr_cli.py gpu_test.png`, record `/health`)
| Scenario | Command | Expected |
|---|---|---|
| No GPU visible | `CUDA_VISIBLE_DEVICES=-1 python paddle_service.py --device auto --no-rapid` (PowerShell: `$env:CUDA_VISIBLE_DEVICES=-1`) | boots on `cpu`, OCR still works, no crash; `/health` `engine_chain` has no `paddle-gpu` |
| Explicit GPU, bad index | `python paddle_service.py --device gpu:99 --no-rapid` | no GPU fallback; `/health` status `error` with `init_error` set; `/ocr` returns 503; process stays up |
| Bad flag | `--conf-gate 2` | exits within a second with `argument --conf-gate: ... must be in (0, 1]` (exit code 2) |

Optional (only if easy): while the service runs on `--device auto`, fill VRAM from another process
and call `/ocr`. Expect either a normal GPU result or a CPU result with `fallback_active: true` and a
`fallback_reason`, and recovery on a later request once VRAM frees. Report what happened.

## 6. Report back
One block, conclusion first:
- **Verdict:** GPU path PASS / FAIL.
- Table of checks 1-5 and sections 4-5 with actual values (device, elapsed_s, text match).
- GPU model, driver version, paddle / paddleocr versions (from `/health`).
- For any failure: exact command, full error, last 30 lines of `GET /logs`.
- Anything surprising, even if it passed.
