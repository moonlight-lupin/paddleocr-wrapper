# GPU path test report

**Verdict: GPU path PASS.** Run on 2026-11-12 following `docs/GPU_TEST_BRIEF.md`. Test only: no service code was changed.

## Environment
| Item | Value |
|---|---|
| GPU / driver | NVIDIA GeForce RTX 4070 Ti (12 GiB), driver 617.42, CUDA UMD 13.4, WDDM |
| OS | Windows 11 Pro 10.0.26300 |
| Python | 3.12.11 (uv venv `.venv-gpu`) |
| paddle / paddleocr / paddlex | `paddlepaddle-gpu` 3.3.1 (cu126), 3.7.0, 3.7.2 |
| cuDNN resolved | `nvidia-cudnn-cu12` 9.5.1.17 (README says 3.3.1 expects 9.9; not upgraded) |
| Other | numpy 2.3.5, opencv-python-headless 5.0.0.93, uvicorn 0.54.0 |
| Load | Desktop GPU already at ~50 % utilisation and ~3.6 GiB used by other apps during the run |

`paddle.device.is_compiled_with_cuda()` = True, `device_count()` = 1. `pytest`: 34 passed (7.8 s).

## GPU path (`--device auto --no-rapid`)
| # | Check | Actual | Result |
|---|---|---|---|
| 1 | `/health` | `status: ok`, `device: gpu:0`, `fallback_active: false`, `paddle_compiled_with_cuda: true`, `engine_chain: [paddle-gpu, paddle-cpu]` | Pass |
| 2 | OCR result | `device: gpu:0`; 3 lines read back exactly; 4-point boxes; confidence 0.9999 / 0.9997 / 0.9891 | Pass |
| 3 | Speed (3 runs) | 0.129 s, 0.088 s, 0.082 s (first call was already warm: model prewarmed at boot) | Pass (< 0.5 s) |
| 4 | GPU really used | Total VRAM 3584 -> 4161 MiB (+~0.58 GiB) with service up. `nvidia-smi` shows `[N/A]` per process under WDDM, so the python process could not be seen directly | Partial |
| 5 | Logs | `initialized device=gpu:0 lang=ch det=default rec=default mkldnn=False fallback_active=False`; four `ocr ok ... device=gpu:0`; no errors | Pass |

Models used: `PP-OCRv6_medium_det` / `PP-OCRv6_medium_rec` (cached in `~/.paddlex`).

## CPU comparison (`--device cpu --no-rapid`)
| Run | device | elapsed_s | Text |
|---|---|---|---|
| 1 | cpu | 3.437 | identical to GPU |
| 2 | cpu | 3.413 | identical |
| 3 | cpu | 3.439 | identical |

GPU (~0.085 s warm) is **~40x** faster than CPU (~3.4 s). Passes the brief's 10x bar; below the README's 80-100x claim.

## Degradation
| Scenario | Result |
|---|---|
| `CUDA_VISIBLE_DEVICES=-1`, `--device auto` | Booted on `cpu` (`fallback_active: false`, `init_error: null`), OCR worked: 3.56 s, same text. No crash. |
| `--device gpu:99` | No fallback. `/health` `status: warming`, `device: uninitialized`, `init_error: "gpu:99: (InvalidArgument) Device id must be less than GPU count ... GPU count is: 1"`. `/ocr` -> HTTP 503 `{"detail":{"error":"model not initialized","detail":"gpu:99: ..."}}`. Process stayed up. Log ends `all device candidates failed`. |
| `--conf-gate 2` | `--conf-gate must be in (0, 1], got 2.0`, exit code 1, after ~8.5 s (paddle import runs first). |

Not run: the optional "fill VRAM from another process" test. On a display-attached WDDM GPU it risks freezing the desktop and other apps.

## Observations and proposed changes
None of these block the GPU path. Listed by value; **all are proposals, not applied.**

1. **`engine_chain` in `/health` is static, not actual.** `engine_chain()` (`paddle_service.py`, ~L537) builds the list from `force_tier` and `rapid_worker` only. With `--device cpu` or `CUDA_VISIBLE_DEVICES=-1` it still reports `["paddle-gpu", "paddle-cpu"]`, which suggests a GPU tier exists when none is active.
   *Proposal:* drop `paddle-gpu` from the list when `state["device"]` is not a `gpu:*` device (keep the list as the configured chain, or add a separate `active_engine` field).
2. **`--conf-gate` validated after heavy imports.** `import paddle` / `paddleocr` (~8 s) run at module top, so a bad flag takes ~8.5 s to fail. The brief expects "exits immediately".
   *Proposal:* validate with an argparse `type=` function (or move the range check before heavy imports) so it fails at parse time. Note the check at ~L953 already runs before model init, so only the import cost remains; deferring the paddle imports until after `parse_args()` would fix it fully.
3. **Brief/README VRAM expectation does not match WDDM.** README says GPU runtime costs ~1 GiB; measured total delta here was ~0.58 GiB, and per-process figures are unavailable on Windows WDDM.
   *Proposal:* in the brief (check 4), say to compare total `memory.used` before/after on Windows, and relax to "roughly 0.5-1 GiB".
4. **README cuDNN note may be stale.** The resolved cuDNN was 9.5.1.17, not the 9.9 the README says the wheel expects, and everything worked, including prewarm and inference.
   *Proposal:* soften to "if you hit cuDNN load errors, install `nvidia-cudnn-cu12==9.9.0.52`", or confirm and update the stated requirement.
5. **README GPU speedup claim.** "80-100x" vs measured ~40x here (CPU 3.4 s). The test image is small and the GPU was shared with a busy desktop, so numbers are not directly comparable.
   *Proposal:* quote the hardware and image used, or say "dozens of times faster".
6. **`gpu:99` leaves `/health` at `status: warming`.** It is accurate that the model is not initialised, but `warming` implies it will become ready, while a bad explicit device never will. `init_error` is present, so it is diagnosable.
   *Proposal:* report `status: error` (or `failed`) when init has failed and no further candidates remain.
7. **Brief: make `uv`/Python 3.12 explicit.** Hosts with only 3.11/3.13/3.14 need `uv venv --python 3.12`, which the README shows but the brief's prereqs do not spell out.
