"""PaddleOCR service — FastAPI wrapper for the Windows PC.

Engine chain (all PP-OCRv6_medium det+rec), serialized by CHAIN_LOCK:
  1. paddle GPU  (paddlepaddle-gpu, ~0.05-0.34s/img). Runtime failures:
     per-request retry (auto-recovery when VRAM frees), then tier 2.
  2. RapidOCR cpu (onnxruntime child process in its own venv, ~1.5-2.2s/img;
     lazy or boot-probed). Reader-thread timeouts (handshake 120s, call 100s);
     a hung child is killed and the request degrades. Two consecutive failed
     fast calls open the tier-2 circuit (reset: service restart).
  3. paddle CPU  (own model slot; mkldnn off, ~7-9.5s/img). Also re-runs when
     tier-2 output has any line below the confidence gate (default 0.70).

Devices:
  --device auto  GPU first (build + driver + real prewarm predict)
  --device cpu   force paddle CPU (chain disabled)
  --device gpu   force GPU, no fallback tiers
  --device gpu:N pin a specific CUDA device
Tier flags:
  --no-rapid            disable tier 2 (GPU -> paddle CPU only)
  --rapid-worker PATH   python.exe of the RapidOCR venv (tier 2)
  --rapid-worker-script PATH  rapid_worker.py (tier 2 child protocol)
  --conf-gate F         tier-2 min line confidence; below -> tier 3 re-run (default 0.70)
  --force-tier T        test/ops override: auto|gpu|paddle-cpu|rapidocr
Models:
  --det-model/--rec-model override packaged defaults (auto-downloaded; e.g. a newer release)
GUI:
  GET /       in-browser UI: upload + box preview + live logs + API cheatsheet
  GET /docs   FastAPI interactive Swagger UI
  GET /logs   recent log lines (ring buffer)
"""
import argparse
import base64
import binascii
import datetime as dt
import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import paddle
import paddleocr
from paddleocr import PaddleOCR

app = FastAPI(title="paddle-ocr-service")
state = {
    "model": None,
    "device": "uninitialized",
    "requested_device": "auto",
    "lang": "ch",
    "det_model": None,
    "rec_model": None,
    "enable_mkldnn": False,
    "fallback_active": False,
    "fallback_reason": None,
    "first_error": None,
    "warm": False,
    "error": None,
    # engine chain (tier 2 = RapidOCR child process)
    # "model" = the boot-time paddle device (GPU on auto, or the forced device).
    # "cpu_model" = tier-3 paddle CPU, own slot so tier 3 never displaces the GPU.
    "force_tier": "auto",
    "conf_gate": 0.70,
    "rapid_worker": None,
    "rapid_worker_py": None,
    "rapid_worker_script": None,
    "rapid": {"uses": 0, "failures": 0, "gated": 0, "failure_requests": 0, "circuit_open": False},
    "gpu_runtime_failures": 0,
    "cpu_model": None,
}

# Serializes engine-chain predict + tier-2 child IPC across request handlers.
# The original also served effectively one predict at a time (sync predict in
# async handlers blocked the loop), so behavior matches the original.
CHAIN_LOCK = threading.Lock()

# --- ring log -------------------------------------------------------------
LOGS = deque(maxlen=300)


def log(msg: str, console: bool = True):
    line = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    LOGS.append(line)
    if console:
        print(line, flush=True)


class _RingHandler(logging.Handler):
    """Capture uvicorn/paddle log records into the ring buffer (no console echo)."""

    def emit(self, record):
        try:
            log(f"[{record.levelname}] {record.getMessage()}", console=False)
        except Exception:
            pass


for _name in ("uvicorn", "uvicorn.access", "uvicorn.error", "paddleocr", "paddlex", "paddleocr.paddleocr"):
    logging.getLogger(_name).addHandler(_RingHandler())


class OcrJson(BaseModel):
    image_b64: str
    mime: str | None = None


def resolve_device(requested: str) -> str:
    """Map CLI arg to a paddle device string."""
    if requested == "cpu":
        return "cpu"
    compiled = paddle.device.is_compiled_with_cuda()
    count = 0
    if compiled:
        try:
            count = paddle.device.cuda.device_count()
        except Exception:
            count = 0
    if requested == "auto":
        return "gpu:0" if compiled and count > 0 else "cpu"
    if requested == "gpu":
        if not compiled:
            raise SystemExit("GPU requested but this paddle build is CPU-only (paddlepaddle, not paddlepaddle-gpu). Reinstall with a GPU wheel.")
        return "gpu:0"
    if requested.startswith("gpu:"):
        if not compiled:
            raise SystemExit(f"GPU requested ({requested}) but this paddle build is CPU-only.")
        return requested
    raise SystemExit(f"Unknown device {requested!r} (use auto|cpu|gpu|gpu:N)")


def get_model():
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail={"error": "model not initialized", "detail": state["error"]})
    return model


def result_fields(res) -> dict:
    """Pull text/score/box fields from a paddleocr 3.x OCRResult, defensively."""
    texts = scores = polys = None
    try:
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        polys = res["rec_polys"]
    except Exception:
        try:
            j = res.json["res"]
            texts, scores, polys = j["rec_texts"], j["rec_scores"], j["rec_polys"]
        except Exception:
            pass
    if texts is None:
        return {"lines": [], "text": ""}
    lines = []
    for i, t in enumerate(texts):
        score = float(scores[i]) if scores is not None and i < len(scores) else None
        poly = polys[i] if polys is not None and i < len(polys) else None
        box = [[float(pt[0]), float(pt[1])] for pt in poly] if poly is not None else None
        lines.append({"index": i, "text": str(t), "confidence": score, "box": box})
    return {"lines": lines, "text": "\n".join(str(t) for t in texts)}


def _run_one(image_bytes: bytes, filename: str | None) -> dict:
    """Shared handler body for both /ocr and /ocr/upload."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"cannot decode image (filename={filename!r})")
    t0 = time.time()
    try:
        with CHAIN_LOCK:
            fields, device_label = ensure_fields(img)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "paddle predict failed", "detail": str(exc)})
    elapsed = time.time() - t0
    log(f"ocr ok {filename or 'base64'} {elapsed:.2f}s {len(fields['lines'])} lines device={device_label}", console=False)
    return {
        "text": fields["text"],
        "lines": fields["lines"],
        "elapsed_s": round(elapsed, 3),
        "device": device_label,
    }


# --- tier 2: RapidOCR child process ----------------------------------------

# Gate calibration (2026-10-18, PP-OCRv6 medium ONNX on the 4070 Ti host):
# corrupted RapidOCR lines arrived at conf 0.61-0.64; the known correct edge
# case ("GST 9%" receipt line) arrives at 0.7159. 0.70 sits between them.
# A false gate only costs a tier-3 re-run (~8 s) with correct output.


def results_to_fields(results) -> dict:
    """Convert a paddleocr predict result list; empty results stay empty (no IndexError)."""
    return result_fields(results[0]) if results else {"lines": [], "text": ""}


def _predict_fields(model, img, device_label: str) -> tuple[dict, str]:
    res = model.predict(img)
    return results_to_fields(res), device_label


class RapidWorker:
    """Persistent rapid_worker.py child process, line-JSON protocol.

    Child prints 'READY' once its engine is loaded, then one 'OCR {json}' line
    per request. Any other stdout line is skipped as child-side noise; one
    overall deadline bounds both the handshake and the call, whatever the
    child emits.
    """

    SPAWN_TIMEOUT_S = 120   # model load + READY handshake (budget-capped per call)
    CALL_TIMEOUT_S = 80     # whole tier-2 attempt budget, sized so the tier-2
                            # timeout plus a cold tier-3 load still fits the
                            # VM client's 120 s urlopen budget

    def __init__(self, python_exe: str, script_path: str):
        self.python_exe = python_exe
        self.script_path = script_path
        self.proc = None
        self.lines = queue.Queue()

    # -- child lifecycle ----------------------------------------------------

    def _kill(self):
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            proc.kill()                      # reap; no zombie accumulation
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        finally:
            # pipes close even if wait timed out (opus R3 finding E)
            for stream in (getattr(proc, "stdin", None), getattr(proc, "stdout", None)):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass

    def _pump(self, stream, out: queue.Queue):
        """Reader thread bound to ONE queue: a stale pump thread can never
        inject into a replacement child's queue (codex R2 finding)."""
        try:
            for line in stream.stdout:
                out.put(line)
            out.put(None)
        except Exception:
            out.put(None)

    def _read(self, timeout_s: float) -> str | None:
        try:
            return self.lines.get(timeout=timeout_s)
        except queue.Empty:
            return ""

    def _spawn(self, timeout_s: float = SPAWN_TIMEOUT_S):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        self.proc = subprocess.Popen(
            [self.python_exe, self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=os.path.dirname(os.path.abspath(self.script_path)) or None,
            env=env,
        )
        q = queue.Queue()
        self.lines = q
        threading.Thread(target=self._pump, args=(self.proc, q), daemon=True).start()
        start = time.time()
        while True:  # ONE deadline bounds the whole handshake, noise included
            remaining = timeout_s - (time.time() - start)
            if remaining <= 0:
                raise RuntimeError("rapid worker handshake timed out")
            line = self._read(remaining)
            if line == "":
                raise RuntimeError("rapid worker handshake read timed out")
            if line is None:  # only a closed pipe means death; blank lines are noise
                raise RuntimeError("rapid worker died before READY")
            if line.strip() == "":
                continue
            if line.strip() == "READY":
                log("rapidocr tier spawned")
                return

    # -- request path -------------------------------------------------------

    def call(self, img: np.ndarray) -> tuple[dict | None, str]:
        """Run OCR on a decoded BGR array. Returns (fields, reason).

        Transport failures kill + respawn the child (one respawn per request).
        A per-image OCR failure ({ok:false}) keeps the healthy child alive and
        only fails this request. None = tier-2 unavailable for this request.
        """
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return None, "image encode failed"
        payload = json.dumps({"image": base64.b64encode(buf.tobytes()).decode("ascii")})

        # ONE budget for the whole call (spawn + both attempts): the request
        # stays inside the VM client's 120 s urlopen budget even when both
        # attempts run (codex R2 finding on the 2 x 100s worst case).
        outer = time.time() + self.CALL_TIMEOUT_S
        for attempt in (1, 2):
            remaining = outer - time.time()
            if remaining <= 0:
                return None, "tier2 call budget exhausted"
            try:
                if self.proc is None or self.proc.poll() is not None:
                    self._kill()
                    self._spawn(timeout_s=min(self.SPAWN_TIMEOUT_S, remaining))
                writer = threading.Thread(
                    target=lambda: (self.proc.stdin.write(payload + "\n"), self.proc.stdin.flush()),
                    daemon=True,
                )
                writer.start()
                while True:
                    remaining = outer - time.time()
                    if remaining <= 0:
                        raise RuntimeError(f"TIMED-OUT: tier2 call budget exhausted ({self.CALL_TIMEOUT_S}s)")
                    line = self._read(remaining)
                    if line is None:  # closed pipe: dead child (respawn retry allowed)
                        raise RuntimeError("DEAD-CHILD: rapid worker died mid-call")
                    if line == "":  # read timed out within budget: hang (no respawn - opus R3 A)
                        raise RuntimeError("TIMED-OUT: rapid worker read timed out")
                    if line.startswith("OCR "):
                        resp = json.loads(line[4:])
                        if resp.get("ok"):
                            return {"lines": resp["lines"], "text": resp["text"]}, "rapidocr-cpu"
                        # per-image failure: child is alive and answered; do not kill it
                        return None, f"rapid worker OCR failed: {resp.get('error')}"
                    # else: noise line, keep reading
            except Exception as exc:
                state["rapid"]["failures"] += 1  # counts attempts, may exceed uses
                log(f"rapidocr tier attempt {attempt} failed: {exc}")
                self._kill()
                if "TIMED-OUT" in str(exc):
                    # a hang or spent budget: do not respawn and wait again (opus R3 A)
                    return None, f"tier2 call failed: {exc}"
                # dead child: allow the one respawn within the remaining budget
        return None, "tier2 call failed"

    def probe(self) -> str | None:
        """Boot-time health check: spawn + handshake, then KEEP the child warm
        on success. Returns None on success, error text otherwise."""
        try:
            self._kill()
            self._spawn()
            return None  # child stays alive and warm for later tier-2 calls
        except Exception as exc:
            self._kill()
            return str(exc)

    def stop(self):
        self._kill()


def rapid_fields(img: np.ndarray) -> tuple[dict | None, str]:
    """Tier 2 attempt. Returns (fields, 'rapidocr-cpu') or (None, reason)."""
    w = state["rapid_worker"]
    if w is None or state["rapid"]["circuit_open"]:
        return None, "tier2 disabled" if w is None else "tier2 circuit open"
    state["rapid"]["uses"] += 1
    fields, why = w.call(img)
    if fields is None:
        # breaker counts TRANSPORT failures only: a per-image OCR error or an
        # image-encode failure is request-specific, the child is healthy
        # (opus R3 finding D)
        request_specific = why.startswith("rapid worker OCR failed") or why.startswith("image encode failed")
        if not request_specific:
            state["rapid"]["failure_requests"] += 1
            if state["rapid"]["failure_requests"] >= 2:
                state["rapid"]["circuit_open"] = True
                log("tier 2 circuit opened: repeated rapid worker transport failures (reset only on service restart)")
        return None, why
    state["rapid"]["failure_requests"] = 0  # a successful call closes the failure streak
    lines = fields["lines"]
    confs = [ln["confidence"] for ln in lines if ln.get("confidence") is not None]
    # Gate takes the minimum over all lines; a line with missing confidence is
    # treated as missing evidence, so the image re-runs on tier 3 (codex finding).
    if len(confs) < len(lines):
        state["rapid"]["gated"] += 1
        log("tier2 output has line(s) with missing confidence; re-running on paddle CPU")
        return None, "tier2 conf gate (missing confidence on a line)"
    min_conf = min(confs) if confs else 0.0  # includes the zero-lines-as-0.0 rule
    if min_conf < state["conf_gate"]:
        state["rapid"]["gated"] += 1
        log(f"tier2 min_conf {min_conf:.3f} < gate {state['conf_gate']}; re-running on paddle CPU")
        return None, f"tier2 conf gate (min_conf {min_conf:.3f})"
    return fields, "rapidocr-cpu"


def tier3_fields(img: np.ndarray, reason: str) -> tuple[dict, str]:
    """Tier 3: paddle CPU in its own model slot; never displaces the GPU model."""
    # explicit CPU boots / forced paddle-cpu reuse the boot model when it IS cpu
    if state["requested_device"] == "cpu" and state["device"] == "cpu" and state["model"] is not None:
        return _predict_fields(state["model"], img, "cpu")
    if state["force_tier"] == "paddle-cpu" and state["device"] == "cpu" and state["model"] is not None:
        return _predict_fields(state["model"], img, "cpu")
    if state["cpu_model"] is None:
        log(f"initializing paddle CPU tier 3 ({reason})")
        model = init_model("cpu")
        if not warm_predict(model):
            raise HTTPException(status_code=500, detail={"error": "paddle predict failed: tier-3 CPU prewarm failed"})
        state["cpu_model"] = model
    state["fallback_active"] = True
    if not state["fallback_reason"]:
        state["fallback_reason"] = reason
    return _predict_fields(state["cpu_model"], img, "cpu")


def ensure_fields(img: np.ndarray) -> tuple[dict, str]:
    """Engine chain. Returns (fields, device_label). Raises HTTPException only if everything fails."""
    ft = state["force_tier"]
    if ft == "rapidocr":
        fields, why = rapid_fields(img)
        if fields is not None:
            return fields, why
        return tier3_fields(img, f"tier2 unavailable ({why})")
    if ft == "paddle-cpu":
        return tier3_fields(img, "forced paddle-cpu")
    if ft == "gpu":
        if state["model"] is None:
            raise HTTPException(status_code=503, detail={"error": "model not initialized", "detail": state["error"]})
        res = state["model"].predict(img)
        return results_to_fields(res), state["device"]
    if state["model"] is None:
        raise HTTPException(status_code=503, detail={"error": "model not initialized", "detail": state["error"]})
    if state["device"] == "cpu":
        # boot-time GPU failure (or --device cpu): serve paddle CPU, but keep
        # tier 2 first in auto mode - the fast tier is the point of auto.
        if state["requested_device"] == "auto":
            fields, _ = rapid_fields(img)
            if fields is not None:
                return fields, "rapidocr-cpu"
        res = state["model"].predict(img)
        return results_to_fields(res), "cpu"
    # tier 1: paddle GPU. Per-request retry keeps auto-recovery: a transient
    # VRAM conflict repairs itself on a later request without a restart. The
    # persistent-failure cost is one fast exception per request; /health
    # tracks it via gpu_runtime_failures.
    try:
        res = state["model"].predict(img)
        if state["fallback_active"]:
            state["fallback_active"] = False
            state["fallback_reason"] = None
            log("GPU recovered: fallback state cleared")
        return results_to_fields(res), state["device"]
    except Exception as exc:
        if state["requested_device"] != "auto":
            # explicit --device gpu: original semantics, no downward chain
            raise HTTPException(status_code=500, detail={"error": "paddle predict failed", "detail": str(exc)}) from exc
        state["gpu_runtime_failures"] += 1
        reason = f"GPU runtime failure ({state['gpu_runtime_failures']}): {exc}"
        state["fallback_active"] = True
        state["fallback_reason"] = reason
        log(f"{reason}; descending engine chain")
        fields, why = rapid_fields(img)
        if fields is not None:
            return fields, "rapidocr-cpu"
        return tier3_fields(img, f"tier 2 unavailable ({why}); {reason}")


@app.get("/health")
def health():
    w = state["rapid_worker"]
    p = w and w.proc  # local: _kill can null w.proc between check and poll (opus R3 fix 4)
    ready = state["model"] is not None or (
        state["force_tier"] == "rapidocr" and w is not None
    )
    return {
        "status": "ok" if ready else "warming",
        "service": "paddle-ocr-service",
        "device": state["device"],
        "device_requested": state["requested_device"],
        "lang": state["lang"],
        "det_model": state["det_model"],
        "rec_model": state["rec_model"],
        "enable_mkldnn": state["enable_mkldnn"],
        "model_initialized": state["model"] is not None,
        "warm": state["warm"],
        "fallback_active": state["fallback_active"],
        "fallback_reason": state["fallback_reason"],
        "init_error": state["error"],
        "paddle_version": paddle.__version__,
        "paddleocr_version": paddleocr.__version__,
        "paddle_compiled_with_cuda": paddle.device.is_compiled_with_cuda(),
        "force_tier": state["force_tier"],
        "engine_chain": engine_chain(),
        "conf_gate": state["conf_gate"],
        "rapid_worker": {
            "enabled": w is not None,
            "python": state["rapid_worker_py"],
            "script": state["rapid_worker_script"],
            "spawned": bool(p is not None and p.poll() is None),
            **state["rapid"],
        },
        "gpu_runtime_failures": state["gpu_runtime_failures"],
    }


def engine_chain() -> list[str]:
    if state["force_tier"] == "rapidocr":
        tiers = ["rapidocr-cpu", "paddle-cpu"]
    elif state["force_tier"] == "paddle-cpu":
        tiers = ["paddle-cpu"]
    elif state["force_tier"] == "gpu":
        tiers = ["paddle-gpu"]
    elif state["rapid_worker"] is not None:
        tiers = ["paddle-gpu", "rapidocr-cpu", "paddle-cpu"]
    else:
        tiers = ["paddle-gpu", "paddle-cpu"]
    return tiers


def check_ready():
    if state["force_tier"] == "rapidocr":
        if state["rapid_worker"] is None:
            raise HTTPException(status_code=503, detail={"error": "rapidocr tier not configured (worker missing)"})
        return
    get_model()


@app.post("/ocr")
async def ocr_json(body: OcrJson):
    check_ready()
    img_bytes = decode_image_bytes(body.image_b64, body.mime)
    # run_in_threadpool: the chain (predict/IPC) never sits on the event loop,
    # so /health and /logs stay responsive during long tier-2/3 waits
    return await run_in_threadpool(_run_one, img_bytes, None)


def decode_image_bytes(raw_b64: str, mime: str | None) -> bytes:
    try:
        raw = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64")
    return raw


@app.post("/ocr/upload")
async def ocr_upload(image: UploadFile = File(...)):
    check_ready()
    raw = await image.read()
    return await run_in_threadpool(_run_one, raw, image.filename)


# --- GUI + logs -----------------------------------------------------------
UI_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaddleOCR service</title>
<style>
:root{--bg:#faf9f7;--panel:#fff;--ink:#1f1d1a;--muted:#6b675f;--line:#e6e2da;
--accent:#8a6d3b;--accent-soft:#f4efe6;--warn:#a1541c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:36px 20px 64px}
h1{font-size:20px;font-weight:600;margin:0 0 2px}
.sub{color:var(--muted);font-size:13px;margin-bottom:22px}
.sub a{color:var(--accent)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}
.chip{font-size:12px;padding:4px 12px;border:1px solid var(--line);
border-radius:999px;background:var(--panel)}
.chip b{font-weight:600}
.chip.warn b{color:var(--warn)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:20px 22px;margin-bottom:18px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.9px;
color:var(--muted);font-weight:600;margin:0 0 12px}
.drop{border:1.5px dashed var(--line);border-radius:10px;padding:36px 16px;
text-align:center;color:var(--muted);cursor:pointer;
transition:border-color .5s cubic-bezier(.25,.46,.45,.94),background .5s cubic-bezier(.25,.46,.45,.94)}
.drop:hover,.drop.on{border-color:var(--accent);background:var(--accent-soft);color:var(--ink)}
.btn{display:inline-block;padding:6px 14px;border:1px solid var(--ink);
border-radius:8px;background:var(--ink);color:#fff;font-size:13px;cursor:pointer}
.btn.ghost{background:transparent;color:var(--ink)}
.result{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}
@media (max-width:760px){.result{grid-template-columns:1fr}}
canvas{width:100%;border:1px solid var(--line);border-radius:8px}
pre{margin:0;font:12.5px/1.5 ui-monospace,Consolas,monospace;
white-space:pre-wrap;word-break:break-word;max-height:360px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;width:180px}
code{font:12.5px ui-monospace,Consolas,monospace;background:var(--accent-soft);
padding:1px 6px;border-radius:5px}
details summary{cursor:pointer;color:var(--muted);font-size:13px;margin-top:12px}
.rowline{padding:5px 6px;border-bottom:1px solid var(--line);font-size:13.5px;cursor:pointer;
transition:background .35s cubic-bezier(.25,.46,.45,.94),border-color .35s cubic-bezier(.25,.46,.45,.94)}
.rowline:hover{background:#f0e8d8}
.rowline.sel{background:#f0e8d8;border-left:2px solid var(--accent);padding-left:4px;box-shadow:inset 2px 0 0 var(--accent)}
.rowline b{color:var(--accent);font-weight:600;margin-right:8px}
.rowline .conf{color:var(--muted);font-size:12px;float:right}
.rowline.low .conf{color:var(--warn)}
.statusbar{min-height:20px;color:var(--muted);font-size:13px;margin-top:10px}
</style></head><body><div class="wrap">
<h1>PaddleOCR service</h1>
<div class="sub">PP-OCR det+rec · Windows PC · LAN only · interactive API docs at
<a href="/docs">/docs</a> (Swagger) and <a href="/redoc">/redoc</a></div>
<div class="chips" id="chips">loading…</div>

<div class="card"><h2>Try it</h2>
<div class="drop" id="drop">Drop an image here, or click to browse<br>
<small>PNG / JPG / WEBP</small>
<input type="file" id="file" accept="image/*" style="display:none"></div>
<div class="statusbar" id="status"></div>
<div class="result hidden" id="result">
<div><canvas id="canvas"></canvas></div>
<div><h2 id="linesTitle">Lines</h2><div id="lines"></div></div>
</div>
<details id="jsonwrap" style="display:none"><summary>Raw JSON response</summary>
<pre id="json"></pre></details>
</div>

<div class="card"><h2>API cheatsheet</h2>
<table>
<tr><th>GET /health</th><td>JSON: status, device, versions, fallback_active, fallback_reason</td></tr>
<tr><th>POST /ocr</th><td>JSON body <code>{"image_b64": "...", "mime": "image/png"}</code> — same result shape, base64 in</td></tr>
<tr><th>POST /ocr/upload</th><td>multipart field <code>image</code> — result: <code>{text, lines:[{index,text,confidence,box}], elapsed_s, device}</code>; <code>box</code> is the 4-point polygon in source-image pixels</td></tr>
<tr><th>GET /docs</th><td>Interactive Swagger — run any endpoint from the browser</td></tr>
<tr><th>GET /logs</th><td>recent log lines (ring buffer, 300)</td></tr>
</table></div>

<div class="card"><h2>Logs</h2>
<div style="display:flex;gap:14px;align-items:center;margin-bottom:10px">
<label style="font-size:13px;color:var(--muted)">
<input type="checkbox" id="livecheck" checked> live (2s)</label>
<button class="btn ghost" id="clearview">clear view</button>
</div>
<pre id="logs" style="max-height:260px">—</pre>
</div>
</div>

<script>
const $=id=>document.getElementById(id);
const DROP_HTML=$('drop').innerHTML;

async function health(){try{
  const d=await (await fetch('/health')).json();
  const chips=[['status',d.status],['device',d.device+(d.fallback_active?' · fallback':'')],
    ['models',(d.det_model||d.rec_model)?`${d.det_model||'default det'} + ${d.rec_model||'default rec'}`:'package default']];
  chips.push(['versions',`paddle ${d.paddle_version} · paddleocr ${d.paddleocr_version} · cuda ${d.paddle_compiled_with_cuda}`]);
  if(d.fallback_reason)chips.push(['reason',d.fallback_reason]);
  $('chips').innerHTML=chips.map(([k,v])=>
    `<span class="chip ${d.fallback_active&&k==='device'?'warn':''}">${k}: <b>${v}</b></span>`).join('');
}catch(e){$('chips').innerHTML='<span class="chip warn">service unreachable</span>';}}
health();setInterval(health,5000);

const drop=$('drop'),file=$('file');
function bindDrop(){
  drop.innerHTML=DROP_HTML;
  drop.onclick=()=>document.getElementById('file').click();
  const nf=document.getElementById('file');
  nf.onchange=()=>{if(nf.files[0])run(nf.files[0]);};
}
bindDrop();
/* drag listeners ride on the drop element itself — attach once */
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('on');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('on');}));
drop.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)run(f);});

async function run(f){
  drop.innerHTML='<span style="opacity:.7">OCR running…</span>';
  $('status').textContent=f.name+' ('+(f.size/1024).toFixed(0)+' KB)';
  const fd=new FormData();fd.append('image',f);
  try{
    const r=await fetch('/ocr/upload',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok){const msg=d.detail&&(d.detail.error||JSON.stringify(d.detail))||r.status;
      $('status').innerHTML='<span style="color:var(--warn)">Failed: '+msg+'</span>';
      $('result').classList.add('hidden');return;}
    $('status').textContent=`${d.lines.length} line${d.lines.length===1?'':'s'} · `+(d.elapsed_s*1000).toFixed(0)+' ms · device '+d.device;
    render(f,d);
  }catch(err){$('status').innerHTML='<span style="color:var(--warn)">Upload failed: '+err+'</span>';}
  finally{bindDrop();}
}

function render(f,d){
  $('jsonwrap').style.display='';
  $('json').textContent=JSON.stringify(d,null,2);
  window.__res=d;window.__sel=null;
  const img=new Image();
  img.onload=()=>{window.__img=img;window.__scale=Math.min(1,760/img.naturalWidth);
    $('canvas').title='click a box to jump to its line';
    drawBoxes(null);};
  img.src=URL.createObjectURL(f);
  /* line rows */
  $('linesTitle').textContent='Lines ('+d.lines.length+')';
  $('lines').innerHTML=d.lines.map((ln,i)=>
    `<div class="rowline ${ln.confidence!=null&&ln.confidence<.9?'low':''}" data-i="${ln.index}">
     <b>${ln.index??i}</b>${(ln.confidence!=null?`<span class="conf">${(ln.confidence*100).toFixed(1)}%</span>`:'')}
     ${ln.text.replace(/</g,'&lt;')}</div>`).join('')
    ||'<div style="color:var(--muted)">No text found.</div>';
  /* copy full text */
  window.__text=d.text;
  $('result').classList.remove('hidden');
}

/* box <-> line interactivity */
function drawBoxes(hi){
  const d=window.__res,img=window.__img;
  if(!d||!img||!window.__scale)return;
  const c=$('canvas'),s=window.__scale;
  c.width=img.naturalWidth*s;c.height=img.naturalHeight*s;
  const ctx=c.getContext('2d');
  ctx.drawImage(img,0,0,c.width,c.height);
  ctx.font=`${Math.max(10,11*s)}px ui-monospace,Consolas,monospace`;
  d.lines.forEach(ln=>{
    if(!ln.box)return;
    const active=(hi!=null&&ln.index===hi)||(window.__sel!=null&&ln.index===window.__sel);
    const p=ln.box.map(([x,y])=>[x*s,y*s]);
    ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);
    for(let i=1;i<4;i++)ctx.lineTo(p[i][0],p[i][1]);
    ctx.closePath();
    if(active){ctx.fillStyle='rgba(138,109,59,.34)';ctx.fill();}
    ctx.lineWidth=Math.max(1,(active?3.2:1.3)*s);
    ctx.strokeStyle=active?'rgba(166,124,46,1)':'rgba(138,109,59,.30)';
    ctx.stroke();
    if(active){ctx.fillStyle='rgba(166,124,46,1)';
      ctx.fillText('#'+ln.index,p[0][0],Math.max(11*s,p[0][1]-4));}
  });
}
function pointBoxDist(px,py,box){
  let inside=false,minD=1e9;
  for(let i=0;i<4;i++){
    const [x1,y1]=box[i],[x2,y2]=box[(i+1)%4];
    const dx=x2-x1,dy=y2-y1;
    const t=Math.max(0,Math.min(1,((px-x1)*dx+(py-y1)*dy)/((dx*dx+dy*dy)||1e-9)));
    const dd=Math.hypot(px-(x1+t*dx),py-(y1+t*dy));
    if(dd<minD)minD=dd;
    if(((y1>py)!==(y2>py))&&px<(x2-x1)*(py-y1)/((y2-y1)||1e-9)+x1)inside=!inside;
  }
  return inside?0:minD;
}
const LINES_=$('lines');
LINES_.addEventListener('mouseover',e=>{const row=e.target.closest('.rowline');if(row)drawBoxes(+row.dataset.i);});
LINES_.addEventListener('mouseout',e=>{if(!e.relatedTarget||!e.relatedTarget.closest('.rowline'))drawBoxes(null);});
LINES_.addEventListener('click',e=>{
  const row=e.target.closest('.rowline');if(!row)return;
  const i=+row.dataset.i;
  window.__sel=(window.__sel===i)?null:i;
  LINES_.querySelectorAll('.rowline').forEach(r=>r.classList.toggle('sel',+r.dataset.i===window.__sel));
  drawBoxes(null);
});
$('canvas').addEventListener('click',e=>{
  const c=$('canvas');
  if(!window.__res||!window.__scale)return;
  const rc=c.getBoundingClientRect();
  const px=(e.clientX-rc.left)*(c.width/rc.width);
  const py=(e.clientY-rc.top)*(c.height/rc.height);
  let best=null,bd=1e9;
  window.__res.lines.forEach(ln=>{if(!ln.box)return;
    const dd=pointBoxDist(px,py,ln.box);if(dd<bd){bd=dd;best=ln.index;}});
  if(best==null||bd>30)return;
  const row=LINES_.querySelector(`.rowline[data-i="${best}"]`);
  if(row){window.__sel=(window.__sel===best)?null:best;
    LINES_.querySelectorAll('.rowline').forEach(r=>r.classList.toggle('sel',+r.dataset.i===window.__sel));
    drawBoxes(null);row.scrollIntoView({block:'nearest',behavior:'smooth'});}
});

/* logs */
let logsTimer=null;
async function refreshLogs(){
  try{const d=await (await fetch('/logs')).json();
      $('logs').textContent=d.lines.join('\\n')||'—';
      $('logs').scrollTop=$('logs').scrollHeight;}catch(e){}
}
function setLive(on){if(logsTimer){clearInterval(logsTimer);logsTimer=null;}if(on)logsTimer=setInterval(refreshLogs,2000);}
$('livecheck').onchange=e=>setLive(e.target.checked);
$('clearview').onclick=()=>{$('logs').textContent='—';};
setLive(true);refreshLogs();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


@app.get("/logs")
def recent_logs():
    return {"count": len(LOGS), "lines": list(LOGS)}


def warm_predict(model) -> bool:
    """Run a tiny predict to prove the device actually works."""
    img = np.full((64, 200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "warmup", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    try:
        model.predict(img)
        return True
    except Exception as exc:
        log(f"prewarm predict failed: {exc}")
        return False


def init_model(dev):
    kwargs = dict(
        lang=state["lang"],
        device=dev,
        enable_mkldnn=state["enable_mkldnn"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    # Optional explicit model overrides. When set, paddleocr auto-downloads
    # the named model on first start. Note: explicit names make lang ignored.
    if state.get("det_model"):
        kwargs["text_detection_model_name"] = state["det_model"]
    if state.get("rec_model"):
        kwargs["text_recognition_model_name"] = state["rec_model"]
    return PaddleOCR(**kwargs)


@app.on_event("startup")
def startup():
    # boot-time tier-2 probe runs in EVERY mode (forced included): a broken
    # rapid venv must surface here, not during the first request. A failed
    # probe disables tier 2 for this boot (health then reports warming/503).
    if state["rapid_worker"] is not None:
        err = state["rapid_worker"].probe()
        if err is None:
            log("tier 2 probe ok (worker spawned and answered READY)")
        else:
            state["rapid_worker"] = None
            log(f"tier 2 disabled: boot probe failed: {err}")
    if state["force_tier"] == "rapidocr":
        # test/ops mode: no paddle init at all; tier 2 only (already probed above)
        log("skipping paddle init: --force-tier rapidocr (tier 2 only)")
        state["warm"] = True
        return
    req = state["requested_device"]
    if state["force_tier"] == "paddle-cpu":
        candidates = ["cpu"]
    elif req == "auto":
        candidates = ["cpu"]
        if paddle.device.is_compiled_with_cuda():
            try:
                if paddle.device.cuda.device_count() > 0:
                    candidates = ["gpu:0", "cpu"]
            except Exception:
                pass
    else:
        candidates = [resolve_device(req)]

    for i, dev in enumerate(candidates):
        try:
            model = init_model(dev)
            if not warm_predict(model):
                raise RuntimeError("prewarm predict failed")
            state["model"] = model
            state["device"] = dev
            state["warm"] = True
            state["fallback_active"] = i > 0
            state["fallback_reason"] = (
                f"GPU on this host not usable ({state['first_error']}), using {dev}" if i > 0 else None
            )
            log(f"initialized device={dev} lang={state['lang']} det={state['det_model'] or 'default'} "
                f"rec={state['rec_model'] or 'default'} mkldnn={state['enable_mkldnn']} "
                f"fallback_active={state['fallback_active']}")
            return
        except SystemExit:
            raise
        except Exception as exc:
            state["error"] = f"{dev}: {exc}"
            if i == 0:
                state["first_error"] = str(exc)
            log(f"init on {dev} failed: {exc}")
    log("all device candidates failed")


@app.on_event("shutdown")
def shutdown():
    w = state["rapid_worker"]
    if w is not None:
        w.stop()
    log("service stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8649)
    parser.add_argument("--device", default="auto",
                        help="auto = GPU first, automatic CPU fallback (init-time and runtime) | cpu | gpu | gpu:N")
    parser.add_argument("--lang", default="ch", help="ch (zh+en) | en | ja | ...")
    parser.add_argument("--det-model", default=None,
                        help="text_detection_model_name override (e.g. PP-OCRv6_server_det or a newer release). "
                             "Downloaded automatically on first start; empty = package default.")
    parser.add_argument("--rec-model", default=None,
                        help="text_recognition_model_name override (e.g. PP-OCRv6_server_rec). "
                             "Downloaded automatically on first start; empty = package default.")
    parser.add_argument("--mkldnn", action="store_true",
                        help="enable oneDNN/MKLDNN on CPU. Default off: paddle 3.3.1 CPU + "
                             "MKLDNN raises '(Unimplemented) ConvertPirAttribute2RuntimeAttribute "
                             "not support' inside predict.")
    parser.add_argument("--no-rapid", action="store_true",
                        help="disable tier 2 (RapidOCR). Chain becomes GPU -> paddle CPU only.")
    parser.add_argument("--rapid-worker", default=None,
                        help="python.exe of the RapidOCR venv for tier 2. "
                             "Default: <service dir>\\.venv-rapid\\Scripts\\python.exe. "
                             "If the path does not exist, tier 2 is disabled with a /health note.")
    parser.add_argument("--rapid-worker-script", default=None,
                        help="rapid_worker.py path. Default: <service dir>\\rapid_worker.py")
    parser.add_argument("--conf-gate", type=float, default=0.70,
                        help="tier-2 min line confidence; any line below triggers a tier-3 "
                             "paddle CPU re-run. Default 0.70.")
    parser.add_argument("--force-tier", choices=["auto", "gpu", "paddle-cpu", "rapidocr"], default="auto",
                        help="test/ops override of the engine chain. rapidocr skips paddle init entirely.")
    args = parser.parse_args()
    state["requested_device"] = args.device
    state["lang"] = args.lang
    state["det_model"] = args.det_model
    state["rec_model"] = args.rec_model
    state["enable_mkldnn"] = bool(args.mkldnn)
    state["force_tier"] = args.force_tier
    state["conf_gate"] = args.conf_gate
    if not (0 < args.conf_gate <= 1):
        raise SystemExit(f"--conf-gate must be in (0, 1], got {args.conf_gate}")

    service_dir = os.path.dirname(os.path.abspath(__file__))
    rapid_py = args.rapid_worker or os.path.join(service_dir, ".venv-rapid", "Scripts", "python.exe")
    rapid_script = args.rapid_worker_script or os.path.join(service_dir, "rapid_worker.py")
    state["rapid_worker_py"] = rapid_py
    state["rapid_worker_script"] = rapid_script
    if args.no_rapid:
        log("tier 2 disabled by --no-rapid")
    elif not os.path.exists(rapid_py):
        log(f"tier 2 disabled: rapid worker python not found at {rapid_py}")
    elif not os.path.exists(rapid_script):
        log(f"tier 2 disabled: rapid worker script not found at {rapid_script}")
    else:
        state["rapid_worker"] = RapidWorker(rapid_py, rapid_script)
        log(f"tier 2 enabled: {rapid_py} conf_gate={args.conf_gate}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")