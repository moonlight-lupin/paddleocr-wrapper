# Review Brief: tier-2 RapidOCR CPU fallback for paddle-ocr-service

You are a read-only reviewer. Do not modify any file. Read the brief, the diff,
both service versions, and the new worker file. Probe for introduced defects.
End with a verdict line: APPROVE / APPROVE WITH FIXES / REVISE, plus findings.

## What changed and why

The OCR service on the Windows PC (<PC-LAN>:8649) currently degrades as:
paddle GPU -> in-place paddle CPU (mkldnn off, measured 6.8-9.5 s/img).
Measured 2026-10-18 on the same PC: RapidOCR 3.10.0 (onnxruntime 1.31.0,
PP-OCRv6 medium ONNX) answers warm in 1.5-2.2 s/img with exact text on all ZH
and receipt variants, but corrupts one digit-bearing lowercase EN line type
(observed conf 0.61-0.64 while correct EN prose lines score >= 0.99 on the
same engine), with instability across near-identical renders.

This change inserts RapidOCR as tier 2 between GPU and paddle CPU, with a
low-confidence gate: any tier-2 line below the gate (default 0.70) re-runs the
whole image on paddle CPU (tier 3) before the response.

Files:
- tier2.diff                 unified diff, original vs candidate
- paddle_service_original.py current live version (ground truth for review)
- paddle_service_candidate.py proposed replacement
- rapid_worker.py            NEW file, tier-2 child process

## Architecture decision

RapidOCR runs in a separate venv as a persistent child process with a
line-JSON protocol (READY, then "OCR {json}" per request). This deliberately
avoids installing onnxruntime/rapidocr/opencv-headless into the live GPU venv
(opencv-contrib-python already owns cv2 there; a second opencv package
collides), and keeps the GPU venv byte-identical to production.

## Verified by live test (2026-10-18, port 8650, service root untouched)

1. --force-tier rapidocr + worker: clean EN fixture -> device rapidocr-cpu,
   2.6 s, 4 lines exact, boxes populated. (Tier-2 pass path.)
2. gen_hard noisy EN render (known tier-2 conf 0.607) -> gated, re-run on
   paddle CPU tier 3, device cpu, 4 lines exact. (Gate path.)
3. tier 3 lazily initialized on first gated/failed request; /health reflects
   uses/failures/gated counts and fallback_reason.
4. zh lowres fixture -> tier-2 pass (ZH is rapidocr-strong).
5. Live service root file was temporarily overwritten by mistake during
   rollout prep and was restored from a verified reconstruction before any
   restart could load it; the running production process was never affected.

## Constraints

- Production response shape must not change: {text, lines:[{index,text,
  confidence, box}], elapsed_s, device}. device gains labels rapidocr-cpu / cpu.
- --device cpu and --device gpu semantics must be unchanged from the original.
- No new dependencies in the GPU venv. The tier-2 child must degrade to tier 3
  on spawn failure, protocol failure, or a timed-out/dead child.
- Startup prewarm/probe behavior for the normal (non-forced) boot must remain.

## Known deltas I want your judgment on

- GPU per-request retry on runtime failure (auto-recovery) vs the original
  one-shot permanent CPU fallback. Intentional; check no pathological loop.
- tier-2 zero-lines result is gated (treated as min_conf 0.0). Intentional.
- receipt "GST 9%" scores 0.7159 on tier 2, only 0.016 above the default gate.
  Consider whether the default 0.70 needs a comment or different default.
- The worker has no per-call timeout: a hung child blocks the request until
  the HTTP client times out. Assess severity given the client (VM) uses a
  120 s urlopen timeout.