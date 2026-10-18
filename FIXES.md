# Fix round R1 — confirmation request

Round 1 verdicts: Muse REVISE, Opus REVISE, Codex REVISE. Every finding below
has been addressed. Re-read the updated files (paddle_service_candidate.py,
rapid_worker.py), verify each item RESOLVED / NOT-RESOLVED, and end with
SHIP / DO-NOT-SHIP.

Do not re-report accepted trade-offs (listed at the end). Read-only.

## Findings → fixes

1. `--device gpu`/`gpu:N` must not descend the chain (Muse M1, Opus 1, Codex 2)
   → `ensure_fields` GPU-runtime except-path now re-raises HTTP 500 "paddle
   predict failed" unless `requested_device == "auto"`. The `ft == "gpu"`
   branch also re-raises without tiers. Original semantics restored.
2. Hung child froze the whole service (Muse M2/op, Opus 3, Codex 1)
   → All child reads now go through a reader thread + `queue.Queue` with
   deadlines: handshake 120 s, per-call 100 s. A timeout kills the child
   (`kill` + `wait(5)` + pipe close = no zombies/fd leaks) and the request
   degrades to tier 3. stdin writes moved to a daemon writer thread so a
   hung child cannot block the parent write either.
3. Tier-3 CPU model displaced the GPU model; one gated/failed request
   stranded the service on CPU forever (Muse M3, Opus 2)
   → Tier 3 now keeps its own `state["cpu_model"]` slot and never writes
   `state["model"]`/`state["device"]`. GPU keeps per-request retry on runtime
   failures; recovery clears `fallback_active`/`fallback_reason` and logs
   "GPU recovered". Boot-failure case: in auto mode the CPU-booted instance
   now serves tier 2 FIRST (fast tier) before the CPU model — Codex 3.
4. Empty paddle results raised IndexError (Codex 4)
   → New `results_to_fields()` helper guards `res[0]` on every paddle path.
5. Missing-confidence line passed the gate (Codex 5)
   → The gate now treats any line with `confidence: None` as gate trigger.
6. No request serialization on the shared child (Muse M2)
   → Module-level `CHAIN_LOCK` thread lock wraps `ensure_fields` in
   `_run_one`. Matches the original's implicit serialization (sync predict
   in async handlers blocked the event loop).
7. Handshake required the first line to be READY (Muse M4, Opus 7)
   → Handshake skips up to 100 noise lines until READY, under the timeout.
8. Windows pipe encoding (Opus 4)
   → Worker replies with `ensure_ascii=True` (pure-ASCII lines) AND the child
   is launched with `PYTHONIOENCODING=utf-8`. Both belts on.
9. Per-image OCR failure killed a healthy child (Opus 5)
   → `{"ok": false}` replies now return `(None, reason)` without killing the
   child; only transport failures kill + respawn (still one respawn/request).
10. Spawn-failure storm (Opus 6)
    → Two consecutive failed tier-2 CALLS open the circuit
    (`rapid.circuit_open`, /health); reset only on service restart. Spawn
    failures count into it via the same counter.
11. /health `status: ok` with worker missing under forced rapidocr
    (Muse m1, Opus minor)
    → `ready` now requires worker presence in that mode.
12. /health misreporting around fallback (Opus 4-judgment)
    → GPU success clears the fallback flags; tier-2 serving sets
    `fallback_active` + `fallback_reason`; counts: `failures` = attempts
    (can exceed uses), `failure_requests` = consecutive failed calls,
    `circuit_open` = breaker state. All in /health.
13. Gate calibration + validation (Muse m2, Codex note)
    → Calibration comment (0.61-0.64 corrupted vs 0.7159 correct edge case)
    sits above `rapid_fields`; CLI validates `--conf-gate` in (0, 1].
    Default stays 0.70 (all three reviewers endorsed).

## Re-verification probes R2 (codex confirm residuals), all PASS — see probe_r2.py output.
# Accepted nuance: /health status ok with rapid_worker configured but spawned=False
# is a transient-by-design state (killed child respawns on the next tier-2 call);
# a genuinely unusable worker is None'd by the boot probe -> warming + 503.

# Accepted trade-offs (do not re-report)

- 500 detail string for the chain wrapper stays "paddle predict failed"
  (restored from the original wording; verified no consumer matches on it).
- /health status ok while the worker is configured but unspawned
  (post-kill transient): accepted nuance. The child respawns on the next
  tier-2 call; a genuinely unusable worker is None'd by the boot probe and
  reports warming + 503. Verified by probe_r2.py probes A-E, all PASS.

# R2 (round-2 fixes against the codex confirmation)

R2-1. Two 100 s attempt deadlines could exceed the client's 120 s budget
  → `call()` now uses ONE outer budget (100 s) across spawn + both attempts;
  the spawn timeout is capped to the remaining budget. Probed: hung call
  bounded end-to-end.
R2-2. Handshake off-by-one and per-line deadline reset
  → `_spawn` now uses a single whole-handshake deadline (no fixed noise-line
  count); 100 noise lines + READY accepted; endless noise deadlined.
R2-3. Stale reader thread injected into a replacement queue
  → `_pump(stream, out)` is bound to one queue instance at spawn; a stale
  pump can never write into a new child's queue. Probed.
R2-4. Forced-rapidocr boot skipped the probe
  → the boot probe now runs in every mode before the forced-mode early
  return; a failed probe leaves the worker None → warming + 503.

# R3 (against the opus confirmation A-E; commit 6dd234d) and R4

R3-A. Hang stall bounded + event loop freed
  → timeout exits without respawn; ONE budget covers spawn + attempts;
  /ocr + /ocr/upload run the chain via run_in_threadpool so /health and
  /logs answer during long runs. CALL_TIMEOUT_S lowered 100 → 80 so a
  full tier-2 timeout plus a cold tier-3 load stays inside the client's
  120 s (opus residual 2).
R3-B/R2-3. Queue-isolated reader threads (fixed, probed).
R3-C. Blank lines before READY are noise; only a closed pipe kills.
R3-D. Breaker counts transport failures only (OCR-failed / encode failures
  are request-specific and never trip it).
R3-E. _kill closes pipes in a finally even when wait(5) times out.
R4-1 (opus residual 2). CALL_TIMEOUT_S 100 → 80 so a full tier-2 timeout
  plus a cold tier-3 load stays inside the client's 120 s budget.
  (Listed under R3-A above; renumbered here to match R4_CONFIRM.md.)
R4-2 (opus residual 3). Dead child vs timeout now use distinct sentinels:
  None → DEAD-CHILD (respawn retry allowed within budget);
  "" → TIMED-OUT (no respawn). Probed.
R4-3 (opus residual 4). /health reads w.proc into a local before poll
  (no race-induced 500). Fixed.

## POST-R3/R4 LIVE VERIFICATION (2026-10-18, both instances since killed)

- auto tier-1 predict on a worker thread: gpu:0, 0.543 s — the SHIP
  condition from the opus confirmation.
- /health served 3 times DURING an 11.2 s tier-3 request (chain ran on the
  threadpool; event loop responsive; slowest /health 3.3 s under CPU
  contention from the paddle CPU model load).
- tier-3 lazy init on a worker thread: cpu, 11.2 s, exact 4 lines.
- opus's process note: the earlier 0.354-3.9 s chain tests plus this smoke
  both ran the R3/R4 build (scp after commit 6dd234d and later fixes).
- Gate threshold takes the minimum over all lines, so long receipts gate
  more often; accepted (cost is a correct slower path).
- GPU persistent-failure retry per request is intentional (auto-recovery);
  cost per request is one fast exception, tracked in /health.
- Zero-lines tier-2 output keeps gating (min_conf 0.0 rule).
- Boot-time tier-2 probe adds ~2-4 s to service startup (worker model load).
- `rapid.failures` counts attempts, documented in /health comment + FIXES.md.

## Post-fix live verification (port 8650 / 8651, both instances since killed)

- tier-2 pass: rapidocr-cpu, 1.8-3.9 s, exact text, boxes populated.
- tier-3 gate path: cpu, 10.8 s (includes lazy tier-3 init), exact text.
- Two concurrent requests under CHAIN_LOCK: both answered correctly.
- Tier-1 healthy production path (--force-tier auto): gpu:0, 0.354 s,
  worker boot-probed (spawned + READY, uses=0).
- /health reports engine_chain, rapid_worker incl. circuit_open, counts.