# R4 micro-confirm (final)

The three SHIP verdicts (muse final, codex final, opus final-conditional) were
issued against the R2+R3 revision. After them, commit 47f06d2 applied four
small extra fixes from the opus final residuals. THIS round reviews only that
delta: read r4.diff (the exact code change) plus the updated
paddle_service_candidate.py, and verify against the live facts below.

R4-1: CALL_TIMEOUT_S 100 -> 80. Rationale: a full tier-2 timeout plus a rare
  cold tier-3 load must stay inside the VM client's 120 s budget.
R4-2: Dead child vs timeout now use distinct read sentinels in call():
  `_read` returns None (pump EOF -> DEAD-CHILD, respawn retry within the same
  budget) vs queue.Empty timeout string "" (TIMED-OUT, no respawn).
R4-3: /health reads `p = w and w.proc` into a local before poll(), closing
  the opus-noted race (cosmetic 500 on /health).

Post-fix verification already done (2026-10-18):
- probe_r2.py probes A-E all PASS after R4 (incl. hung-call bounded, 100 noise
  lines + READY accepted, stale-pump queue isolation).
- py_compile clean.
- Live smoke (8650 forced-tier + 8651 auto): GPU tier-1 on a worker thread
  gpu:0 0.543 s; /health served 3x during an 11.2 s tier-3 request; tier-3
  lazy init on a worker thread worked (cpu, 4 lines exact). Both instances
  killed after. (test_threadpool.py in this directory.)

Verdict format: SHIP / DO-NOT-SHIP first line. Do not re-report accepted
trade-offs in FIXES.md. Read-only.