# Review record

Historical artefacts from the tier-2 (RapidOCR) review rounds. Nothing here is run or imported.

- `BRIEF.md`, `R4_CONFIRM.md` – briefs sent to the reviewers
- `FIXES.md` – findings, fixes and accepted trade-offs per round
- `logs/` – raw reviewer transcripts (codex / muse / opus)
- `diffs/` – `tier2.diff` (original vs candidate), `r4.diff`
- `pre-tier2_paddle_service.py` – the service before tier 2 (diff base)

File names inside these documents (`paddle_service_candidate.py`, `probe_r2.py`, …) refer to the
review-time layout. The shipped code is `paddle_service.py`; the probes live on as `tests/`.
