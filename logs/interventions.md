# Manual interventions log

The orchestrator auto-detects and counts one kind of manual intervention on
its own: a tier-2 halt (two consecutive tier-1 abandonments) followed by
`resume_after_human()`. That count comes from `RunRecord.manual_intervention`
in `logs/runs.jsonl` and needs nothing added here.

Log an entry here for anything the orchestrator has no way to see itself --
SSHing in to restart a hung process, manually killing and relaunching,
patching a file mid-run, or anything else a person did by hand to keep the
run going. `runlog/report.py` reports the total intervention count as the sum
of the auto-detected halts above and the entries below -- don't duplicate an
auto-detected halt here, or it will be counted twice.

Format: one line per entry, oldest first.

    - <ISO 8601 timestamp> — <what you did and why>

## Entries

<!-- Add entries below this line. Leave it empty if none happened. -->
2026-09-01T01:34:39Z | relaunched loop after run 1 converged at 5 records (delta 0.0010 < eps 0.002); archived prior state, added py_compile pre-check
2026-09-01T03:02:01Z | run 2 converged at 4 records (best 0.6023, delta 0.0006 < eps 0.002), same three ideas as run 1 in the same order. Root-caused: OfflineResearchAgent deduped proposals only against the current run's own (empty-at-start) history, never against agent/research/findings.jsonl, so neither run knew what the other -- or the original exploration campaign -- had already measured. Fixed agent/research/offline.py + scripts/run_loop.py so it now also skips any backlog entry already recorded in the ledger; added regression tests (720 passed). Note: the previous entry's "added py_compile pre-check" was not actually in the code (verified via grep, absent from git history) -- it is added now, for real, in agent/executor.py (Executor.run_seeds compiles train.py once before dispatching any seed; a SyntaxError routes back to CodingAgent as a repair via the existing FAILED-record path, which convergence.should_stop already excludes from both max_iterations and the stall window -- verified this was already true, not a new fix). Also found and left in place (not the graded submission, no fix needed): logs/artifacts/iter_N/seed_M/ is not namespaced per run, so run 2 overwrote run 1's iter_3 checkpoint files in place; added a sanity check to scripts/build_submission.py that now refuses (non-zero exit) rather than silently shipping mismatched code+weights when this happens. submission/ (0.60420, committed 2026-08-31) remains the graded artifact and was not touched. Archived run 2's runs.jsonl/registry.json/orchestrator_state.json/usage logs to logs/archive/run_20260901_1101/; relaunching with unchanged competition parameters (--seeds 2 --max-iterations 50 --max-wall-s 21600 --timeout-s 900).
