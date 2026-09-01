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
