"""Generates docs/run_report.md (Deliverable 3) from the real run logs.

Rerun this after archiving a new run (see logs/interventions.md for the
archive convention) to fold it into the report -- add a RunSpec below for it.
Not part of the agent-facing loop; nothing under agent/ imports this.

Usage: PYTHONPATH=. python3 scripts/generate_run_report.py
"""
from pathlib import Path

from runlog.report import RunSpec, render_full_report

NARRATIVE = """\
Run 1 and Run 2 both used the default deterministic `OfflineResearchAgent`
(no `--live-research`; both runs' `summary.json` shows `"research": null`
usage, i.e. zero LLM cost for the Research role), which selects proposals
from a fixed, ranked backlog. Its duplicate-check only compared a proposal
against the CURRENT run's own history, which starts empty at the top of
every run -- it never consulted the cross-run ledger
(`agent/research/findings.jsonl`). Both runs therefore walked the identical
top of the backlog in the identical order: `OFFLINE-HYBRID-BPR`, then
`OFFLINE-GAUC-WEIGHTED-BPR`, then `OFFLINE-DIN-SHORT-HISTORY` -- and got
materially the same outcomes (revert, accept, revert), because the ledger's
merge logic correctly recognizes a repeat of an already-known variant name
and does not let it manufacture new confidence. Run 1 additionally lost a
`fix_attempts` slot (**not** a stall-window slot -- `convergence.should_stop`
already excludes FAILED records from both the iteration cap and the stall
window, by design) to a SyntaxError on its second idea, caught and repaired
via the existing retry path before this session added an explicit
`python -m py_compile` pre-check for it.

Both runs converged the same way: the competition's stall rule (no
improvement > `eps=0.002` in best validation primary over the trailing
`n_window=3` scored iterations) tripped once the same already-known ideas had
been tried and reverted/marginally accepted. Run 1: 5 records, best 0.6027
(iteration 3, delta +0.0010 over baseline -- inside the window, insufficient
to clear it). Run 2: 4 records, best 0.6023 (iteration 2, delta +0.0006 --
same story).

**What changed for Run 3:** this session traced the repetition to the dedup
gap above and fixed it (`agent/research/offline.py` +
`scripts/run_loop.py`) -- `OfflineResearchAgent` now also skips any backlog
entry already recorded in `findings.jsonl` by an EARLIER run, not just this
run. Run 3 then immediately skipped all three previously-explored
directions, tried the one remaining untried backlog entry (a read-only
randomized-exposure diagnostic with no modeling change, correctly reverted
since it structurally cannot move `primary`), and stopped -- not via the
stall rule, but via `OfflineBacklogExhausted`: every other feasible backlog
entry was already covered by `findings.jsonl` (14 feasible, 14 already
attempted). This is the fix working as intended, and it also reveals the
deterministic backlog's real capacity limit: the pre-run exploration
campaign already covered nearly all of it, so a further deterministic-mode
run has essentially nothing new left to try. Going wider would require
`--live-research` (LLM-backed, not template-backlog-bound) rather than
another deterministic run.
"""

RUNS = [
    RunSpec(
        "Run 1", Path("logs/archive/run_20260901_0934/runs.jsonl"),
        "archived -- converged early at 5 records (stall rule, eps=0.002)",
    ),
    RunSpec(
        "Run 2", Path("logs/archive/run_20260901_1101/runs.jsonl"),
        "archived -- converged early at 4 records (stall rule, eps=0.002)",
    ),
    RunSpec(
        "Run 3", Path("logs/runs.jsonl"),
        "complete -- 2 records, stopped via OfflineBacklogExhausted "
        "(research backlog fully covered by the cross-run ledger)",
    ),
]

if __name__ == "__main__":
    out = render_full_report(RUNS, Path("logs/interventions.md"), convergence_narrative=NARRATIVE)
    Path("docs/run_report.md").write_text(out + "\n")
    print(f"{len(out)} chars, {out.count(chr(10))} lines written to docs/run_report.md")
