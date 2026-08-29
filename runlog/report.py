"""Human-readable summary of logs/runs.jsonl. Debug/reporting only -- not
part of the agent-facing loop; nothing under agent/ imports this."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent.records import RunLog, RunRecord, Status

INTERVENTIONS_ENTRIES_HEADER = "## Entries"


def _parse_intervention_log_entries(path: Path) -> list[str]:
    """Parses logs/interventions.md's '## Entries' section for '- ...' lines.
    Missing file, or file with no such section, means no entries -- most runs
    never need this file, and that must summarize cleanly, not error."""
    if not path.exists():
        return []
    text = path.read_text()
    if INTERVENTIONS_ENTRIES_HEADER not in text:
        return []
    section = text.split(INTERVENTIONS_ENTRIES_HEADER, 1)[1]
    return [line.strip() for line in section.splitlines() if line.strip().startswith("- ")]


def count_manual_interventions(records: list[RunRecord], interventions_md_path: Path) -> dict:
    """The graded deliverable's manual-intervention count: auto-detected
    tier-2 halt/resumes (RunRecord.manual_intervention, set by
    Orchestrator.resume_after_human()) PLUS entries logged by hand in
    logs/interventions.md for anything the orchestrator has no way to see
    itself. Added, not deduplicated -- see that file's own header."""
    auto_detected = sum(1 for r in records if r.manual_intervention)
    logged = _parse_intervention_log_entries(interventions_md_path)
    return {"auto_detected": auto_detected, "logged": len(logged), "total": auto_detected + len(logged)}


def summarize(run_log_path: Path, interventions_md_path: Optional[Path] = None) -> str:
    records = RunLog(run_log_path).read_all()
    if not records:
        return "no iterations recorded yet"

    lines = [f"{len(records)} iterations recorded"]
    for status in Status:
        n = sum(1 for r in records if r.status == status)
        if n:
            lines.append(f"  {status.value}: {n}")

    # Whether an iteration's code change can actually be inspected.
    # RunRecord.patch_path is None for producers that make no diff -- the
    # bootstrapped baseline (a pre-existing solution, not an edit) and
    # FakeCodingAgent -- so a count below the total is normal, not a fault.
    with_patch = sum(1 for r in records if r.patch_path)
    lines.append(f"code diffs recorded: {with_patch}/{len(records)} iterations")

    interventions_path = interventions_md_path or (Path(run_log_path).parent / "interventions.md")
    iv = count_manual_interventions(records, interventions_path)
    lines.append(
        f"manual interventions: {iv['total']} "
        f"({iv['auto_detected']} auto-detected halt/resume, {iv['logged']} logged in interventions.md)"
    )

    scored = [r for r in records if r.aggregate is not None]
    if scored:
        best = max(scored, key=lambda r: r.aggregate.primary_mean)
        lines.append(
            f"best validation primary: {best.aggregate.primary_mean:.4f} "
            f"(iteration {best.iteration}: {best.hypothesis})"
        )
    return "\n".join(lines)
