"""Human-readable summary of logs/runs.jsonl. Debug/reporting only -- not
part of the agent-facing loop; nothing under agent/ imports this."""
from __future__ import annotations

from pathlib import Path

from agent.records import RunLog, Status


def summarize(run_log_path: Path) -> str:
    records = RunLog(run_log_path).read_all()
    if not records:
        return "no iterations recorded yet"

    lines = [f"{len(records)} iterations recorded"]
    for status in Status:
        n = sum(1 for r in records if r.status == status)
        if n:
            lines.append(f"  {status.value}: {n}")

    scored = [r for r in records if r.aggregate is not None]
    if scored:
        best = max(scored, key=lambda r: r.aggregate.primary_mean)
        lines.append(
            f"best validation primary: {best.aggregate.primary_mean:.4f} "
            f"(iteration {best.iteration}: {best.hypothesis})"
        )
    return "\n".join(lines)
