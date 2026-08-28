"""Convergence state machine: pure function of RunRecord history.

No wall-clock reads, no I/O -- everything comes from the timestamps and
metrics already in the records passed in, so this is unit-testable with
synthetic histories and nothing running.
"""
from __future__ import annotations

from datetime import datetime, timezone

from agent.config import ConvergenceConfig, DEFAULT_CONFIG
from agent.records import RunRecord, Status


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def should_stop(
    history: list[RunRecord],
    cfg: ConvergenceConfig = DEFAULT_CONFIG.convergence,
) -> tuple[bool, str]:
    """Competition rule: stop at whichever of these fires first.

      1. concluded-idea count >= max_iterations
      2. wall clock (first record's timestamp -> last record's timestamp) >= max_wall_s
      3. validation `primary` best-so-far hasn't improved by more than epsilon
         over each of the last n_window iterations that produced a score

    "Iteration" means a research idea that reached a conclusion (SUCCESS,
    SUCCESS_AFTER_RETRY, or ABANDONED) -- a FAILED record is a fix attempt
    still in flight on the *same* idea, not a new iteration of the research
    loop, so it doesn't count against max_iterations. Otherwise a
    systematically flaky CodingAgent that never quite exhausts its 3-attempt
    budget could burn the whole iteration cap on a single idea's retries.

    Returns (stop, reason). reason is "" when stop is False.
    """
    if not history:
        return False, ""

    n_iter = sum(1 for r in history if r.status != Status.FAILED)
    if n_iter >= cfg.max_iterations:
        return True, f"max_iterations reached ({n_iter} concluded ideas >= {cfg.max_iterations})"

    wall_s = (_parse_ts(history[-1].timestamp) - _parse_ts(history[0].timestamp)).total_seconds()
    if wall_s >= cfg.max_wall_s:
        return True, f"wall clock budget exceeded ({wall_s:.0f}s >= {cfg.max_wall_s:.0f}s)"

    # Only records with a valid aggregate carry a comparable validation score;
    # failed/abandoned iterations don't move the "best so far" line.
    scored = [r for r in history if r.aggregate is not None]
    if len(scored) < cfg.n_window + 1:
        return False, ""

    best_so_far: list[float] = []
    best = float("-inf")
    for r in scored:
        best = max(best, r.aggregate.primary_mean)
        best_so_far.append(best)

    trailing = best_so_far[-(cfg.n_window + 1):]
    deltas = [trailing[i + 1] - trailing[i] for i in range(len(trailing) - 1)]
    if all(d <= cfg.epsilon for d in deltas):
        return True, (
            f"no improvement > {cfg.epsilon} in best validation primary "
            f"over the last {cfg.n_window} scored iterations"
        )

    return False, ""
