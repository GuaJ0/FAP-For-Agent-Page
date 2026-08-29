"""Convergence state machine: pure function of RunRecord history.

No wall-clock reads, no I/O -- everything comes from the timestamps and
metrics already in the records passed in, so this is unit-testable with
synthetic histories and nothing running.
"""
from __future__ import annotations

from datetime import datetime, timezone

from agent.config import BOOTSTRAP_ITERATION, ConvergenceConfig, DEFAULT_CONFIG
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

      1. concluded *research* idea count >= max_iterations
      2. wall clock (first record's timestamp -> last record's timestamp) >= max_wall_s
      3. validation `primary` best-so-far hasn't improved by more than epsilon
         over each of the last n_window iterations that produced a score

    "Iteration" means a research idea that reached a conclusion (SUCCESS,
    SUCCESS_AFTER_RETRY, or ABANDONED) -- a FAILED record is a fix attempt
    still in flight on the *same* idea, not a new iteration of the research
    loop, so it doesn't count against max_iterations. Otherwise a
    systematically flaky CodingAgent that never quite exhausts its 3-attempt
    budget could burn the whole iteration cap on a single idea's retries.

    THE BOOTSTRAP BASELINE IS COUNTED BY (3) BUT NOT BY (1)
    ------------------------------------------------------
    That asymmetry is deliberate, and it is the whole point of this docstring
    section. Orchestrator.bootstrap_baseline() writes a real, non-FAILED
    RunRecord at BOOTSTRAP_ITERATION before the research loop starts, so it is
    visible to all three checks and each one wants something different from it.

    (1) max_iterations counts *research attempts*, and the baseline is not one
    -- it is the incumbent the attempts are measured against. The competition
    number is "50 research iterations", so a run configured with 50 must get
    50 real attempts. Counting the baseline would silently make that 49, and
    the alternative (every caller passing 51) is caller-side arithmetic that
    one entrypoint will eventually forget.

    (3) The stalled-progress window counts *scored results*, and the baseline
    is emphatically one: it is the score everything else has to beat, so it
    belongs in the trailing window as the first data point. Excluding it would
    take an extra iteration to notice a run that stalls right at the baseline
    -- which is the exact failure bootstrap_baseline() was built to fix. Do not
    "make this consistent" with (1); they are counting different things.

    (2) Wall clock is left alone too, and for a reason that is easy to get
    backwards: RunRecords are timestamped at *completion*, so history[0] is the
    moment the baseline FINISHED, not when it started. The baseline's own
    runtime is therefore already outside the window. Keeping it as history[0]
    starts the clock when research actually began, so the window covers
    iterations 1..N in full. Skipping it would restart the clock at iteration
    1's completion and silently drop iteration 1's duration from the budget --
    strictly less accurate, and more permissive, than counting it.

    Returns (stop, reason). reason is "" when stop is False.
    """
    if not history:
        return False, ""

    # Research iterations only: the bootstrap baseline is an incumbent, not an
    # attempt. See the docstring -- this is the one check it is excluded from.
    n_iter = sum(
        1 for r in history
        if r.status != Status.FAILED and r.iteration != BOOTSTRAP_ITERATION
    )
    if n_iter >= cfg.max_iterations:
        return True, (
            f"max_iterations reached ({n_iter} concluded research ideas "
            f">= {cfg.max_iterations})"
        )

    # history[0] deliberately includes the bootstrap baseline -- its timestamp
    # is the baseline's completion, i.e. when research began. See the docstring.
    wall_s = (_parse_ts(history[-1].timestamp) - _parse_ts(history[0].timestamp)).total_seconds()
    if wall_s >= cfg.max_wall_s:
        return True, f"wall clock budget exceeded ({wall_s:.0f}s >= {cfg.max_wall_s:.0f}s)"

    # Only records with a valid aggregate carry a comparable validation score;
    # failed/abandoned iterations don't move the "best so far" line. The
    # bootstrap baseline IS included here on purpose -- it is the score
    # everything else has to beat. See the docstring.
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
