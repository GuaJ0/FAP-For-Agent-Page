"""The controller: routes Research -> Coding -> Evaluator, enforces the
tier-1/tier-2 abandonment policy and the time backstop, and checkpoints its
own state so it can resume after a crash mid-idea.

Iteration numbering: every executor run (including a failed attempt still
eligible for a fix retry) gets its own sequence number and its own
RunRecord in logs/runs.jsonl, so the audit trail shows each fix attempt.
convergence.py's max_iterations cap, however, only counts records whose
status != FAILED (i.e. SUCCESS, SUCCESS_AFTER_RETRY, ABANDONED) -- a FAILED
record means "still retrying the same idea," not a new iteration of the
research loop. See convergence.should_stop()'s docstring for why.

Iteration 0 is reserved for the seeded baseline, established by
bootstrap_baseline() before the research loop starts. Research-driven
iterations are numbered from 1. A run that never calls bootstrap_baseline()
simply has no iteration 0 and behaves exactly as it always did.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.agents import CodingAgent, Diff, EvaluatorAgent, Idea, ResearchAgent
from agent.config import BOOTSTRAP_ITERATION, Config, DEFAULT_CONFIG
from agent.convergence import should_stop
from agent.executor import Executor
from agent.records import (
    Decision,
    Event,
    ResourceUsage,
    RunLog,
    RunRecord,
    SeedMetrics,
    Status,
)
from agent.registry import CheckpointRegistry
from agent.state import OrchestratorState, StateStore


# Re-exported from agent.config, which is where it now lives so convergence.py
# can import it too without a cycle (orchestrator imports should_stop FROM
# convergence). Kept importable from here because that is where callers and
# tests already reach for it.
#
# _step() numbers research iterations from state.iteration + 1, and
# state.iteration starts at 0, so the first research iteration is 1 and can
# never collide with this.
__all__ = ["BOOTSTRAP_ITERATION", "BootstrapError", "Orchestrator", "OrchestratorHalted"]


class OrchestratorHalted(RuntimeError):
    """Raised when tier-2 has fired: two consecutive tier-1 abandonments.
    The loop stops and waits for a human; call Orchestrator.resume_after_human()
    to clear the halt and continue."""


class BootstrapError(RuntimeError):
    """bootstrap_baseline() could not establish a usable iteration 0.

    Raised rather than returned: without an incumbent, every downstream
    delta, ACCEPT/REVERT decision and convergence check is computed against
    nothing, which is the exact failure mode bootstrapping exists to prevent.
    A caller that swallowed a soft failure would silently get that broken run
    back."""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Orchestrator:
    research: ResearchAgent
    coding: CodingAgent
    evaluator: EvaluatorAgent
    executor: Executor
    run_log: RunLog
    registry: CheckpointRegistry
    state_store: StateStore
    cfg: Config = DEFAULT_CONFIG

    def __post_init__(self) -> None:
        self.state: OrchestratorState = self.state_store.load()

    def resume_after_human(self) -> None:
        """Clear a tier-2 halt so run() can continue. Does not reset the
        abandonment streak -- that's a judgment call for whoever's driving,
        made by calling this at all."""
        self.state.halted = False
        self.state.halt_reason = None
        self._save_state()

    def run(self) -> list[RunRecord]:
        while True:
            if self.state.halted:
                raise OrchestratorHalted(self.state.halt_reason)
            history = self.run_log.read_all()
            stop, reason = should_stop(history, self.cfg.convergence)
            if stop:
                break
            self._step(history)
        return self.run_log.read_all()

    def bootstrap_baseline(self, idea: Idea, diff: Diff) -> RunRecord:
        """Run a known-good solution as iteration 0, before the research loop.

        WHY THIS EXISTS
        ---------------
        `solution/` holds a real, already-verified baseline, but until it has
        actually run through this loop the loop knows nothing about it:
        runs.jsonl is empty, the registry is empty, and convergence's trailing
        window has no scored iteration to trail. Three separate symptoms
        follow, and they are all the same missing fact:

          - `_handle_successful_run` computes
            `delta = primary_mean - best.val_primary if best else primary_mean`,
            so with an empty registry the first real iteration's "delta" is its
            absolute score. An Evaluator that accepts on `delta > margin` then
            accepts anything, including a result that lost to the baseline.
          - `convergence.should_stop` needs `n_window + 1` scored records
            before it evaluates progress at all, and the baseline was not one
            of them -- so stalling near the baseline took an extra iteration to
            notice.
          - `Idea.parent_iteration` for the first hypothesis resolved to None
            (empty history) even though it conceptually builds on the baseline.

        Rather than patch delta computation, registry.py and convergence.py
        separately, this makes the baseline a genuine iteration 0: a real
        RunRecord in the history, registered through the same
        `_register_checkpoint` an accepted iteration uses. All three symptoms
        then resolve with no change to any of those files.

        It runs the solution for real through `self.executor` rather than
        taking a score on faith. Two reasons: the number stays honest as
        `solution/` changes, and -- more concretely -- if nothing the Coding
        agent tries ever beats the baseline, then `registry.best()` IS the
        baseline, and whatever generates a submission needs a complete artifact
        (result.json, checkpoint.npz) actually sitting at that path, not just a
        number in a JSON file. Running it produces that as a side effect.

        DELIBERATE CHOICES
        ------------------
        *The Evaluator is not consulted.* `EvaluatorAgent.judge` decides whether
        to keep, discard, or give up on an iteration -- all three are incoherent
        for the baseline. There is nothing to compare it against, and
        "reverting" it would leave the run with no incumbent at all, which is
        the state this method exists to escape. Worse, `FakeEvaluatorAgent`
        judges on `delta_vs_current_best`, which is exactly the field that has
        no meaning yet. So the decision is recorded as ACCEPT directly.

        *`delta_vs_current_best` is None, not the absolute score.* There is no
        current best -- establishing one is the point. Writing the raw score
        into a field named "delta" is the original bug, not a workaround for
        it. Nothing reads the field for the baseline: convergence.py keys off
        `aggregate.primary_mean`, and the Evaluator is not called.

        *The baseline counts toward `max_iterations`.* It is a concluded,
        non-FAILED record, so `convergence.should_stop` counts it like any
        other. That trades one research iteration for consistency; excluding it
        would mean special-casing iteration 0 inside convergence.py, which is
        the per-place patching this design avoids.

        IDEMPOTENCE AND CRASH-RESUME
        ----------------------------
        Safe to call more than once and safe across a restart. The run log on
        disk -- not in-memory state -- is the source of truth:

          - a *successful* iteration 0 already in the log -> no-op, return it.
          - any record from a later iteration -> raise, because the loop has
            already produced real history and bootstrapping now would insert a
            baseline underneath results that were computed without one.
          - a *failed* iteration 0 and nothing else -> retry. A failed
            bootstrap registers nothing, so there is no incumbent to
            double-register, and re-running after fixing (say) the data path is
            the obviously right behaviour.

        Callers that never call this are completely unaffected: with no
        iteration 0, everything behaves exactly as it did before.

        Args:
            idea: describes the baseline. `parent_iteration` is ignored --
                iteration 0 is the root and has no parent.
            diff: `solution_dir` is the directory the executor runs `train.py`
                in; `diff_path` is the config file passed to it. Keeping the
                config a sibling of `train.py` matters for anything that later
                resolves a registry entry back to the source that produced it.

        Returns:
            The iteration-0 RunRecord -- freshly created, or the existing one
            on a no-op.

        Raises:
            BootstrapError: the loop has already run, or the baseline failed.
        """
        history = self.run_log.read_all()

        existing = next(
            (r for r in history
             if r.iteration == BOOTSTRAP_ITERATION and r.aggregate is not None),
            None,
        )
        if existing is not None:
            return existing

        later = [r for r in history if r.iteration != BOOTSTRAP_ITERATION]
        if later:
            raise BootstrapError(
                f"cannot bootstrap iteration {BOOTSTRAP_ITERATION}: the run log already holds "
                f"{len(later)} record(s) from iteration(s) "
                f"{sorted({r.iteration for r in later})}. Bootstrapping must happen before the "
                "research loop starts, or the baseline would be inserted underneath results "
                "that were already scored without it."
            )

        if self.state.run_start_time is None:
            self.state.run_start_time = time.time()

        seeds, agg = self.executor.run_seeds(
            Path(diff.solution_dir), Path(diff.diff_path),
            BOOTSTRAP_ITERATION, list(range(self._adaptive_n_seeds())),
        )
        self.state.seed_costs.extend(s.wall_s for s in seeds)

        if agg is None:
            # Record the failure before raising: the audit trail should show
            # why bootstrapping failed, and a FAILED iteration-0 record does
            # not block the retry (see IDEMPOTENCE above).
            self.run_log.append(self._baseline_record(
                idea, diff, seeds, agg=None,
                detail="baseline run produced no usable validation metrics",
            ))
            self._save_state()
            raise BootstrapError(
                "the baseline itself failed to produce validation metrics, so there is no "
                "incumbent to measure anything against. First seed reported: "
                f"{self._feedback_from(seeds)}"
            )

        record = self._baseline_record(
            idea, diff, seeds, agg=agg,
            detail=f"baseline established as iteration {BOOTSTRAP_ITERATION}: "
                   f"primary={agg.primary_mean:.4f} over {agg.n_seeds} seed(s)",
        )
        self.run_log.append(record)
        self._register_checkpoint(BOOTSTRAP_ITERATION, diff, seeds, agg)
        self._save_state()
        return record

    def _baseline_record(self, idea: Idea, diff: Diff, seeds, agg, detail: str) -> RunRecord:
        succeeded = agg is not None
        return RunRecord(
            iteration=BOOTSTRAP_ITERATION,
            parent_iteration=None,          # iteration 0 is the root
            timestamp=_iso_now(),
            hypothesis=idea.hypothesis,
            diff_path=diff.diff_path,
            status=Status.SUCCESS if succeeded else Status.FAILED,
            seeds=seeds,
            aggregate=agg,
            # None, not the absolute score -- see the docstring.
            delta_vs_current_best=None,
            decision=Decision.ACCEPT if succeeded else None,
            events=[Event(type="bootstrap", detail=detail, agent_action="orchestrator")],
            resources=ResourceUsage(wall_s=sum(s.wall_s for s in seeds)),
        )

    def _step(self, history: list[RunRecord]) -> None:
        now = time.time()
        if self.state.run_start_time is None:
            self.state.run_start_time = now

        idea = self.state.get_current_idea()
        if idea is None:
            idea = self.research.propose(history)
            self.state.set_current_idea(idea)
            self.state.fix_attempts = 0
            self.state.idea_start_time = now
            self.state.last_failure_feedback = None
            self._save_state()

        diff = self.coding.implement(idea, self.state.last_failure_feedback)

        n_seeds = self._adaptive_n_seeds()
        iteration = self.state.iteration + 1
        seeds, agg = self.executor.run_seeds(
            Path(diff.solution_dir), Path(diff.diff_path), iteration, list(range(n_seeds)),
        )
        self.state.seed_costs.extend(s.wall_s for s in seeds)

        # Freshly-read time, not the pre-run `now` -- idea_start_time is set
        # once per idea (not per attempt), so this must reflect real
        # wall-clock elapsed across however many retries have happened so far.
        idea_elapsed = time.time() - (self.state.idea_start_time or time.time())

        if agg is None:
            self._handle_failed_run(iteration, idea, diff, seeds, idea_elapsed)
        else:
            self._handle_successful_run(iteration, idea, diff, seeds, agg, history)

        self.state.iteration = iteration
        self._save_state()

    def _handle_failed_run(
        self,
        iteration: int,
        idea: Idea,
        diff: Diff,
        seeds: list[SeedMetrics],
        idea_elapsed: float,
    ) -> None:
        self.state.fix_attempts += 1
        should_abandon = (
            self.state.fix_attempts >= self.cfg.retry.max_fix_attempts
            or idea_elapsed >= self.cfg.retry.idea_time_backstop_s
        )
        record = RunRecord(
            iteration=iteration,
            parent_iteration=idea.parent_iteration,
            timestamp=_iso_now(),
            hypothesis=idea.hypothesis,
            diff_path=diff.diff_path,
            status=Status.ABANDONED if should_abandon else Status.FAILED,
            seeds=seeds,
            aggregate=None,
            delta_vs_current_best=None,
            decision=Decision.ABANDON if should_abandon else None,
            events=[Event(
                type="abandon" if should_abandon else "retry",
                detail=(
                    f"fix_attempts={self.state.fix_attempts} idea_elapsed_s={idea_elapsed:.0f} "
                    f"reason={'attempt_cap' if self.state.fix_attempts >= self.cfg.retry.max_fix_attempts else 'time_backstop' if should_abandon else 'will_retry'}"
                ),
                agent_action="orchestrator",
            )],
            resources=ResourceUsage(wall_s=sum(s.wall_s for s in seeds)),
        )
        self.run_log.append(record)

        if should_abandon:
            self._close_idea(abandoned=True)
        else:
            self.state.last_failure_feedback = self._feedback_from(seeds)

    def _handle_successful_run(
        self,
        iteration: int,
        idea: Idea,
        diff: Diff,
        seeds: list[SeedMetrics],
        agg,
        history: list[RunRecord],
    ) -> None:
        best = self.registry.best()
        delta = agg.primary_mean - best.val_primary if best else agg.primary_mean
        status = Status.SUCCESS_AFTER_RETRY if self.state.fix_attempts > 0 else Status.SUCCESS

        record = RunRecord(
            iteration=iteration,
            parent_iteration=idea.parent_iteration,
            timestamp=_iso_now(),
            hypothesis=idea.hypothesis,
            diff_path=diff.diff_path,
            status=status,
            seeds=seeds,
            aggregate=agg,
            delta_vs_current_best=delta,
            decision=None,
            events=[Event(type="eval_finished", detail=f"primary={agg.primary_mean:.4f}", agent_action="evaluator")],
            resources=ResourceUsage(wall_s=sum(s.wall_s for s in seeds)),
        )
        decision = self.evaluator.judge(record, history)
        record.decision = decision
        self.run_log.append(record)

        if decision == Decision.ACCEPT:
            self._register_checkpoint(iteration, diff, seeds, agg)

        self._close_idea(abandoned=False)

    def _register_checkpoint(self, iteration: int, diff: Diff, seeds: list[SeedMetrics], agg) -> None:
        """Register the accepted iteration's checkpoint.

        "The" checkpoint for a multi-seed run is the single seed that actually
        scored best -- primary_mean is an average across seeds and isn't any
        one checkpoint's number. Convention: a real train.py writes its
        checkpoint into the same directory as --out result.json, which is what
        artifact_dir points to.

        Extracted from _handle_successful_run so bootstrap_baseline() registers
        through this exact code rather than a parallel copy of it.
        """
        ok_seeds = [s for s in seeds if s.failure_kind is None]
        best_seed = max(ok_seeds, key=lambda s: s.primary)
        checkpoint_path = best_seed.artifact_dir or diff.solution_dir
        self.registry.register(iteration, checkpoint_path, agg.primary_mean)

    def _close_idea(self, abandoned: bool) -> None:
        if abandoned:
            self.state.consecutive_abandonments += 1
            if self.state.consecutive_abandonments >= self.cfg.retry.max_consecutive_abandonments:
                self.state.halted = True
                self.state.halt_reason = (
                    f"{self.state.consecutive_abandonments} consecutive tier-1 abandonments "
                    "reached; escalating to human"
                )
        else:
            self.state.consecutive_abandonments = 0

        self.state.set_current_idea(None)
        self.state.fix_attempts = 0
        self.state.idea_start_time = None
        self.state.last_failure_feedback = None

    def _feedback_from(self, seeds: list[SeedMetrics]) -> str:
        failing = [s for s in seeds if s.failure_kind is not None]
        first = failing[0] if failing else seeds[0]
        return f"{first.failure_kind.value if first.failure_kind else 'unknown'}: {first.traceback_tail or ''}"

    def _adaptive_n_seeds(self) -> int:
        """Drop to min_seeds once the observed mean cost per run, projected
        across max_seeds, would blow the remaining wall-clock budget."""
        seeding = self.cfg.seeding
        if not self.state.seed_costs:
            return seeding.max_seeds
        mean_cost = statistics.mean(self.state.seed_costs)
        elapsed = time.time() - (self.state.run_start_time or time.time())
        remaining = self.cfg.convergence.max_wall_s - elapsed
        projected_full = mean_cost * seeding.max_seeds
        if projected_full > max(remaining, 0.0):
            return seeding.min_seeds
        return seeding.max_seeds

    def _save_state(self) -> None:
        self.state_store.save(self.state)
