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
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.agents import CodingAgent, Diff, EvaluatorAgent, Idea, ResearchAgent
from agent.config import Config, DEFAULT_CONFIG
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


class OrchestratorHalted(RuntimeError):
    """Raised when tier-2 has fired: two consecutive tier-1 abandonments.
    The loop stops and waits for a human; call Orchestrator.resume_after_human()
    to clear the halt and continue."""


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
            # "The" checkpoint for a multi-seed run is the single seed that
            # actually scored best -- primary_mean is an average across
            # seeds and isn't any one checkpoint's number. Convention: a
            # real train.py writes its checkpoint into the same directory as
            # --out result.json, which is what artifact_dir points to.
            ok_seeds = [s for s in seeds if s.failure_kind is None]
            best_seed = max(ok_seeds, key=lambda s: s.primary)
            checkpoint_path = best_seed.artifact_dir or diff.solution_dir
            self.registry.register(iteration, checkpoint_path, agg.primary_mean)

        self._close_idea(abandoned=False)

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
