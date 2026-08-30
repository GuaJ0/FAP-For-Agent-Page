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

import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.agents import CodingAgent, Diff, EvaluatorAgent, Idea, ResearchAgent
from agent.config import BOOTSTRAP_ITERATION, Config, DEFAULT_CONFIG
# _parse_ts is imported rather than re-implemented on purpose: summary.json's
# wall_s and convergence.should_stop's wall-clock budget must be the SAME
# number. A local copy could drift and quietly report a different elapsed
# time than the rule that actually stops the run.
from agent.convergence import _parse_ts, should_stop
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
    # Cross-run Do/Don't ledger (agent/research/findings.py). Optional: None
    # means "don't record findings", which is what every existing caller and
    # test gets. run_loop.py supplies the real one.
    findings: Optional[Any] = None

    def __post_init__(self) -> None:
        self.state: OrchestratorState = self.state_store.load()

    def resume_after_human(self) -> None:
        """Clear a tier-2 halt so run() can continue. Does not reset the
        abandonment streak -- that's a judgment call for whoever's driving,
        made by calling this at all.

        Marks manual_intervention_pending so the next RunRecord produced is
        stamped manual_intervention=True -- see _consume_manual_intervention_flag.
        AUDIT-3(a): before this, resuming from a tier-2 halt left no trace in
        runs.jsonl at all; the graded run's "manual intervention count" had no
        way to ever be non-zero from an auto-detected halt/resume.
        """
        self.state.halted = False
        self.state.halt_reason = None
        self.state.manual_intervention_pending = True
        self._save_state()

    def _consume_manual_intervention_flag(self) -> bool:
        """One-shot read-and-clear: True exactly for the first RunRecord built
        after a resume_after_human() call, False for every one after that."""
        pending = self.state.manual_intervention_pending
        self.state.manual_intervention_pending = False
        return pending

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
                in; `config_path` is the config file passed to it. Keeping the
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
            Path(diff.solution_dir), Path(diff.config_path),
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
        # Iteration 0 is a concluded iteration like any other, and bootstrapping
        # runs real seeds -- a kill during the first research iteration should
        # still find a correct summary on disk.
        self._write_summary()
        return record

    def _baseline_record(self, idea: Idea, diff: Diff, seeds, agg, detail: str) -> RunRecord:
        succeeded = agg is not None
        return RunRecord(
            iteration=BOOTSTRAP_ITERATION,
            parent_iteration=None,          # iteration 0 is the root
            timestamp=_iso_now(),
            hypothesis=idea.hypothesis,
            diff_path=self._record_diff_path(diff),
            patch_path=diff.patch_path,
            status=Status.SUCCESS if succeeded else Status.FAILED,
            seeds=seeds,
            aggregate=agg,
            # None, not the absolute score -- see the docstring.
            delta_vs_current_best=None,
            decision=Decision.ACCEPT if succeeded else None,
            events=[Event(type="bootstrap", detail=detail, agent_action="orchestrator")],
            # The baseline is a pre-existing solution, not something an agent
            # wrote, so it has no LLM usage to attribute.
            resources=self._resources(diff, seeds),
            # In the normal call order this is always False -- bootstrapping
            # happens before run() ever starts, so nothing could have set the
            # flag yet. Wired anyway so the invariant ("next record after a
            # resume gets the flag") holds regardless of call order.
            manual_intervention=self._consume_manual_intervention_flag(),
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
            Path(diff.solution_dir), Path(diff.config_path), iteration, list(range(n_seeds)),
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
        self._write_summary()

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
            diff_path=self._record_diff_path(diff),
            patch_path=diff.patch_path,
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
            )] + self._usage_event(diff),
            resources=self._resources(diff, seeds),
            manual_intervention=self._consume_manual_intervention_flag(),
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
            diff_path=self._record_diff_path(diff),
            patch_path=diff.patch_path,
            status=status,
            seeds=seeds,
            aggregate=agg,
            delta_vs_current_best=delta,
            decision=None,
            events=[Event(type="eval_finished", detail=f"primary={agg.primary_mean:.4f}", agent_action="evaluator")]
                   + self._usage_event(diff),
            resources=self._resources(diff, seeds),
            manual_intervention=self._consume_manual_intervention_flag(),
        )

        verdict = self.evaluator.judge(record, history)
        record.decision = verdict.decision
        if verdict.commentary:
            # The write-back to Research: whatever a real Evaluator reasoned
            # about this result lands in the shared history, not a channel
            # only the orchestrator sees.
            record.events.append(Event(
                type="evaluator_commentary", detail=verdict.commentary, agent_action="evaluator",
            ))
        if verdict.usage is not None:
            # ResourceUsage is frozen -- rebuild it rather than mutate, adding
            # the Evaluator's tokens on top of whatever the CodingAgent already
            # contributed via self._resources(). An LLM-backed Evaluator
            # spends real tokens judging a result; that cost is exactly as
            # real as the CodingAgent's and belongs in the same total.
            r = record.resources
            record.resources = ResourceUsage(
                wall_s=r.wall_s, cpu_hours=r.cpu_hours,
                tokens_in=r.tokens_in + verdict.usage.tokens_in,
                tokens_out=r.tokens_out + verdict.usage.tokens_out,
            )

        self.run_log.append(record)

        if verdict.decision == Decision.ACCEPT:
            self._register_checkpoint(iteration, diff, seeds, agg)

        # AUDIT-EVALUATOR: previously hardcoded False, so an Evaluator that
        # returned ABANDON (a documented, legitimate verdict) was silently
        # treated exactly like REVERT -- consecutive_abandonments never
        # incremented, and tier-2 could never fire from an Evaluator's own
        # judgment that a hypothesis was a dead end, only from tier-1's
        # attempt-cap exhaustion. FakeEvaluatorAgent never exercised this path
        # (it never returns ABANDON), which is why it went unnoticed until a
        # real Evaluator that can made it observable.
        self._record_finding(record)
        self._close_idea(abandoned=(verdict.decision == Decision.ABANDON))

    def _resources(self, diff: Diff, seeds: list[SeedMetrics]) -> ResourceUsage:
        """Build the iteration's ResourceUsage, including LLM tokens and CPU time.

        ResourceUsage has carried tokens_in/tokens_out since the schema was
        written and nothing ever populated them, so every RunRecord reported
        zero tokens no matter what the run actually spent. A CodingAgent that
        tracks usage now reports it on the Diff it returns, and this folds it
        in. `usage=None` (FakeCodingAgent, and any agent that doesn't track)
        yields exactly the wall_s-only ResourceUsage as before.

        cpu_hours sums SeedMetrics.cpu_s -- real measured CPU time from
        executor.py's getrusage deltas, not derived from wall_s -- across
        every seed. gpu_s stays 0.0: this system trains on numpy over CPU
        only, by design, and cpu_hours is what actually reports the compute
        a Feasibility scorer would otherwise find only a bare zero for.

        Called from the failed path as well as the successful one: a failed
        attempt still costs tokens and CPU time, and is often the most
        expensive kind of either, since it is the one that burned repairs.
        """
        wall_s = sum(s.wall_s for s in seeds)
        cpu_hours = sum(s.cpu_s for s in seeds) / 3600.0
        if diff.usage is None:
            return ResourceUsage(wall_s=wall_s, cpu_hours=cpu_hours)
        return ResourceUsage(
            wall_s=wall_s,
            cpu_hours=cpu_hours,
            tokens_in=diff.usage.tokens_in,
            tokens_out=diff.usage.tokens_out,
        )

    @staticmethod
    def _record_diff_path(diff: Diff) -> str:
        """What goes into RunRecord.diff_path.

        Deliberately the CONFIG path, not diff.patch_path, for three reasons:

          1. It keeps runs.jsonl's meaning unchanged. That field has always
             held the config the executor was pointed at; repointing it would
             silently make old and new lines mean different things in an
             append-only log, with no version marker to tell them apart.
          2. It is the only path always present. FakeCodingAgent and the
             bootstrapped baseline produce no patch, so patch_path is None for
             them and the field would be half-empty.
          3. LLMCodingAgent._current_best_source() resolves the current best
             solution by taking the sibling train.py of this path. That works
             precisely because a config lives inside its solution dir.

        The residual mismatch -- a field named diff_path holding a config path
        -- is knowingly left. Fixing it means renaming a field in records.py
        and migrating what already-written JSONL lines mean, which is a bigger
        call than this change. agents.py's Diff no longer has a `diff_path` at
        all, so at least there is exactly one in the codebase with one meaning.
        """
        return diff.config_path

    @staticmethod
    def _usage_event(diff: Diff) -> list[Event]:
        """cost_usd has no field on ResourceUsage (see AgentUsage's docstring),
        so it lands in the audit trail as an event instead -- visible in
        runs.jsonl without changing records.py's schema. Emitted only when
        there was real usage, so records from agents that don't track usage are
        byte-identical to before."""
        u = diff.usage
        if u is None or (u.tokens_in == 0 and u.tokens_out == 0):
            return []
        return [Event(
            type="coding_usage",
            detail=f"tokens_in={u.tokens_in} tokens_out={u.tokens_out} cost_usd={u.cost_usd:.6f}",
            agent_action="coding",
        )]

    def _record_finding(self, record: RunRecord) -> None:
        """Log this Evaluator-judged outcome to the cross-run Do/Don't ledger.

        Only judged outcomes reach here -- _handle_failed_run's technical
        abandons deliberately do not. A crash means the Coding Agent could not
        build the idea, which is no evidence about the idea itself.

        Deterministic and LLM-free: every field is copied out of the RunRecord
        and the Evaluator's own commentary Event. Never fatal -- the record is
        already durably appended, so a ledger problem must not take down a run.
        """
        if self.findings is None:
            return
        try:
            from agent.research.agent import _historical_hypothesis, _historical_hypothesis_id
            from agent.research.findings import build_finding

            direction = _historical_hypothesis_id(record)
            if direction is None:
                # No proposal id -- the seeded baseline, or a hand-supplied
                # hypothesis. Nothing stable to deduplicate a direction on.
                return
            self.findings.record(
                build_finding(record, direction=direction,
                              title=_historical_hypothesis(record))
            )
        except Exception as e:  # noqa: BLE001
            print(f"[orchestrator] could not record research finding: "
                  f"{type(e).__name__}: {e}", flush=True)

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

    def _write_summary(self) -> None:
        """Refresh logs/summary.json: cost, progress and elapsed time so far.

        Written after every record is appended rather than at the end of the
        run, because there is no "end of run" to hook when a process is
        SIGKILLed -- nothing after the kill point executes. Writing eagerly
        means the file is already correct on disk *before* any kill could
        happen, so it is never more than one iteration stale, and a run that is
        hard-killed still has an accurate account of exactly where it was.

        wall_s comes from the persisted RunRecord timestamps, never from
        time.time(): an in-process timer only measures the current process's
        uptime and silently resets to near-zero on a crash-resume. This mirrors
        convergence.should_stop, and deliberately shares its _parse_ts so the
        two can never disagree.

        Cheap by construction -- it only re-reads what has already been
        computed and persisted (the run log, the registry, each agent's own
        usage log). No LLM calls, no recomputation.
        """
        try:
            self._write_summary_unguarded()
        except Exception as e:  # noqa: BLE001
            # Never fatal. The RunRecord is already durably appended by the
            # time this runs, so a summary problem must not take down a
            # multi-hour autonomous run. Reported, not swallowed silently.
            print(f"[orchestrator] could not write summary.json: "
                  f"{type(e).__name__}: {e}", flush=True)

    def _write_summary_unguarded(self) -> None:
        history = self.run_log.read_all()
        if not history:
            return

        # Not every agent tracks usage: OfflineResearchAgent and
        # FakeEvaluatorAgent have no .usage at all, so null rather than zeros --
        # "not measured" and "measured as nothing" are different claims.
        #
        # The duck-test is on a callable totals(), not merely on the attribute
        # existing. `usage` is a plausible attribute name for other things (a
        # CodingAgent holding an AgentUsage value, for instance), and a summary
        # writer must not be able to abort a live iteration over one.
        usage: dict[str, Optional[dict]] = {}
        for name, agent in (
            ("coding", self.coding),
            ("research", self.research),
            ("evaluator", self.evaluator),
        ):
            totals = getattr(getattr(agent, "usage", None), "totals", None)
            usage[name] = totals() if callable(totals) else None

        best = self.registry.best()
        payload = {
            # Same definition convergence.should_stop counts by.
            "iterations_concluded": sum(1 for r in history if r.status != Status.FAILED),
            "wall_s": (
                _parse_ts(history[-1].timestamp) - _parse_ts(history[0].timestamp)
            ).total_seconds(),
            "best_iteration": best.iteration if best else None,
            "best_val_primary": best.val_primary if best else None,
            "usage": usage,
        }

        path = self.cfg.paths.logs_dir / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)  # atomic on POSIX: a crash mid-write can't leave a partial file

    def _save_state(self) -> None:
        self.state_store.save(self.state)
