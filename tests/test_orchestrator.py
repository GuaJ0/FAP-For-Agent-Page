import pytest

from agent.agents import AgentUsage, FakeCodingAgent, Verdict
from agent.orchestrator import OrchestratorHalted
from agent.records import Decision, Status
from conftest import make_orchestrator, make_test_config


def test_tier1_abandons_after_three_failed_fix_attempts(tmp_path):
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "crash", "sleep_s": 0.0}] * 3
    orc = make_orchestrator(tmp_path, cfg, outcomes)

    for _ in range(3):
        orc._step(orc.run_log.read_all())

    history = orc.run_log.read_all()
    assert len(history) == 3
    assert [r.status for r in history] == [Status.FAILED, Status.FAILED, Status.ABANDONED]
    assert history[-1].decision == Decision.ABANDON
    assert orc.state.consecutive_abandonments == 1
    assert orc.state.get_current_idea() is None  # ready to accept a new idea
    assert orc.state.fix_attempts == 0


def test_time_backstop_abandons_regardless_of_attempt_count(tmp_path):
    from agent.config import RetryConfig
    cfg = make_test_config(tmp_path, retry=RetryConfig(max_fix_attempts=100, idea_time_backstop_s=0.0))
    outcomes = [{"mode": "crash", "sleep_s": 0.0}]
    orc = make_orchestrator(tmp_path, cfg, outcomes)

    orc._step(orc.run_log.read_all())

    history = orc.run_log.read_all()
    assert history[-1].status == Status.ABANDONED
    assert orc.state.consecutive_abandonments == 1


def test_tier2_halts_after_two_consecutive_abandonments(tmp_path):
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "crash", "sleep_s": 0.0}] * 6  # 3 attempts x 2 ideas
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=("idea A", "idea B"))

    for _ in range(6):
        orc._step(orc.run_log.read_all())

    assert orc.state.halted is True
    assert orc.state.consecutive_abandonments == 2
    with pytest.raises(OrchestratorHalted):
        orc.run()


def test_registered_checkpoint_points_at_the_best_seeds_artifact_dir(tmp_path):
    from agent.config import SeedingConfig
    from pathlib import Path

    cfg = make_test_config(tmp_path, seeding=SeedingConfig(max_seeds=3, min_seeds=1))
    outcomes = [{"mode": "normal", "sleep_s": 0.0, "mean": 0.6, "std": 0.05}]
    orc = make_orchestrator(tmp_path, cfg, outcomes)

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    ok_seeds = [s for s in record.seeds if s.failure_kind is None]
    best_seed = max(ok_seeds, key=lambda s: s.primary)

    entry = orc.registry.best()
    assert entry is not None
    assert entry.checkpoint_path == best_seed.artifact_dir
    assert Path(entry.checkpoint_path).is_dir()
    assert (Path(entry.checkpoint_path) / "result.json").exists()


def test_max_iterations_counts_concluded_ideas_not_retries(tmp_path):
    from agent.config import ConvergenceConfig

    cfg = make_test_config(tmp_path, convergence=ConvergenceConfig(max_iterations=2))
    # idea A: 2 failures then abandon (3 attempts) -- 1 concluded idea.
    # idea B: 1 success -- 2nd concluded idea, hits the cap.
    outcomes = [
        {"mode": "crash", "sleep_s": 0.0},
        {"mode": "crash", "sleep_s": 0.0},
        {"mode": "crash", "sleep_s": 0.0},
        {"mode": "normal", "sleep_s": 0.0},
    ]
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=("idea A", "idea B"))

    orc.run()  # should stop cleanly via convergence, not raise

    history = orc.run_log.read_all()
    assert len(history) == 4  # all 4 attempts ran, including the 2 retries that don't count as iterations
    concluded = [r for r in history if r.status != Status.FAILED]
    assert len(concluded) == 2
    assert [r.status for r in concluded] == [Status.ABANDONED, Status.SUCCESS]


def test_success_resets_the_abandonment_streak(tmp_path):
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "crash", "sleep_s": 0.0}] * 3 + [{"mode": "normal", "sleep_s": 0.0}]
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=("idea A", "idea B"))

    for _ in range(4):
        orc._step(orc.run_log.read_all())

    history = orc.run_log.read_all()
    assert history[-1].status == Status.SUCCESS
    assert history[-1].decision in (Decision.ACCEPT, Decision.REVERT)
    assert orc.state.consecutive_abandonments == 0


def test_timeout_counts_as_a_fix_attempt_failure(tmp_path):
    from agent.config import ExecutorConfig
    cfg = make_test_config(tmp_path, executor=ExecutorConfig(per_run_timeout_s=0.3))
    outcomes = [{"mode": "timeout"}]
    orc = make_orchestrator(tmp_path, cfg, outcomes)

    orc._step(orc.run_log.read_all())

    history = orc.run_log.read_all()
    assert history[-1].status == Status.FAILED
    assert history[-1].seeds[0].failure_kind.value == "timeout"
    assert orc.state.fix_attempts == 1


def test_resume_after_simulated_crash_preserves_fix_attempts_and_idea(tmp_path):
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "crash", "sleep_s": 0.0}]
    orc1 = make_orchestrator(tmp_path, cfg, outcomes)

    orc1._step(orc1.run_log.read_all())
    assert orc1.state.fix_attempts == 1
    idea_before = orc1.state.get_current_idea()
    assert idea_before is not None

    # Simulate a crash: build a brand-new Orchestrator (fresh in-memory state,
    # fresh CodingAgent) pointed at the same on-disk log/state/registry files,
    # as a restarted process would be.
    orc2 = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    assert orc2.state.fix_attempts == 1
    assert orc2.state.get_current_idea() == idea_before

    orc2._step(orc2.run_log.read_all())
    history = orc2.run_log.read_all()
    assert history[-1].status == Status.SUCCESS_AFTER_RETRY
    assert orc2.state.get_current_idea() is None


def test_adaptive_seeding_drops_to_min_seeds_when_budget_projected_to_blow(tmp_path):
    from agent.config import ConvergenceConfig, SeedingConfig
    import time

    cfg = make_test_config(
        tmp_path,
        convergence=ConvergenceConfig(max_wall_s=10.0),
        seeding=SeedingConfig(max_seeds=3, min_seeds=1),
    )
    orc = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    # No observed cost yet -> use the full seed count.
    assert orc._adaptive_n_seeds() == 3

    # Observed cost per seed is high enough that 3 seeds would blow the
    # remaining 10s wall-clock budget -> drop to min_seeds.
    orc.state.run_start_time = time.time() - 8.0  # 2s of the 10s budget left
    orc.state.seed_costs = [5.0, 5.0]
    assert orc._adaptive_n_seeds() == 1


# ---------------------------------------------------------------------------
# AUDIT-2: the registry pointer must actually track validation-best, not just
# "the last accepted iteration" -- these two coincide in every other test in
# this file, since none of them run more than one accepted iteration. std=0
# in fake_train.py's config makes `primary` land on `mean` exactly (no
# seed-to-seed noise to account for), so each iteration's score is pinned by
# construction rather than approximately achieved.
# ---------------------------------------------------------------------------

def test_registry_advances_to_each_new_best_across_accepted_iterations(tmp_path):
    cfg = make_test_config(tmp_path)
    means = [0.50, 0.55, 0.60, 0.65]
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": m} for m in means]
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=tuple(f"idea {i}" for i in range(4)))

    observed = []
    for _ in range(4):
        orc._step(orc.run_log.read_all())
        best = orc.registry.best()
        observed.append((best.iteration, best.val_primary))

    history = orc.run_log.read_all()
    assert [r.decision for r in history] == [Decision.ACCEPT] * 4, \
        "test is only meaningful if every iteration was actually accepted"

    # The pointer must advance on every single step, landing on that step's
    # own iteration -- not lag behind, and not jump straight to the final one.
    assert observed == [(1, 0.50), (2, 0.55), (3, 0.60), (4, 0.65)], observed


def test_registry_stays_pinned_to_the_earlier_best_after_a_regression(tmp_path):
    cfg = make_test_config(tmp_path)
    means = [0.50, 0.60, 0.55]  # improve, improve, then regress below 0.60
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": m} for m in means]
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=tuple(f"idea {i}" for i in range(3)))

    observed = []
    for _ in range(3):
        orc._step(orc.run_log.read_all())
        best = orc.registry.best()
        observed.append((best.iteration, best.val_primary))

    history = orc.run_log.read_all()
    assert [r.decision for r in history] == [Decision.ACCEPT, Decision.ACCEPT, Decision.REVERT], \
        "test is only meaningful if the third iteration was actually reverted, not accepted"

    # Iteration 3 (0.55) must NOT move the pointer: it stays on iteration 2
    # (0.60) through all three steps, even though a third, worse-scoring
    # iteration exists in the run log right alongside it.
    assert observed == [(1, 0.50), (2, 0.60), (2, 0.60)], observed


# ---------------------------------------------------------------------------
# AUDIT-3(a): a tier-2 halt/resume must leave a trace in runs.jsonl. Before
# this, resume_after_human() cleared the halt silently -- nothing distinguished
# "this iteration ran because a human intervened" from any other iteration.
# ---------------------------------------------------------------------------

def test_manual_intervention_flag_is_set_on_the_record_right_after_a_resume(tmp_path):
    cfg = make_test_config(tmp_path)
    # 6 crashes = 2 ideas x 3 attempts each -> tier-2 halts after the 2nd
    # abandonment. Then 2 more normal outcomes for the post-resume ideas.
    outcomes = (
        [{"mode": "crash", "sleep_s": 0.0}] * 6
        + [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.55}] * 2
    )
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=("A", "B", "C", "D"))

    for _ in range(6):
        orc._step(orc.run_log.read_all())
    assert orc.state.halted is True, "test setup didn't actually reach tier-2"

    orc.resume_after_human()
    assert orc.state.manual_intervention_pending is True

    orc._step(orc.run_log.read_all())   # the record produced right after resume
    orc._step(orc.run_log.read_all())   # the one after that

    history = orc.run_log.read_all()
    assert history[-2].manual_intervention is True, \
        "the first record after resume_after_human() must be flagged"
    assert history[-1].manual_intervention is False, \
        "the flag must be one-shot -- the record after that must NOT be flagged"
    # And nothing before the halt was retroactively flagged.
    assert all(r.manual_intervention is False for r in history[:6])


# ---------------------------------------------------------------------------
# EvaluatorAgent.judge() now returns a Verdict (decision + commentary +
# usage), not a bare Decision -- these three tests cover exactly what that
# unlocked, including a real pre-existing bug: _close_idea's `abandoned` flag
# was hardcoded False in the success path, so an Evaluator-issued ABANDON was
# silently treated like REVERT and never counted toward tier-2. It went
# unnoticed because FakeEvaluatorAgent never returns ABANDON.
# ---------------------------------------------------------------------------

class _ScriptedEvaluator:
    """Returns one scripted Verdict per judge() call, in order."""

    def __init__(self, verdicts):
        self._verdicts = iter(verdicts)

    def judge(self, record, history):
        return next(self._verdicts)


def test_evaluator_issued_abandon_counts_toward_tier2_like_a_tier1_exhaustion(tmp_path):
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.5}] * 2
    evaluator = _ScriptedEvaluator([Verdict(Decision.ABANDON), Verdict(Decision.ABANDON)])
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=("A", "B"), evaluator=evaluator)

    orc._step(orc.run_log.read_all())
    assert orc.state.consecutive_abandonments == 1, \
        "a single Evaluator ABANDON must increment the same counter tier-1 exhaustion does"
    assert orc.state.halted is False

    orc._step(orc.run_log.read_all())
    assert orc.state.consecutive_abandonments == 2
    assert orc.state.halted is True, "two Evaluator-issued ABANDONs in a row must trip tier-2, same as tier-1"

    history = orc.run_log.read_all()
    assert [r.decision for r in history] == [Decision.ABANDON, Decision.ABANDON]


def test_evaluator_accept_still_resets_the_abandonment_streak(tmp_path):
    """The fix must not make ACCEPT/REVERT behave like ABANDON -- only an
    actual ABANDON verdict should feed the tier-2 counter."""
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.5}] * 2
    evaluator = _ScriptedEvaluator([Verdict(Decision.ABANDON), Verdict(Decision.ACCEPT)])
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=("A", "B"), evaluator=evaluator)

    orc._step(orc.run_log.read_all())
    assert orc.state.consecutive_abandonments == 1

    orc._step(orc.run_log.read_all())
    assert orc.state.consecutive_abandonments == 0, "ACCEPT must reset the streak, not extend it"
    assert orc.state.halted is False


def test_evaluator_commentary_is_written_back_as_an_event(tmp_path):
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.5}]
    evaluator = _ScriptedEvaluator([
        Verdict(Decision.REVERT, commentary="This underperforms the baseline by 0.05, likely due to an unweighted sampler."),
    ])
    orc = make_orchestrator(tmp_path, cfg, outcomes, evaluator=evaluator)

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    commentary_events = [e for e in record.events if e.type == "evaluator_commentary"]
    assert len(commentary_events) == 1
    assert commentary_events[0].detail == "This underperforms the baseline by 0.05, likely due to an unweighted sampler."
    assert commentary_events[0].agent_action == "evaluator"


def test_no_commentary_event_when_verdict_carries_none(tmp_path):
    """FakeEvaluatorAgent and any Verdict with commentary="" must not litter
    the events list with an empty evaluator_commentary entry."""
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.5}]
    orc = make_orchestrator(tmp_path, cfg, outcomes)  # default FakeEvaluatorAgent

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert not any(e.type == "evaluator_commentary" for e in record.events)


def test_evaluator_usage_is_folded_into_resources_on_top_of_codings(tmp_path):
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.5}]
    evaluator = _ScriptedEvaluator([
        Verdict(Decision.ACCEPT, usage=AgentUsage(tokens_in=500, tokens_out=120, cost_usd=0.01)),
    ])
    orc = make_orchestrator(tmp_path, cfg, outcomes, evaluator=evaluator)
    # FakeCodingAgent's Diff carries no usage, so the record's tokens should
    # be exactly the Evaluator's -- proving they're additive, not overwriting,
    # requires a case with a non-zero CodingAgent contribution too (covered
    # separately by tests/test_resource_usage.py for the CodingAgent side);
    # here the point is that the Evaluator's own usage reaches resources at all.

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.resources.tokens_in == 500
    assert record.resources.tokens_out == 120


# ---------------------------------------------------------------------------
# logs/summary.json: a cost/progress account that survives a hard kill.
#
# "Summarise when the run finishes" cannot satisfy this -- nothing after a
# SIGKILL executes. The file has to be correct on disk BEFORE any kill, so it
# is rewritten after every appended RunRecord. These tests check it exists and
# is valid after each step (not just at the end), and that wall_s is derived
# from persisted timestamps so it stays correct across a resume.
# ---------------------------------------------------------------------------

def _summary(cfg):
    import json
    return json.loads((cfg.paths.logs_dir / "summary.json").read_text())


def test_summary_is_written_and_valid_after_every_step(tmp_path):
    cfg = make_test_config(tmp_path)
    outcomes = [{"mode": "normal", "sleep_s": 0.0, "mean": 0.6, "std": 0.0}] * 3
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=("a", "b", "c"))

    for expected in (1, 2, 3):
        orc._step(orc.run_log.read_all())
        # Readable mid-run, after every single step -- not only at the end.
        assert _summary(cfg)["iterations_concluded"] == expected


def test_summary_reports_best_iteration_and_score(tmp_path):
    cfg = make_test_config(tmp_path)
    means = [0.50, 0.65, 0.55]
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": m} for m in means]
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=("a", "b", "c"))

    for _ in range(3):
        orc._step(orc.run_log.read_all())

    summary = _summary(cfg)
    assert summary["best_iteration"] == 2                      # 0.65, not the latest
    assert summary["best_val_primary"] == pytest.approx(0.65, abs=1e-6)


def test_summary_best_is_null_before_anything_is_accepted(tmp_path):
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, [{"mode": "crash", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    summary = _summary(cfg)
    assert summary["best_iteration"] is None
    assert summary["best_val_primary"] is None
    assert summary["iterations_concluded"] == 0                # a FAILED retry isn't concluded


def test_summary_is_written_even_when_the_iteration_failed(tmp_path):
    """A failed attempt still costs tokens and time; the account must include
    it, or cost-to-convergence is understated."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, [{"mode": "crash", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    assert (cfg.paths.logs_dir / "summary.json").exists()


def test_summary_wall_s_spans_the_crash_boundary(tmp_path):
    """The point of deriving wall_s from persisted timestamps rather than
    time.time(): an in-process timer restarts at ~0 on resume and would report
    only the second process's uptime.

    Simulates a crash the same way
    test_resume_after_simulated_crash_preserves_fix_attempts_and_idea does --
    a brand-new Orchestrator over the same on-disk paths.
    """
    import json
    from datetime import datetime, timedelta, timezone

    cfg = make_test_config(tmp_path)
    orc1 = make_orchestrator(tmp_path, cfg, [{"mode": "normal", "sleep_s": 0.0}])
    orc1._step(orc1.run_log.read_all())

    # Backdate the first record by an hour: a resumed process cannot know that
    # elapsed time from its own clock, only from the persisted timestamp.
    path = cfg.paths.runs_jsonl
    lines = path.read_text().strip().splitlines()
    first = json.loads(lines[0])
    started = datetime.now(timezone.utc) - timedelta(hours=1)
    first["timestamp"] = started.isoformat()
    path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")

    # Fresh Orchestrator over the same files, as a restarted process would be.
    orc2 = make_orchestrator(tmp_path, cfg, [{"mode": "normal", "sleep_s": 0.0}])
    orc2._step(orc2.run_log.read_all())

    summary = _summary(cfg)
    assert summary["iterations_concluded"] == 2
    assert summary["wall_s"] >= 3600, (
        f"wall_s={summary['wall_s']} -- it measured only the new process's uptime "
        "instead of the span since the first persisted record"
    )


def test_summary_wall_s_matches_the_convergence_budget_calculation(tmp_path):
    """summary.json's elapsed time and the rule that actually stops the run
    must be the same number, or the reported budget is fiction."""
    from agent.convergence import _parse_ts

    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, [{"mode": "normal", "sleep_s": 0.0}] * 2,
                            hypotheses=("a", "b"))
    for _ in range(2):
        orc._step(orc.run_log.read_all())

    history = orc.run_log.read_all()
    expected = (_parse_ts(history[-1].timestamp) - _parse_ts(history[0].timestamp)).total_seconds()

    assert _summary(cfg)["wall_s"] == pytest.approx(expected)


def test_agents_without_a_usage_log_report_null_not_zero(tmp_path):
    """FakeCodingAgent/FakeResearchAgent/FakeEvaluatorAgent track no usage.
    null says "not measured"; 0 would claim "measured, and it was free"."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, [{"mode": "normal", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    usage = _summary(cfg)["usage"]
    assert set(usage) == {"coding", "research", "evaluator"}
    assert all(v is None for v in usage.values())


def test_an_agent_whose_usage_is_not_a_log_does_not_break_the_run(tmp_path):
    """`usage` is a plausible attribute name for other things. Duck-testing on
    the attribute alone would raise here and kill a live run from a purely
    observational write."""
    from agent.agents import AgentUsage, Diff

    class _OddCodingAgent:
        def __init__(self, inner):
            self.inner = inner
            self.usage = AgentUsage(tokens_in=5, tokens_out=7)   # not a UsageLog

        def implement(self, idea, feedback):
            d = self.inner.implement(idea, feedback)
            return Diff(config_path=d.config_path, solution_dir=d.solution_dir)

    cfg = make_test_config(tmp_path)
    inner = FakeCodingAgent(tmp_path / "sols", [{"mode": "normal", "sleep_s": 0.0}])
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=_OddCodingAgent(inner))

    orc._step(orc.run_log.read_all())

    assert orc.run_log.read_all()[-1].status == Status.SUCCESS   # the run survived
    assert _summary(cfg)["usage"]["coding"] is None


def test_a_usage_log_reports_its_totals(tmp_path):
    """Positive control for the two tests above: when an agent does have a real
    usage log, the summary carries its numbers rather than null."""
    from agent.agents import Diff
    from agent.coding.llm import LLMResponse, UsageLog

    class _TrackedCodingAgent:
        def __init__(self, inner, path):
            self.inner = inner
            self.usage = UsageLog(path)
            self.usage.record(LLMResponse("x", "gpt-5", 1200, 800, 0.0095),
                              purpose="generate", idea="i", attempt=0)

        def implement(self, idea, feedback):
            d = self.inner.implement(idea, feedback)
            return Diff(config_path=d.config_path, solution_dir=d.solution_dir)

    cfg = make_test_config(tmp_path)
    inner = FakeCodingAgent(tmp_path / "sols", [{"mode": "normal", "sleep_s": 0.0}])
    coding = _TrackedCodingAgent(inner, tmp_path / "coding_usage.jsonl")
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=coding)

    orc._step(orc.run_log.read_all())

    coding_usage = _summary(cfg)["usage"]["coding"]
    assert coding_usage["tokens_in"] == 1200
    assert coding_usage["tokens_out"] == 800
    assert coding_usage["cost_usd"] == pytest.approx(0.0095)


def test_summary_is_written_after_bootstrap_before_any_research_iteration(tmp_path):
    """A kill during the first research iteration should still find a correct
    account of iteration 0 -- which ran real seeds and took real time."""
    import json
    import shutil
    from pathlib import Path

    from agent.agents import Diff, Idea

    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "fake_train.py"
    sol = tmp_path / "baseline"
    sol.mkdir()
    shutil.copy(fixture, sol / "train.py")
    cfg_file = sol / "config.json"
    cfg_file.write_text(json.dumps({"mode": "normal", "sleep_s": 0.0, "mean": 0.6015, "std": 0.0}))

    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[])

    orc.bootstrap_baseline(Idea("baseline", None),
                           Diff(config_path=str(cfg_file), solution_dir=str(sol)))

    summary = _summary(cfg)
    assert summary["iterations_concluded"] == 1
    assert summary["best_iteration"] == 0
    assert summary["best_val_primary"] == pytest.approx(0.6015, abs=1e-3)


def test_summary_write_is_atomic_leaving_no_tmp_file(tmp_path):
    """temp + os.replace, as registry.py and state.py already do: a crash
    mid-write must never leave a partial or corrupt summary.json."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, [{"mode": "normal", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    assert (cfg.paths.logs_dir / "summary.json").exists()
    assert not list(cfg.paths.logs_dir.glob("summary*.tmp"))


# ---------------------------------------------------------------------------
# Cross-run Do/Don't ledger: the Orchestrator logs Evaluator-judged outcomes so
# a later run with an empty runs.jsonl still knows what has been measured.
# The Orchestrator owns the write; the Evaluator's job stays judging.
# ---------------------------------------------------------------------------

def _proposal_hypothesis(hid="RP-001", text="use a pairwise BPR ranking loss"):
    return (
        "[RESEARCH_PROPOSAL v1]\n"
        f"ID: {hid}\n"
        "TITLE: a title\n"
        "PARENT ITERATION: 0\n"
        "\n"
        "HYPOTHESIS:\n"
        f"{text}\n"
    )


def _ledger_orchestrator(tmp_path, cfg, outcomes, hypotheses, evaluator=None):
    from agent.research.findings import FindingsLedger

    ledger = FindingsLedger(tmp_path / "findings.jsonl")
    orc = make_orchestrator(tmp_path, cfg, outcomes, hypotheses=hypotheses,
                            evaluator=evaluator)
    orc.findings = ledger
    return orc, ledger


def test_a_reverted_iteration_is_logged_as_a_dont(tmp_path):
    cfg = make_test_config(tmp_path)
    # mean below the incumbent so FakeEvaluatorAgent reverts it.
    orc, ledger = _ledger_orchestrator(
        tmp_path, cfg,
        [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.60},
         {"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.55}],
        hypotheses=(_proposal_hypothesis("RP-001"), _proposal_hypothesis("RP-002")),
    )

    orc._step(orc.run_log.read_all())      # accepted, becomes the incumbent
    orc._step(orc.run_log.read_all())      # loses to it -> revert

    stored = {f.direction: f for f in ledger.load()}
    assert stored["RP-001"].verdict == "do"
    assert stored["RP-002"].verdict == "dont"
    assert stored["RP-002"].delta_vs_incumbent < 0


def test_the_finding_survives_a_full_reset_of_the_run_state(tmp_path):
    """The point of the feature: runs.jsonl, registry.json and
    orchestrator_state.json all reset, and the ledger still remembers."""
    from agent.research.findings import FindingsLedger

    cfg = make_test_config(tmp_path)
    orc, ledger = _ledger_orchestrator(
        tmp_path, cfg, [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.6}],
        hypotheses=(_proposal_hypothesis("RP-001"),),
    )
    orc._step(orc.run_log.read_all())

    # Wipe everything a "fresh run" resets.
    for p in (cfg.paths.runs_jsonl, cfg.paths.registry_json, cfg.paths.orchestrator_state):
        if p.exists():
            p.unlink()

    assert [f.direction for f in FindingsLedger(tmp_path / "findings.jsonl").load()] == ["RP-001"]


def test_a_technical_abandon_is_not_logged(tmp_path):
    """_handle_failed_run's abandon path has no Evaluator judgment and no
    delta. A crash means the Coding agent couldn't build it -- not evidence
    against the direction."""
    cfg = make_test_config(tmp_path)
    orc, ledger = _ledger_orchestrator(
        tmp_path, cfg, [{"mode": "crash", "sleep_s": 0.0}] * 3,
        hypotheses=(_proposal_hypothesis("RP-009"),),
    )

    for _ in range(3):
        orc._step(orc.run_log.read_all())

    assert orc.run_log.read_all()[-1].status == Status.ABANDONED
    assert ledger.load() == ()


def test_a_record_without_a_proposal_id_is_not_logged(tmp_path):
    """The seeded baseline and hand-written hypotheses have no stable direction
    key to deduplicate on."""
    cfg = make_test_config(tmp_path)
    orc, ledger = _ledger_orchestrator(
        tmp_path, cfg, [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.6}],
        hypotheses=("a plain hypothesis with no proposal envelope",),
    )

    orc._step(orc.run_log.read_all())

    assert orc.run_log.read_all()[-1].decision is not None   # it WAS judged
    assert ledger.load() == ()                                # but not logged


def test_the_evaluators_own_words_become_the_reason(tmp_path):
    """The 'why' is extracted from the Evaluator's commentary event -- no LLM
    call in the write path."""
    from agent.agents import Verdict

    class _CommentingEvaluator:
        def judge(self, record, history):
            return Verdict(decision=Decision.REVERT,
                           commentary="Regressed on validation primary; the loss is misaligned.")

    cfg = make_test_config(tmp_path)
    orc, ledger = _ledger_orchestrator(
        tmp_path, cfg, [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.6}],
        hypotheses=(_proposal_hypothesis("RP-011"),),
        evaluator=_CommentingEvaluator(),
    )

    orc._step(orc.run_log.read_all())

    assert "misaligned" in ledger.load()[0].why


def test_orchestrators_without_a_ledger_are_unaffected(tmp_path):
    """findings defaults to None, so every existing caller and test is
    unchanged and nothing is written anywhere."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, [{"mode": "normal", "sleep_s": 0.0}])

    assert orc.findings is None
    orc._step(orc.run_log.read_all())

    assert orc.run_log.read_all()[-1].status == Status.SUCCESS
    assert not list(tmp_path.glob("**/findings.jsonl"))


def test_a_broken_ledger_does_not_take_down_the_run(tmp_path):
    """Observational write: the RunRecord is already appended by then."""
    class _ExplodingLedger:
        def record(self, finding):
            raise RuntimeError("disk on fire")

    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, [{"mode": "normal", "sleep_s": 0.0}],
                            hypotheses=(_proposal_hypothesis("RP-012"),))
    orc.findings = _ExplodingLedger()

    orc._step(orc.run_log.read_all())

    assert orc.run_log.read_all()[-1].status == Status.SUCCESS


# ---------------------------------------------------------------------------
# A ResearchAgent.propose() failure must never crash the whole run -- exposed
# by a real live run where the Research agent's depth-phase validation raised
# ResearchOutputError with nothing catching it, killing the entire process
# after only the baseline had run. CodingAgent.implement() and
# EvaluatorAgent.judge() were both built to never raise; propose() had no such
# contract and _step() had nothing guarding the call.
# ---------------------------------------------------------------------------

class _FlakyResearchAgent:
    """Raises on propose() for the first `fail_times` calls, then succeeds."""

    def __init__(self, fail_times, hypothesis="a real idea"):
        self._remaining = fail_times
        self._hypothesis = hypothesis

    def propose(self, history):
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError("simulated Research proposal failure")
        from agent.agents import Idea
        parent = history[-1].iteration if history else None
        return Idea(hypothesis=self._hypothesis, parent_iteration=parent)


def test_a_research_failure_is_recorded_and_the_run_continues(tmp_path):
    cfg = make_test_config(tmp_path)
    research = _FlakyResearchAgent(fail_times=1)
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.5}]
    orc = make_orchestrator(tmp_path, cfg, outcomes, research=research)

    orc._step(orc.run_log.read_all())   # the failed propose() attempt
    history = orc.run_log.read_all()
    assert len(history) == 1
    assert history[0].status == Status.FAILED
    assert history[0].aggregate is None
    assert "RuntimeError" in history[0].events[0].detail
    assert orc.state.halted is False
    assert orc.state.consecutive_research_failures == 1

    orc._step(orc.run_log.read_all())   # this time propose() succeeds
    history = orc.run_log.read_all()
    assert len(history) == 2
    assert history[-1].status == Status.SUCCESS
    assert orc.state.consecutive_research_failures == 0, "a successful propose() must reset the counter"


def test_repeated_research_failures_halt_for_a_human(tmp_path):
    cfg = make_test_config(tmp_path)   # default max_consecutive_research_failures=3
    research = _FlakyResearchAgent(fail_times=10)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], research=research)

    for _ in range(2):
        orc._step(orc.run_log.read_all())
    assert orc.state.halted is False

    orc._step(orc.run_log.read_all())   # the 3rd consecutive failure
    assert orc.state.halted is True
    assert "3 consecutive Research proposal failures" in orc.state.halt_reason

    history = orc.run_log.read_all()
    assert len(history) == 3
    assert [r.status for r in history] == [Status.FAILED, Status.FAILED, Status.ABANDONED]

    with pytest.raises(OrchestratorHalted):
        orc.run()


def test_research_failure_records_do_not_consume_a_max_iterations_slot(tmp_path):
    """2 failures stay below the default max_consecutive_research_failures=3,
    so both records are FAILED (not ABANDONED) -- and FAILED records, like a
    failed training attempt, must not count against max_iterations."""
    from agent.config import ConvergenceConfig
    from agent.convergence import should_stop

    cfg = make_test_config(tmp_path, convergence=ConvergenceConfig(max_iterations=1))
    research = _FlakyResearchAgent(fail_times=2)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], research=research)

    for _ in range(2):
        orc._step(orc.run_log.read_all())

    history = orc.run_log.read_all()
    assert [r.status for r in history] == [Status.FAILED, Status.FAILED]

    stop, reason = should_stop(history, cfg.convergence)
    assert stop is False, "FAILED research-proposal records must not trip max_iterations=1"


# ---------------------------------------------------------------------------
# An exhausted Research agent has FINISHED, not failed.
#
# The distinction matters because the two are handled oppositely. A failure to
# propose is recorded, retried, and escalated to a human once enough pile up --
# right for an agent that cannot produce a valid proposal, wrong for a
# deterministic backlog that has proposed every idea it holds. Without this
# split, an exploration campaign reaching its natural end would write failed
# records describing no real experiment and then halt as if it had broken.
# ---------------------------------------------------------------------------

class _ExhaustedResearchAgent:
    """Proposes `n` real ideas, then reports it has nothing left."""

    def __init__(self, n=0, hypothesis="a real idea"):
        self._remaining = n
        self._hypothesis = hypothesis
        self.calls = 0

    def propose(self, history):
        from agent.agents import Idea, ResearchExhausted

        self.calls += 1
        if self._remaining <= 0:
            raise ResearchExhausted("backlog exhausted: every idea has been tried")
        self._remaining -= 1
        return Idea(hypothesis=self._hypothesis,
                    parent_iteration=history[-1].iteration if history else None)


def test_an_exhausted_research_agent_stops_the_run_without_recording_a_failure(tmp_path):
    cfg = make_test_config(tmp_path)
    research = _ExhaustedResearchAgent(n=0)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], research=research)

    history = orc.run()

    assert history == []                                   # no junk iteration recorded
    assert orc.state.research_exhausted is True
    assert "exhausted" in orc.state.research_exhausted_reason
    assert orc.state.consecutive_research_failures == 0     # it did not fail at anything
    assert orc.state.halted is False                        # and nobody is being paged


def test_exhaustion_never_escalates_however_long_the_run_would_continue(tmp_path):
    """The failure path halts after max_consecutive_research_failures. If
    exhaustion went down that path, a finished campaign would report itself as
    a breakdown needing human intervention."""
    from agent.config import ConvergenceConfig

    cfg = make_test_config(tmp_path, convergence=ConvergenceConfig(max_iterations=50))
    research = _ExhaustedResearchAgent(n=0)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], research=research)

    orc.run()

    assert research.calls == 1          # asked once, believed the answer
    assert orc.state.halted is False
    assert not orc.run_log.read_all()


def test_a_resumed_run_whose_backlog_was_exhausted_stops_immediately(tmp_path):
    """Terminal and persisted: re-running must not call propose() again just to
    be told the same thing."""
    cfg = make_test_config(tmp_path)
    research = _ExhaustedResearchAgent(n=0)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], research=research)
    orc.run()
    assert research.calls == 1

    resumed = make_orchestrator(tmp_path, cfg, outcomes=[], research=research)
    assert resumed.state.research_exhausted is True     # survived the state file
    resumed.run()

    assert research.calls == 1                          # not asked a second time


def test_ideas_proposed_before_exhaustion_still_run_normally(tmp_path):
    """Exhaustion ends the run at the point it happens -- it does not discard
    the work the campaign already did."""
    from agent.config import ConvergenceConfig

    cfg = make_test_config(tmp_path, convergence=ConvergenceConfig(max_iterations=50))
    research = _ExhaustedResearchAgent(n=1)
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.5}]
    orc = make_orchestrator(tmp_path, cfg, outcomes, research=research)

    history = orc.run()

    assert [r.status for r in history] == [Status.SUCCESS]
    assert history[0].decision is not None               # really judged
    assert orc.state.research_exhausted is True


def test_a_plain_exception_is_still_a_failure_not_an_orderly_finish(tmp_path):
    """The split must not have loosened the failure path: anything that is not
    ResearchExhausted still records, retries and can escalate."""
    cfg = make_test_config(tmp_path)
    research = _FlakyResearchAgent(fail_times=1)
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.5}]
    orc = make_orchestrator(tmp_path, cfg, outcomes, research=research)

    orc._step(orc.run_log.read_all())

    assert orc.state.research_exhausted is False
    assert orc.state.consecutive_research_failures == 1
    assert [r.status for r in orc.run_log.read_all()] == [Status.FAILED]


def test_an_orderly_finish_is_visible_in_summary_json(tmp_path):
    """Why a run stopped has to be on disk -- no RunRecord is written for an
    orderly finish, so summary.json is where it has to show up."""
    import json

    from agent.config import ConvergenceConfig

    cfg = make_test_config(tmp_path, convergence=ConvergenceConfig(max_iterations=50))
    research = _ExhaustedResearchAgent(n=1)
    outcomes = [{"mode": "normal", "std": 0.0, "sleep_s": 0.0, "mean": 0.5}]
    orc = make_orchestrator(tmp_path, cfg, outcomes, research=research)

    orc.run()

    summary = json.loads((cfg.paths.logs_dir / "summary.json").read_text())
    assert "exhausted" in summary["stopped_because"]


def test_the_offline_backlog_exhaustion_error_is_an_orderly_finish(tmp_path):
    """The concrete case this exists for: OfflineResearchAgent running out of
    backlog entries at the end of an exploration campaign."""
    from agent.agents import ResearchExhausted
    from agent.research.offline import OfflineBacklogExhausted, OfflineResearchError

    assert issubclass(OfflineBacklogExhausted, ResearchExhausted)
    # and still what existing callers/tests catch
    assert issubclass(OfflineBacklogExhausted, OfflineResearchError)
