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
