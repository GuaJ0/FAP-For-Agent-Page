import pytest

from agent.agents import FakeCodingAgent
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
