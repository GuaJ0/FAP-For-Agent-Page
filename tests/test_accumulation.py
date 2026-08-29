"""Accumulation: a new idea builds on the best ACCEPTED solution, not always
on the static solution/train.py.

Without this, improvements never compounded -- iteration 5 was written against
the baseline, not against whatever iterations 1-4 had established.

The correctness anchor for the whole feature is
test_right_after_bootstrap_the_lookup_and_the_static_baseline_agree: straight
after bootstrap_baseline(), registry.best() is iteration 0 whose config sits
in solution/ itself, so "current best" and "static baseline" must resolve to
literally the same file. The lookup may only diverge once something actually
beats the baseline.
"""
import json
import shutil
from pathlib import Path

import pytest

from agent.agents import Diff, Idea
from agent.coding import LLMCodingAgent, ScriptedClient
from agent.coding import prompts
from agent.records import (
    AggregateMetrics,
    Event,
    ResourceUsage,
    RunLog,
    RunRecord,
    Status,
)
from agent.registry import CheckpointRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_BASELINE = (REPO_ROOT / "solution" / "train.py").read_text()
RANKING_TEMPLATE = (REPO_ROOT / "agent" / "coding" / "templates" / "train_ranking.py").read_text()


def _fenced(source):
    return f"```python\n{source}\n```\n"


def _agent(tmp_path, responses=(), **kwargs):
    kwargs.setdefault("run_smoke_test", False)
    return LLMCodingAgent(
        work_dir=tmp_path / "solutions",
        data_dir=str(tmp_path / "data"),
        llm=ScriptedClient(list(responses) or [_fenced(RANKING_TEMPLATE)]),
        usage_log_path=tmp_path / "usage.jsonl",
        **kwargs,
    )


def _accepted_iteration(tmp_path, iteration, source, primary, name=None):
    """Write the on-disk state the orchestrator leaves behind for an accepted
    iteration: a solution dir with train.py + its config, a RunRecord in
    runs.jsonl, and a registry entry."""
    sol = tmp_path / (name or f"sol_{iteration}")
    sol.mkdir(parents=True, exist_ok=True)
    (sol / "train.py").write_text(source)
    cfg = sol / "config.json"
    cfg.write_text(json.dumps({"iteration": iteration}))

    RunLog(tmp_path / "runs.jsonl").append(RunRecord(
        iteration=iteration, parent_iteration=None, timestamp="2026-01-01T00:00:00+00:00",
        hypothesis=f"idea {iteration}", diff_path=str(cfg), status=Status.SUCCESS,
        seeds=[],
        aggregate=AggregateMetrics(primary, 0.0, primary, primary, 1),
        delta_vs_current_best=None, decision=None, events=[],
        resources=ResourceUsage(wall_s=1.0),
    ))
    CheckpointRegistry(tmp_path / "registry.json").register(iteration, str(sol), primary)
    return sol


def _lookup_paths(tmp_path):
    return {"registry_path": tmp_path / "registry.json", "run_log_path": tmp_path / "runs.jsonl"}


# ---------------------------------------------------------------------------
# The correctness anchor.
# ---------------------------------------------------------------------------

def test_right_after_bootstrap_the_lookup_and_the_static_baseline_agree(tmp_path):
    """Straight after bootstrap_baseline() the best is iteration 0, whose
    diff_path is solution/config.yaml -- so the sibling train.py IS
    solution/train.py. The new lookup must be a no-op in that state."""
    RunLog(tmp_path / "runs.jsonl").append(RunRecord(
        iteration=0, parent_iteration=None, timestamp="2026-01-01T00:00:00+00:00",
        hypothesis="baseline", diff_path=str(REPO_ROOT / "solution" / "config.yaml"),
        status=Status.SUCCESS, seeds=[],
        aggregate=AggregateMetrics(0.6015, 0.0, 0.667, 0.536, 1),
        delta_vs_current_best=None, decision=None,
        events=[Event(type="bootstrap", detail="", agent_action="orchestrator")],
        resources=ResourceUsage(wall_s=1.0),
    ))
    CheckpointRegistry(tmp_path / "registry.json").register(
        0, str(REPO_ROOT / "solution"), 0.6015)

    agent = _agent(tmp_path, **_lookup_paths(tmp_path))
    source, provenance = agent._current_best_source()

    assert source == STATIC_BASELINE == prompts.load_baseline_source()
    assert "iteration 0" in provenance


def test_it_diverges_only_once_something_beats_the_baseline(tmp_path):
    _accepted_iteration(tmp_path, 0, STATIC_BASELINE, 0.6015, name="solution_copy")
    before, _ = _agent(tmp_path, **_lookup_paths(tmp_path))._current_best_source()

    _accepted_iteration(tmp_path, 1, RANKING_TEMPLATE, 0.6200)
    after, provenance = _agent(tmp_path, **_lookup_paths(tmp_path))._current_best_source()

    assert before == STATIC_BASELINE
    assert after == RANKING_TEMPLATE
    assert "iteration 1" in provenance and "0.6200" in provenance


# ---------------------------------------------------------------------------
# Resolution.
# ---------------------------------------------------------------------------

def test_resolves_the_best_iteration_not_the_latest(tmp_path):
    """registry.best() is by score, not recency. A later, worse accepted
    iteration must not become the thing new ideas are written against."""
    _accepted_iteration(tmp_path, 1, RANKING_TEMPLATE, 0.6200)
    _accepted_iteration(tmp_path, 2, "# a worse idea\n", 0.5800)

    source, provenance = _agent(tmp_path, **_lookup_paths(tmp_path))._current_best_source()

    assert source == RANKING_TEMPLATE
    assert "iteration 1" in provenance


def test_the_generate_prompt_carries_the_accumulated_source(tmp_path):
    """End of the chain: the resolved source is what the model actually sees."""
    _accepted_iteration(tmp_path, 1, RANKING_TEMPLATE, 0.6200)
    client = ScriptedClient([_fenced(RANKING_TEMPLATE)])
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=client,
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=False,
        **_lookup_paths(tmp_path),
    )

    agent.implement(Idea("next idea", None), None)

    _, user, _ = client.calls[0]
    assert "step_bpr" in user            # the accepted iteration's source
    assert "Current best train.py" in user


def test_the_manifest_records_what_the_attempt_was_built_from(tmp_path):
    _accepted_iteration(tmp_path, 1, RANKING_TEMPLATE, 0.6200)
    agent = _agent(tmp_path, **_lookup_paths(tmp_path))

    diff = agent.implement(Idea("next idea", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert "iteration 1" in manifest["built_from"]


def test_the_patch_is_diffed_against_the_accumulated_source(tmp_path):
    """changes.patch should show the delta from what this idea actually built
    on, not from a baseline that was superseded three iterations ago."""
    _accepted_iteration(tmp_path, 1, RANKING_TEMPLATE, 0.6200)
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE + "\n# a tiny change\n")],
                   **_lookup_paths(tmp_path))

    diff = agent.implement(Idea("next idea", None), None)

    patch = (Path(diff.solution_dir) / "changes.patch").read_text()
    assert "a tiny change" in patch
    # Against the accumulated source this is a one-line addition, not a
    # wholesale rewrite of the baseline.
    assert len([l for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]) < 5


# ---------------------------------------------------------------------------
# Graceful degradation: implement() must never crash over provenance.
# ---------------------------------------------------------------------------

def test_without_lookup_paths_behaviour_is_exactly_as_before(tmp_path):
    """The default. Existing callers and tests pass neither path."""
    source, provenance = _agent(tmp_path)._current_best_source()

    assert source == STATIC_BASELINE
    assert "static baseline" in provenance


def test_empty_registry_falls_back(tmp_path):
    (tmp_path / "runs.jsonl").write_text("")
    source, provenance = _agent(tmp_path, **_lookup_paths(tmp_path))._current_best_source()
    assert source == STATIC_BASELINE
    assert "static baseline" in provenance


def test_missing_files_fall_back(tmp_path):
    agent = _agent(tmp_path, registry_path=tmp_path / "nope.json",
                   run_log_path=tmp_path / "nope.jsonl")
    assert agent._current_best_source()[0] == STATIC_BASELINE


def test_registry_entry_with_no_matching_run_record_falls_back(tmp_path):
    CheckpointRegistry(tmp_path / "registry.json").register(7, str(tmp_path), 0.7)
    (tmp_path / "runs.jsonl").write_text("")

    assert _agent(tmp_path, **_lookup_paths(tmp_path))._current_best_source()[0] == STATIC_BASELINE


def test_a_cleaned_up_solution_dir_falls_back(tmp_path):
    """Attempt dirs are gitignored and get cleaned. A registry entry pointing
    at a directory that no longer exists must not break the next idea."""
    sol = _accepted_iteration(tmp_path, 1, RANKING_TEMPLATE, 0.6200)
    shutil.rmtree(sol)

    assert _agent(tmp_path, **_lookup_paths(tmp_path))._current_best_source()[0] == STATIC_BASELINE


def test_a_corrupt_run_log_falls_back_without_raising(tmp_path):
    CheckpointRegistry(tmp_path / "registry.json").register(1, str(tmp_path), 0.7)
    (tmp_path / "runs.jsonl").write_text("{not json\n")

    source, provenance = _agent(tmp_path, **_lookup_paths(tmp_path))._current_best_source()

    assert source == STATIC_BASELINE
    assert "static baseline" in provenance


def test_implement_still_succeeds_when_resolution_fails(tmp_path):
    """The whole point of failing soft: a provenance problem must not take
    down the run."""
    CheckpointRegistry(tmp_path / "registry.json").register(1, str(tmp_path), 0.7)
    (tmp_path / "runs.jsonl").write_text("{not json\n")
    agent = _agent(tmp_path, **_lookup_paths(tmp_path))

    diff = agent.implement(Idea("an idea", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is True
