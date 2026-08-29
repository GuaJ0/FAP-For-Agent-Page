"""The Diff contract after splitting config_path from patch_path.

Diff used to carry a single `diff_path`, documented in agents.py as "where the
change is recorded (patch file, commit ref, ...)" but consumed by
orchestrator.py as the config path handed to the executor. Documented meaning
and executable meaning had drifted, and the executable one was load-bearing.

The split is deliberate about one thing in particular: the name `diff_path` is
GONE from Diff rather than reused for the patch. RunRecord.diff_path still
means "the config the executor ran", and having one name mean a patch file at
one layer and a config path at the next would be a worse trap than the
ambiguity it replaced. These tests pin both halves.
"""
import json
from pathlib import Path

import pytest

from agent.agents import AgentUsage, Diff, FakeCodingAgent, Idea
from agent.records import Status
from conftest import make_orchestrator, make_test_config


# ---------------------------------------------------------------------------
# The dataclass itself.
# ---------------------------------------------------------------------------

def test_diff_has_no_diff_path_field_at_all():
    """The point of the split. If this name comes back, so does the
    cross-layer collision with RunRecord.diff_path."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(Diff)}
    assert names == {"config_path", "solution_dir", "patch_path", "usage"}
    assert "diff_path" not in names


def test_optional_fields_default_to_none():
    """An agent that produces neither a patch nor usage stays a one-liner."""
    d = Diff(config_path="c", solution_dir="s")
    assert d.patch_path is None
    assert d.usage is None


def test_fake_coding_agent_reports_no_patch(tmp_path):
    """FakeCodingAgent copies a fixed fixture rather than editing anything, so
    there is genuinely no diff to point at. None is the honest value."""
    agent = FakeCodingAgent(tmp_path, [{"mode": "normal", "sleep_s": 0.0}])

    diff = agent.implement(Idea("h", None), None)

    assert diff.patch_path is None
    assert Path(diff.config_path).exists()
    assert Path(diff.solution_dir).is_dir()
    assert Path(diff.config_path).parent == Path(diff.solution_dir)


# ---------------------------------------------------------------------------
# What the orchestrator does with it.
# ---------------------------------------------------------------------------

def test_the_executor_is_pointed_at_config_path(tmp_path):
    """End to end through the real orchestrator: if config_path stopped being
    what reaches the executor, every run would fail at the first seed."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    assert orc.run_log.read_all()[-1].status == Status.SUCCESS


def test_run_record_diff_path_holds_the_config_path(tmp_path):
    """Deliberate: runs.jsonl's meaning is unchanged, the field is the only
    path always present, and accumulation resolves a sibling train.py from it.
    See Orchestrator._record_diff_path for the full reasoning."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.diff_path is not None
    assert Path(record.diff_path).exists()
    # It is a config, not a patch: readable, and sitting in the solution dir.
    assert json.loads(Path(record.diff_path).read_text())["mode"] == "normal"
    assert (Path(record.diff_path).parent / "train.py").exists()


def test_the_sibling_train_py_resolution_accumulation_depends_on_still_works(tmp_path):
    """The concrete reason RunRecord.diff_path keeps holding the config path:
    LLMCodingAgent._current_best_source() takes this path's sibling train.py.
    Pointing the field at a patch would break that for every agent, and for
    the Fake agent and the baseline there is no patch to point at anyway."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    train_py = Path(record.diff_path).parent / "train.py"
    assert train_py.exists()
    assert "def main" in train_py.read_text()


def test_failed_iterations_also_record_the_config_path(tmp_path):
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "crash", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.status == Status.FAILED
    assert Path(record.diff_path).exists()
