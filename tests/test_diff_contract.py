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


# ---------------------------------------------------------------------------
# RunRecord.patch_path: the permanent record now remembers the code change,
# not just the settings.
#
# The hypothesis log is a graded artifact and an LLM wrote the code, so
# "did this iteration actually implement what its hypothesis claimed" has to
# be answerable by following runs.jsonl -- not by hunting for the solution
# directory by hand.
# ---------------------------------------------------------------------------

def test_a_real_coding_agent_run_records_a_readable_patch(tmp_path):
    """End to end through the real LLMCodingAgent and the real Orchestrator.

    Asserts the file on disk, not just that the field holds a string: a path
    that doesn't resolve, or resolves to an empty file, would satisfy a
    type check while being useless for the verification this exists for.
    """
    from agent.coding import LLMCodingAgent, ScriptedClient

    REPO_ROOT = Path(__file__).resolve().parent.parent
    template = (REPO_ROOT / "agent" / "coding" / "templates" / "train_ranking.py").read_text()

    cfg = make_test_config(tmp_path)
    coding = LLMCodingAgent(
        work_dir=tmp_path / "solutions",
        data_dir=str(tmp_path),
        llm=ScriptedClient([f"```python\n{template}\n```\n"]),
        usage_log_path=tmp_path / "usage.jsonl",
        run_smoke_test=False,
    )
    # The generated train.py needs the real data to run, which this test does
    # not have -- the executor failing is fine and is itself the failed-path
    # case. What matters is the patch the CodingAgent recorded on the way.
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=coding)

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.patch_path is not None, "the real agent produced no patch_path"

    patch = Path(record.patch_path)
    assert patch.exists(), f"patch_path points at nothing: {patch}"
    content = patch.read_text()
    assert content.strip(), "the recorded patch is empty"
    assert patch.name == "changes.patch"
    # It is a real unified diff of the change this hypothesis made.
    assert "@@" in content or content.startswith("---")
    assert "bpr" in content.lower()

    # And it is genuinely a different artifact from the config.
    assert record.diff_path != record.patch_path
    assert Path(record.diff_path).exists()


def test_patch_path_survives_the_jsonl_round_trip_in_a_real_run(tmp_path):
    """It has to still be there after the record has been through disk."""
    from agent.coding import LLMCodingAgent, ScriptedClient

    REPO_ROOT = Path(__file__).resolve().parent.parent
    template = (REPO_ROOT / "agent" / "coding" / "templates" / "train_ranking.py").read_text()

    cfg = make_test_config(tmp_path)
    coding = LLMCodingAgent(
        work_dir=tmp_path / "solutions", data_dir=str(tmp_path),
        llm=ScriptedClient([f"```python\n{template}\n```\n"]),
        usage_log_path=tmp_path / "usage.jsonl", run_smoke_test=False,
    )
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=coding)

    orc._step(orc.run_log.read_all())

    raw = json.loads(cfg.paths.runs_jsonl.read_text().strip().splitlines()[-1])
    assert raw["patch_path"] == orc.run_log.read_all()[-1].patch_path
    assert Path(raw["patch_path"]).exists()


def test_fake_agent_records_no_patch_path(tmp_path):
    """FakeCodingAgent copies a fixture rather than editing anything. None,
    not an error and not a bogus path."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.patch_path is None
    assert record.diff_path is not None      # the config is still recorded


def test_failed_iterations_record_the_patch_too(tmp_path):
    """A failed attempt is exactly when you most want to see what code ran."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "crash", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.status == Status.FAILED
    assert record.patch_path is None         # the Fake makes no patch


def test_bootstrap_baseline_records_no_patch_path(tmp_path):
    """The seeded baseline is a pre-existing solution, not an edit to one, so
    there is no diff for it to point at."""
    import shutil

    from agent.agents import Diff, Idea

    FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "fake_train.py"
    sol = tmp_path / "baseline"
    sol.mkdir()
    shutil.copy(FIXTURE, sol / "train.py")
    cfg_file = sol / "config.json"
    cfg_file.write_text(json.dumps({"mode": "normal", "sleep_s": 0.0, "mean": 0.6015, "std": 0.0}))

    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[])

    record = orc.bootstrap_baseline(
        Idea("baseline", None),
        Diff(config_path=str(cfg_file), solution_dir=str(sol)),
    )

    assert record.patch_path is None
    assert record.diff_path == str(cfg_file)
    assert orc.run_log.read_all()[0].patch_path is None
