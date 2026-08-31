"""Unit tests for the real CodingAgent (agent/coding/).

Deterministic, offline and free: every test here drives the agent with a
ScriptedClient, so no OpenAI call is made and no KuaiRand data is touched.
The tests that need a real API call or the real dataset live in
tests/test_coding_agent_e2e.py behind the -m llm / -m slow gates.
"""
import json
from pathlib import Path

import pytest

from agent.agents import Idea
from agent.coding import LLMCodingAgent, ScriptedClient, extract_code, static_check
from agent.coding.agent import CodeExtractionError, AttemptOutcome
from agent.coding.llm import (
    LLMResponse,
    OpenAIClient,
    OpenAIClientError,
    TemplateLibraryClient,
    UsageLog,
    estimate_cost,
    price_for,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = (REPO_ROOT / "solution" / "train.py").read_text()
RANKING_TEMPLATE = (REPO_ROOT / "agent" / "coding" / "templates" / "train_ranking.py").read_text()


def _agent(tmp_path, responses, **kwargs):
    kwargs.setdefault("run_smoke_test", False)
    return LLMCodingAgent(
        work_dir=tmp_path / "solutions",
        data_dir=str(tmp_path / "data"),
        llm=ScriptedClient(list(responses)),
        usage_log_path=tmp_path / "usage.jsonl",
        **kwargs,
    )


def _fenced(source):
    return f"Here you go:\n\n```python\n{source}\n```\n"


# ---------------------------------------------------------------------------
# extract_code
# ---------------------------------------------------------------------------

def test_extract_code_takes_a_fenced_block():
    assert extract_code("blah\n```python\nimport numpy\n```\n").strip() == "import numpy"


def test_extract_code_takes_the_longest_block_not_the_first():
    """Models often open with a short illustrative snippet before the real
    file. Taking the first block silently ships the snippet."""
    text = (
        "First, the key change:\n```python\nloss = bpr(z)\n```\n"
        "And here is the full file:\n```python\nimport numpy as np\n\n"
        "def main():\n    pass\n```\n"
    )
    assert "def main" in extract_code(text)
    assert extract_code(text).strip() != "loss = bpr(z)"


def test_extract_code_accepts_an_unfenced_file():
    assert "import numpy" in extract_code("import numpy as np\n\ndef main():\n    pass\n")


def test_extract_code_rejects_prose():
    with pytest.raises(CodeExtractionError):
        extract_code("I would suggest using a pairwise loss. Let me know if you want code.")


# ---------------------------------------------------------------------------
# static_check
# ---------------------------------------------------------------------------

def test_static_check_passes_the_real_baseline_and_template():
    """Negative control: if these fail, the checker is wrong, not the code."""
    assert static_check(BASELINE) == []
    assert static_check(RANKING_TEMPLATE) == []


def test_static_check_rejects_a_syntax_error():
    problems = static_check("def f(:\n    pass\n")
    assert len(problems) == 1 and "does not parse" in problems[0]


@pytest.mark.parametrize("mod,hint", [
    ("subprocess", "spawning subprocesses is not permitted"),
    ("socket", "no network access"),
    ("requests", "no network access"),
    ("urllib", "no network access"),
])
def test_static_check_refuses_sandbox_violations_however_it_is_provisioned(mod, hint):
    """These are sandbox rules, not packaging facts -- every one of them is in
    the stdlib and would sail through an availability check."""
    problems = static_check(f"import {mod}\n")
    assert any(hint in p for p in problems), problems


def test_an_installed_third_party_library_is_allowed():
    """External libraries are permitted (docs/coding-agent.md). The old
    allowlist rejected pandas, sklearn and scipy as "not installed" while all
    three were installed the whole time."""
    from agent.coding.agent import available_third_party

    installed = [m for m in available_third_party() if m != "numpy"]
    if not installed:
        pytest.skip("no third-party library beyond numpy is installed here")

    for mod in installed:
        problems = static_check(f"import {mod}\n")
        assert not any("not installed" in p for p in problems), (mod, problems)


def test_an_uninstalled_library_is_rejected_early_with_an_actionable_reason():
    """Permitted-but-absent must still fail HERE. Letting it reach the executor
    turns a free static check into a CRASH after a full multi-seed run, and the
    traceback reads like a modelling bug rather than a missing package."""
    from agent.coding.agent import module_available

    absent = next((m for m in ("torch", "jax", "tensorflow", "lightgbm")
                   if not module_available(m)), None)
    if absent is None:
        pytest.skip("every candidate library is installed here")

    problems = static_check(f"import {absent}\n")
    assert any("not installed in this environment" in p for p in problems), problems
    # and it must say what IS available, or the repair prompt is a guessing game
    assert any("Installed and available" in p for p in problems), problems


def test_availability_is_measured_not_declared(monkeypatch):
    """The whole point of the refactor: provisioning a library must make it
    legal without anyone editing an allowlist, and un-provisioning it must make
    it illegal again. Both directions are simulated so the test says the same
    thing whether or not torch happens to be installed on this machine."""
    from agent.coding import agent as coding_agent

    real = coding_agent.module_available

    monkeypatch.setattr(coding_agent, "module_available",
                        lambda root: False if root == "torch" else real(root))
    assert any("not installed" in p for p in static_check("import torch\n"))

    monkeypatch.setattr(coding_agent, "module_available",
                        lambda root: True if root == "torch" else real(root))
    assert not any("not installed" in p for p in static_check("import torch\n"))


def test_static_check_rejects_a_reimplemented_metric():
    """evaluate.py is the sole scoring authority. A local evaluate() would be
    caught later by executor-side verification anyway, but at the cost of a
    full training run -- catch it for free instead."""
    src = BASELINE.replace("def train(cfg, seed):", "def evaluate(u, y, s):\n    return {}\n\n\ndef train(cfg, seed):")
    problems = static_check(src)
    assert any("sole scoring authority" in p for p in problems), problems


def test_static_check_rejects_a_test_split_key_in_result_json():
    problems = static_check(BASELINE.replace('"seed": int(seed),', '"test_primary": 0.6,'))
    assert any("test_primary" in p for p in problems), problems


@pytest.mark.parametrize("flag", ["--config", "--seed", "--out"])
def test_static_check_requires_every_cli_flag(flag):
    problems = static_check(BASELINE.replace(flag, "--nope"))
    assert any(flag in p for p in problems), problems


def test_static_check_requires_raw_predictions_and_test_metrics_line():
    problems = static_check(BASELINE.replace("val_predictions", "vp").replace("TEST_METRICS", "TM"))
    assert any("val_predictions.npz" in p for p in problems), problems
    assert any("TEST_METRICS" in p for p in problems), problems


def test_static_check_requires_a_call_to_evaluate():
    src = "import argparse\n# --config --seed --out TEST_METRICS val_predictions\n"
    assert any("never calls evaluate()" in p for p in static_check(src)), static_check(src)


# ---------------------------------------------------------------------------
# implement(): the happy path and the artifacts it leaves behind
# ---------------------------------------------------------------------------

def test_implement_returns_a_runnable_self_contained_solution_dir(tmp_path):
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    sol = Path(diff.solution_dir)
    assert (sol / "train.py").read_text().strip() == RANKING_TEMPLATE.strip()
    # Vendored harness copied in: the executor runs train.py with
    # cwd=solution_dir, so these must be importable as top-level modules.
    assert (sol / "evaluate.py").exists()
    assert (sol / "data.py").exists()


def test_config_path_is_what_the_orchestrator_will_pass_to_the_executor(tmp_path):
    """orchestrator.py hands Diff.config_path to Executor.run_seeds. If this
    ever stops being a readable config, every run fails at the first seed.

    This test used to be named for `diff_path`, back when Diff had a single
    field documented as a patch file but consumed as the config. The field is
    now split into config_path and patch_path, and this pins the one the
    executor depends on.
    """
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    cfg_path = Path(diff.config_path)
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text())["data_dir"] == str(tmp_path / "data")


def test_patch_path_points_at_the_real_unified_diff(tmp_path):
    """The other half of the split: patch_path is an actual patch file, which
    is what Diff's field was always documented to be."""
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    assert diff.patch_path is not None
    patch = Path(diff.patch_path)
    assert patch.name == "changes.patch"
    assert patch.exists()
    assert patch.parent == Path(diff.solution_dir)


def test_a_real_unified_diff_is_still_written_for_the_audit_trail(tmp_path):
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    patch = (Path(diff.solution_dir) / "changes.patch").read_text()
    assert patch.startswith("---") or "@@" in patch
    assert "bpr" in patch.lower()


def test_base_config_overrides_reach_the_config_file(tmp_path):
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)], base_config={"loss": "bpr", "epochs": 5})

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    cfg = json.loads(Path(diff.config_path).read_text())
    assert cfg["loss"] == "bpr" and cfg["epochs"] == 5


def test_attempt_manifest_records_what_happened(tmp_path):
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])

    diff = agent.implement(Idea("use a pairwise BPR loss", 3), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is True
    assert manifest["parent_iteration"] == 3
    assert manifest["had_orchestrator_feedback"] is False
    assert manifest["cycles"][-1]["ok"] is True


def test_successive_calls_get_their_own_solution_dirs(tmp_path):
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE), _fenced(RANKING_TEMPLATE)])

    a = agent.implement(Idea("idea one", None), None)
    b = agent.implement(Idea("idea two", None), None)

    assert a.solution_dir != b.solution_dir
    assert Path(a.solution_dir).exists() and Path(b.solution_dir).exists()


# ---------------------------------------------------------------------------
# The inner repair loop
# ---------------------------------------------------------------------------

def test_a_static_violation_is_repaired_without_costing_an_executor_run(tmp_path):
    agent = _agent(tmp_path, [_fenced("import subprocess\n" + BASELINE), _fenced(RANKING_TEMPLATE)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert [c["ok"] for c in manifest["cycles"]] == [False, True]
    assert "subprocess" in manifest["cycles"][0]["detail"]
    assert "not permitted" in manifest["cycles"][0]["detail"]
    assert manifest["succeeded"] is True


def test_the_repair_prompt_contains_the_diagnosis_and_the_broken_source(tmp_path):
    """A repair prompt without the previous source makes the model rewrite
    from scratch, which loses whatever it had already got right."""
    client = ScriptedClient([_fenced("import subprocess\n" + BASELINE), _fenced(RANKING_TEMPLATE)])
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=client,
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=False,
    )

    agent.implement(Idea("use a pairwise BPR loss", None), None)

    _, repair_user, purpose = client.calls[1]
    assert purpose == "repair"
    assert "subprocess" in repair_user and "not permitted" in repair_user
    assert "import subprocess" in repair_user
    assert "COMPLETE corrected" in repair_user


def test_repairs_are_capped_and_the_last_attempt_is_still_shipped(tmp_path):
    """When every repair fails the agent must still return a Diff. Raising
    would kill the orchestrator's whole run; returning lets the executor
    record one honest failed iteration and tier-1 retry take over."""
    bad = _fenced("import subprocess\n" + BASELINE)
    agent = _agent(tmp_path, [bad, bad, bad], max_repair_attempts=2)

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is False
    assert len(manifest["cycles"]) == 3          # 1 generate + 2 repairs, no more
    assert (Path(diff.solution_dir) / "train.py").exists()
    assert Path(diff.config_path).exists()


def test_prose_only_response_is_treated_as_a_failed_cycle(tmp_path):
    agent = _agent(tmp_path, ["I'd recommend BPR.", _fenced(RANKING_TEMPLATE)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["cycles"][0]["stage"] == "extract"
    assert manifest["succeeded"] is True


# ---------------------------------------------------------------------------
# The orchestrator's `feedback` argument
# ---------------------------------------------------------------------------

def test_orchestrator_feedback_produces_a_repair_prompt_not_a_fresh_generate(tmp_path):
    client = ScriptedClient([_fenced(RANKING_TEMPLATE)])
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=client,
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=False,
    )

    agent.implement(
        Idea("use a pairwise BPR loss", None),
        "crash: Traceback...\nValueError: operands could not be broadcast together",
    )

    _, user, purpose = client.calls[0]
    assert purpose == "repair_after_orchestrator_feedback"
    assert "could not be broadcast" in user
    assert "How it failed" in user


def test_long_feedback_is_trimmed_from_the_front(tmp_path):
    """The tail of a traceback is the useful part, so trimming must not drop
    the exception line."""
    from agent.coding.prompts import format_failure_feedback

    feedback = "noise\n" * 5000 + "ValueError: the actual problem"
    trimmed = format_failure_feedback(feedback, limit=200)

    assert "ValueError: the actual problem" in trimmed
    assert len(trimmed) < 400


def test_a_retry_repairs_the_previously_shipped_source(tmp_path):
    client = ScriptedClient([_fenced(RANKING_TEMPLATE), _fenced(RANKING_TEMPLATE)])
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=client,
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=False,
    )

    agent.implement(Idea("use a pairwise BPR loss", None), None)
    agent.implement(Idea("use a pairwise BPR loss", None), "crash: boom")

    _, retry_user, _ = client.calls[1]
    assert "step_bpr" in retry_user   # the source it actually shipped last time


# ---------------------------------------------------------------------------
# Token / cost accounting
# ---------------------------------------------------------------------------

def test_usage_is_logged_per_call_with_tokens_and_cost(tmp_path):
    agent = _agent(tmp_path, [_fenced("import subprocess\n" + BASELINE), _fenced(RANKING_TEMPLATE)])

    agent.implement(Idea("use a pairwise BPR loss", None), None)

    rows = [json.loads(l) for l in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert [r["purpose"] for r in rows] == ["generate", "repair"]
    assert all(r["tokens_in"] > 0 and r["tokens_out"] > 0 for r in rows)
    assert all(r["agent"] == "coding" for r in rows)
    assert agent.last_usage["llm_calls"] == 2
    assert agent.last_usage["tokens_in"] == sum(r["tokens_in"] for r in rows)


def test_usage_log_totals(tmp_path):
    log = UsageLog(tmp_path / "u.jsonl")
    log.record(LLMResponse("x", "gpt-5", 1000, 2000, estimate_cost("gpt-5", 1000, 2000)),
               purpose="generate", idea="i", attempt=0)
    log.record(LLMResponse("y", "gpt-5", 500, 100, estimate_cost("gpt-5", 500, 100)),
               purpose="repair", idea="i", attempt=1)

    totals = log.totals()
    # gpt-5 at (1.25, 10.00) USD/Mtok:
    #   (1000*1.25 + 2000*10)/1e6 = 0.021250
    #   ( 500*1.25 +  100*10)/1e6 = 0.001625
    assert totals == {"calls": 2, "real_model_calls": 2, "tokens_in": 1500,
                      "tokens_out": 2100, "cost_usd": pytest.approx(0.022875)}


def test_usage_log_never_records_an_api_key(tmp_path):
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])
    agent.implement(Idea("use a pairwise BPR loss", None), None)
    assert "sk-" not in (tmp_path / "usage.jsonl").read_text()


def test_as_resource_usage_shim_for_the_unwired_gap(tmp_path):
    """RunRecord.resources is not populated -- wiring it needs orchestrator.py.
    This shim is what should be used when that happens."""
    log = UsageLog(tmp_path / "u.jsonl")
    log.record(LLMResponse("x", "gpt-5", 10, 20, 0.001), purpose="generate", idea="i", attempt=0)

    ru = log.as_resource_usage(wall_s=12.5)
    assert (ru.wall_s, ru.tokens_in, ru.tokens_out) == (12.5, 10, 20)


@pytest.mark.parametrize("model,expected", [
    ("gpt-5", (1.25, 10.00)),
    ("gpt-5-mini", (0.25, 2.00)),          # must NOT be priced as gpt-5
    ("gpt-5-2025-01-01", (1.25, 10.00)),   # dated snapshots inherit the base price
    ("something-unknown", (0.0, 0.0)),
])
def test_pricing_prefix_resolution(model, expected):
    assert price_for(model) == expected


def test_pricing_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICE_IN_PER_MTOK", "2.0")
    monkeypatch.setenv("OPENAI_PRICE_OUT_PER_MTOK", "8.0")
    assert price_for("gpt-5") == (2.0, 8.0)
    assert estimate_cost("gpt-5", 1_000_000, 1_000_000) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def test_openai_client_refuses_to_construct_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(OpenAIClientError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_template_library_is_never_counted_as_a_model_call():
    resp = TemplateLibraryClient().complete("sys", "use a pairwise bpr loss")
    assert resp.is_real_model_call is False
    assert resp.cost_usd == 0.0
    assert "step_bpr" in resp.text


def test_template_library_says_so_when_it_has_no_match():
    resp = TemplateLibraryClient().complete("sys", "model the censored watch-time distribution")
    assert resp.text.startswith("NO_TEMPLATE")


def test_unmatched_template_becomes_a_failed_attempt_not_a_crash(tmp_path):
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=TemplateLibraryClient(),
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=False, max_repair_attempts=0,
    )

    diff = agent.implement(Idea("model the censored watch-time distribution", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is False
    assert "NO_TEMPLATE" in manifest["cycles"][0]["detail"]


def test_scripted_client_running_dry_is_an_explicit_failure():
    client = ScriptedClient([])
    with pytest.raises(AssertionError, match="ran out of responses"):
        client.complete("s", "u")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_llm_coding_agent_satisfies_the_coding_agent_protocol(tmp_path):
    import inspect

    from agent.agents import CodingAgent, Diff

    # CodingAgent isn't @runtime_checkable, so conformance is checked
    # structurally: same method, same signature the orchestrator calls with.
    expected = inspect.signature(CodingAgent.implement)
    actual = inspect.signature(LLMCodingAgent.implement)
    assert list(actual.parameters) == list(expected.parameters)

    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])
    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)
    assert isinstance(diff, Diff)


# ---------------------------------------------------------------------------
# The smoke run. Exercised here without KuaiRand data by pointing the agent at
# fixtures/lying_train.py, which honours the same CLI contract on synthetic
# data -- so this stays a fast offline test of the smoke path itself.
# ---------------------------------------------------------------------------

LIAR = (REPO_ROOT / "fixtures" / "lying_train.py").read_text()


def _smoke_agent(tmp_path, responses, **kwargs):
    kwargs.setdefault("max_repair_attempts", 0)
    kwargs.setdefault("base_config", {"mode": "honest"})
    return LLMCodingAgent(
        work_dir=tmp_path / "solutions",
        data_dir=str(tmp_path),
        llm=ScriptedClient(list(responses)),
        usage_log_path=tmp_path / "usage.jsonl",
        run_smoke_test=True,
        smoke_timeout_s=120.0,
        **kwargs,
    )


def test_smoke_run_accepts_a_solution_that_honours_the_contract(tmp_path):
    agent = _smoke_agent(tmp_path, [_fenced(LIAR)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is True, manifest["cycles"]
    assert "smoke run passed" in manifest["cycles"][-1]["detail"]


def test_smoke_run_catches_a_train_py_that_misreports_before_it_costs_a_real_run(tmp_path):
    """The executor would catch this too -- but only after a full multi-seed
    training run, and at the cost of one of three tier-1 attempts. The smoke
    run catches it in seconds instead."""
    agent = _smoke_agent(tmp_path, [_fenced(LIAR)], base_config={"mode": "inflate"})

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is False
    assert manifest["cycles"][-1]["stage"] == "verify"
    assert "mismatch" in manifest["cycles"][-1]["detail"]


def test_smoke_run_catches_a_crash(tmp_path):
    broken = LIAR.replace("def main():", "def main():\n    raise RuntimeError('boom')\n")
    agent = _smoke_agent(tmp_path, [_fenced(broken)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is False
    assert manifest["cycles"][-1]["stage"] == "smoke"
    assert "boom" in manifest["cycles"][-1]["detail"]


def test_a_crash_after_printing_test_metrics_does_not_leak_them(tmp_path):
    """The leak this guards: a candidate that prints its TEST_METRICS line and
    then exits non-zero WITHOUT raising.

    sys.exit(1) leaves stderr empty, so the crash path's
    `proc.stderr or proc.stdout` falls through to stdout -- whose last lines
    still carry the hidden-test metrics. That text becomes
    AttemptOutcome.detail, which is fed into a repair prompt and sent to a
    real, billable LLM call.

    test_smoke_run_catches_a_crash does NOT cover this: it raises at the top of
    main(), before any TEST_METRICS print, and a raised exception fills stderr
    with a traceback, so the stdout fallback is never reached.
    """
    # Crash after main() has run, so the TEST_METRICS line is already on
    # stdout. sys.exit(1) with an int writes nothing to stderr.
    crashing = LIAR.replace(
        'if __name__ == "__main__":\n    sys.exit(main())',
        'if __name__ == "__main__":\n    main()\n    sys.exit(1)',
    )
    assert crashing != LIAR, "the fixture's entrypoint changed; this mutation no longer applies"

    secret = 0.987654          # a value that appears ONLY on the TEST_METRICS line
    client = ScriptedClient([_fenced(crashing), _fenced(LIAR)])
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=client,
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=True, smoke_timeout_s=120.0,
        base_config={"mode": "inflate",
                     "claimed": {"primary": secret, "gauc": secret, "ndcg5": secret}},
        max_repair_attempts=1,
    )

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    detail = manifest["cycles"][0]["detail"]

    # It still diagnoses the crash...
    assert manifest["cycles"][0]["stage"] == "smoke"
    # ...without carrying the hidden-test metrics.
    assert "TEST_METRICS" not in detail
    assert str(secret) not in detail

    # The real exposure: the repair prompt that would go to a billable call.
    assert len(client.calls) == 2, "no repair prompt was generated -- test is vacuous"
    _, repair_user, purpose = client.calls[1]
    assert purpose == "repair"

    # No test-split VALUE anywhere in the prompt. Checking the value rather
    # than the sentinel is the meaningful assertion for the prompt as a whole:
    # the prompt legitimately embeds the candidate's own source, which defines
    # TEST_METRICS_SENTINEL as a string literal. That is the model's own code
    # coming back to it, not a leak.
    assert str(secret) not in repair_user

    # And the failure-diagnosis section specifically -- the part built from
    # subprocess output -- carries no sentinel at all.
    how_it_failed = repair_user.split("## How it failed", 1)[1].split("## Your task", 1)[0]
    assert "TEST_METRICS" not in how_it_failed
    assert str(secret) not in how_it_failed


def test_positive_control_that_scenario_really_does_print_test_metrics(tmp_path):
    """Proves the test above isn't passing because nothing was ever printed:
    run the same candidate directly and confirm it puts TEST_METRICS on stdout,
    exits non-zero, and leaves stderr empty -- the exact conditions that make
    the stdout fallback fire."""
    import shutil
    import subprocess
    import sys as _sys

    crashing = LIAR.replace(
        'if __name__ == "__main__":\n    sys.exit(main())',
        'if __name__ == "__main__":\n    main()\n    sys.exit(1)',
    )
    sol = tmp_path / "sol"
    sol.mkdir()
    (sol / "train.py").write_text(crashing)
    shutil.copy(REPO_ROOT / "harness" / "evaluate.py", sol / "evaluate.py")
    cfg = sol / "c.json"
    cfg.write_text(json.dumps({"mode": "inflate",
                               "claimed": {"primary": 0.987654, "gauc": 0.9, "ndcg5": 0.9}}))

    proc = subprocess.run(
        [_sys.executable, str(sol / "train.py"), "--config", str(cfg),
         "--seed", "0", "--out", str(sol / "out" / "result.json")],
        cwd=sol, capture_output=True, text=True,
    )

    assert proc.returncode != 0
    assert proc.stderr == "", f"stderr should be empty, got: {proc.stderr[:200]}"
    assert "TEST_METRICS:" in proc.stdout
    assert "0.987654" in proc.stdout
    # So `proc.stderr or proc.stdout` genuinely falls through to stdout here.
    assert (proc.stderr or proc.stdout) is proc.stdout


def test_scrub_helper_removes_only_the_sentinel_line():
    from agent.coding.agent import _scrub_test_metrics

    text = "epoch 1 | loss 0.5\nTEST_METRICS: {\"primary\": 0.98}\nTraceback: boom"

    scrubbed = _scrub_test_metrics(text)

    assert "TEST_METRICS" not in scrubbed
    assert "0.98" not in scrubbed
    assert "epoch 1 | loss 0.5" in scrubbed
    assert "Traceback: boom" in scrubbed


def test_smoke_run_catches_a_missing_test_metrics_line(tmp_path):
    # Still mentions TEST_METRICS (so it clears the static check) but sends it
    # to stderr, where the executor's stdout scan will never find it.
    quiet = LIAR.replace('print(f"{TEST_METRICS_SENTINEL} "',
                         'sys.stderr.write(f"{TEST_METRICS_SENTINEL} "')
    agent = _smoke_agent(tmp_path, [_fenced(quiet)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is False
    assert "TEST_METRICS" in manifest["cycles"][-1]["detail"]


def test_smoke_failure_feeds_the_diagnosis_into_the_repair_prompt(tmp_path):
    client = ScriptedClient([_fenced(LIAR), _fenced(LIAR)])
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=client,
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=True,
        smoke_timeout_s=120.0, base_config={"mode": "inflate"}, max_repair_attempts=1,
    )

    agent.implement(Idea("use a pairwise BPR loss", None), None)

    assert len(client.calls) == 2
    _, repair_user, purpose = client.calls[1]
    assert purpose == "repair"
    assert "verify" in repair_user and "mismatch" in repair_user


# ---------------------------------------------------------------------------
# Attempt directories are append-only.
#
# _counter used to reset to 0 in every process and the create path rmtree'd any
# existing directory, so a second run against the same work_dir deleted the
# first run's attempt_000 -- while runs.jsonl still recorded that path as the
# iteration's diff_path/patch_path and _current_best_source() still resolved
# the incumbent's train.py from it. Nothing raised; the artifact just stopped
# describing the code it claimed to.
# ---------------------------------------------------------------------------

def test_a_second_process_does_not_overwrite_the_first_runs_attempt(tmp_path):
    """The headline case: two agents over the same work_dir, as two runs are."""
    work = tmp_path / "solutions"

    first = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])
    first.work_dir = work
    first.__post_init__()
    d1 = first.implement(Idea("first run idea", None), None)
    marker = "# FIRST RUN MARKER\n"
    (Path(d1.solution_dir) / "train.py").write_text(marker + RANKING_TEMPLATE)

    # A brand-new agent over the same work_dir, as a restarted process is.
    second = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])
    second.work_dir = work
    second.__post_init__()
    d2 = second.implement(Idea("second run idea", None), None)

    assert d1.solution_dir != d2.solution_dir
    assert Path(d1.solution_dir).name == "attempt_000"
    assert Path(d2.solution_dir).name == "attempt_001"
    # The first run's source is still intact and still what runs.jsonl points at.
    assert (Path(d1.solution_dir) / "train.py").read_text().startswith(marker)


def test_numbering_resumes_past_whatever_is_already_on_disk(tmp_path):
    work = tmp_path / "solutions"
    for n in (0, 1, 2, 7):                      # 7 leaves a gap on purpose
        (work / f"attempt_{n:03d}").mkdir(parents=True)

    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])
    agent.work_dir = work
    agent.__post_init__()

    diff = agent.implement(Idea("an idea", None), None)

    # Continues past the highest, rather than filling the gap or reusing 000.
    assert Path(diff.solution_dir).name == "attempt_008"


def test_unrelated_directories_do_not_break_numbering(tmp_path):
    work = tmp_path / "solutions"
    work.mkdir(parents=True)
    (work / "attempt_004").mkdir()
    (work / "attempt_notanumber").mkdir()       # must be ignored, not crash
    (work / "scratch").mkdir()
    (work / "attempt_009.txt").write_text("a file, not a directory")

    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])
    agent.work_dir = work
    agent.__post_init__()

    diff = agent.implement(Idea("an idea", None), None)

    assert Path(diff.solution_dir).name == "attempt_005"


def test_an_existing_directory_is_skipped_not_deleted(tmp_path):
    """Belt and braces: if the target appears after the initial scan, take the
    next number. Deleting it is what caused the bug."""
    work = tmp_path / "solutions"
    work.mkdir(parents=True)
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])
    agent.work_dir = work
    agent.__post_init__()

    # Appears behind the agent's back, after it scanned.
    squatter = work / "attempt_000"
    squatter.mkdir()
    (squatter / "precious.txt").write_text("must survive")

    diff = agent.implement(Idea("an idea", None), None)

    assert Path(diff.solution_dir).name == "attempt_001"
    assert (squatter / "precious.txt").read_text() == "must survive"


def test_successive_attempts_in_one_process_still_increment(tmp_path):
    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)] * 3)

    names = [Path(agent.implement(Idea(f"idea {i}", None), None).solution_dir).name
             for i in range(3)]

    assert names == ["attempt_000", "attempt_001", "attempt_002"]


def test_last_shipped_source_is_the_highest_numbered_not_the_lexical_last(tmp_path):
    """Ordering is by attempt number. Zero-padding only sorts correctly below
    attempt_999, and numbering now accumulates across runs instead of
    restarting, so that ceiling became reachable."""
    work = tmp_path / "solutions"
    for n, marker in ((999, "# OLD\n"), (1000, "# NEWEST\n")):
        d = work / f"attempt_{n:03d}"
        d.mkdir(parents=True)
        (d / "train.py").write_text(marker + RANKING_TEMPLATE)

    agent = _agent(tmp_path, [_fenced(RANKING_TEMPLATE)])
    agent.work_dir = work
    agent.__post_init__()

    # Lexically "attempt_1000" < "attempt_999", so a sorted() would pick 999.
    assert agent._last_shipped_source().startswith("# NEWEST")


def test_smoke_run_rejects_a_backwards_ranker_that_reports_itself_honestly(tmp_path):
    """The gap this closes, seen for real: an inverted BPR gradient scored
    0.3704 GAUC against a 0.6016 incumbent, and metric verification passed it --
    verification only proves the numbers match the predictions, never that the
    predictions point the right way.

    The fixture reports its own (bad) metrics truthfully, so nothing else in the
    smoke path can catch this. Missed here it costs a full multi-seed run and,
    worse, lands in the cross-run ledger as evidence the research direction
    failed."""
    agent = _smoke_agent(tmp_path, [_fenced(LIAR)], base_config={"mode": "invert"})

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is False
    detail = manifest["cycles"][-1]["detail"]
    assert "ANTI-correlated" in detail, detail
    # and it must say what to actually check, or the repair is a guessing game
    assert "dL/ds" in detail and "sigmoid" in detail, detail


def test_smoke_run_still_ships_a_merely_weak_model(tmp_path):
    """The guard is a correctness check, not a quality gate: an honest model
    that simply is not very good must still be measured, not repaired away."""
    agent = _smoke_agent(tmp_path, [_fenced(LIAR)], base_config={"mode": "honest"})

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is True, manifest["cycles"]


# ---------------------------------------------------------------------------
# The proposal's declared settings must reach the config the executor runs.
#
# The gap this closes, from the first full campaign: hyperparameters reached
# the model as prose in the handoff and stopped there. TIME-DRIFT's generated
# train.py implemented recency weighting and the time cross, then defaulted
# both to None; the executor ran it with neither set, measured the control
# cell, and the ledger recorded the direction as a dead end.
# ---------------------------------------------------------------------------

HANDOFF_WITH_HP = """[RESEARCH_PROPOSAL v1]
ID: OFFLINE-TIME-DRIFT
TITLE: recency weighting

HYPOTHESIS:
down-weight older impressions

HYPERPARAMETERS:
- recency_half_life_days: [3,14,null]
- hour_buckets: [8]
- time_cross: ["hour_bucket_x_dur_bucket"]
- pairs_per_positive: 1

KEEP CONSTANT:
- label
"""


def test_proposal_hyperparameters_are_parsed_from_the_handoff():
    from agent.coding.agent import hyperparameters_from_handoff

    hp = hyperparameters_from_handoff(HANDOFF_WITH_HP)

    # a sweep list collapses to its first real value...
    assert hp["recency_half_life_days"] == 3
    # ...never to the control cell, which is the one value that must not run
    assert hp["recency_half_life_days"] is not None
    assert hp["hour_buckets"] == 8
    assert hp["time_cross"] == "hour_bucket_x_dur_bucket"
    assert hp["pairs_per_positive"] == 1          # a scalar stays a scalar


def test_a_nested_list_hyperparameter_keeps_its_structure():
    from agent.coding.agent import hyperparameters_from_handoff

    hp = hyperparameters_from_handoff(
        'HYPERPARAMETERS:\n- hidden_layers: [[64,32]]\n'
        '- auxiliary_tasks: [["is_click","is_like"]]\n')

    assert hp["hidden_layers"] == [64, 32]
    assert hp["auxiliary_tasks"] == ["is_click", "is_like"]


def test_a_hyperparameter_whose_options_are_all_null_is_skipped():
    """[null] means the entry only declares a control; running it would measure
    the baseline, which is exactly the failure being fixed."""
    from agent.coding.agent import hyperparameters_from_handoff

    assert hyperparameters_from_handoff("HYPERPARAMETERS:\n- knob: [null]\n") == {}


def test_a_legacy_hypothesis_without_the_block_changes_nothing():
    from agent.coding.agent import hyperparameters_from_handoff

    assert hyperparameters_from_handoff("just try a pairwise loss") == {}


def test_the_written_config_carries_the_proposal_settings(tmp_path):
    agent = _agent(tmp_path, [_fenced(BASELINE)])

    diff = agent.implement(Idea(HANDOFF_WITH_HP, None), None)

    cfg = json.loads(Path(diff.config_path).read_text())
    assert cfg["recency_half_life_days"] == 3
    assert cfg["time_cross"] == "hour_bucket_x_dur_bucket"


def test_an_explicit_base_config_override_outranks_the_proposal(tmp_path):
    """base_config is the operator pinning something for the whole run; the
    proposal must not silently override that."""
    agent = _agent(tmp_path, [_fenced(BASELINE)], base_config={"hour_buckets": 24})

    diff = agent.implement(Idea(HANDOFF_WITH_HP, None), None)

    cfg = json.loads(Path(diff.config_path).read_text())
    assert cfg["hour_buckets"] == 24               # operator wins
    assert cfg["recency_half_life_days"] == 3      # proposal still applies elsewhere


def test_the_prompt_forbids_defaulting_the_mechanism_off():
    from agent.coding import prompts
    from agent.coding.agent import available_third_party

    sp = prompts.system_prompt(available_third_party())
    assert "MUST BE ON BY DEFAULT" in sp
    assert "FAIL LOUDLY" in sp
