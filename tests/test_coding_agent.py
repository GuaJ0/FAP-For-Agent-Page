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
    ("torch", "PyTorch is not installed"),
    ("pandas", "pandas is not installed"),
    ("sklearn", "scikit-learn is not installed"),
    ("requests", "no network access"),
])
def test_static_check_rejects_unavailable_imports_with_a_reason(mod, hint):
    problems = static_check(f"import {mod}\n")
    assert any(hint in p for p in problems), problems


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
    agent = _agent(tmp_path, [_fenced("import torch\n" + BASELINE), _fenced(RANKING_TEMPLATE)])

    diff = agent.implement(Idea("use a pairwise BPR loss", None), None)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert [c["ok"] for c in manifest["cycles"]] == [False, True]
    assert "PyTorch is not installed" in manifest["cycles"][0]["detail"]
    assert manifest["succeeded"] is True


def test_the_repair_prompt_contains_the_diagnosis_and_the_broken_source(tmp_path):
    """A repair prompt without the previous source makes the model rewrite
    from scratch, which loses whatever it had already got right."""
    client = ScriptedClient([_fenced("import torch\n" + BASELINE), _fenced(RANKING_TEMPLATE)])
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=client,
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=False,
    )

    agent.implement(Idea("use a pairwise BPR loss", None), None)

    _, repair_user, purpose = client.calls[1]
    assert purpose == "repair"
    assert "PyTorch is not installed" in repair_user
    assert "import torch" in repair_user
    assert "COMPLETE corrected" in repair_user


def test_repairs_are_capped_and_the_last_attempt_is_still_shipped(tmp_path):
    """When every repair fails the agent must still return a Diff. Raising
    would kill the orchestrator's whole run; returning lets the executor
    record one honest failed iteration and tier-1 retry take over."""
    bad = _fenced("import torch\n" + BASELINE)
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
    agent = _agent(tmp_path, [_fenced("import torch\n" + BASELINE), _fenced(RANKING_TEMPLATE)])

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
