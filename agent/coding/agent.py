"""LLMCodingAgent -- a real implementation of the CodingAgent Protocol.

    implement(idea, feedback) -> Diff

Turns one hypothesis into a solution dir the existing executor can run as-is:
a generated `train.py`, a `config.yaml`, and copies of the vendored
`evaluate.py` / `data.py` so the dir is self-contained under the executor's
`cwd=<solution_dir>` subprocess.

THE INNER LOOP, AND WHY IT EXISTS ALONGSIDE THE ORCHESTRATOR'S
--------------------------------------------------------------
The orchestrator already retries: up to `max_fix_attempts` (3) attempts per
idea, passing the previous failure back in as `feedback`. This agent does NOT
duplicate that cap. It runs a *different*, much cheaper loop underneath it:

    generate -> static checks -> smoke run on a small data subsample -> repair

An orchestrator-level attempt costs a full multi-seed training run, and there
are only three of them before the idea is abandoned. Burning one on a NameError
is waste. The smoke run trains on ~20k rows for one epoch, so a syntactically
broken or contract-violating train.py is caught in seconds and repaired without
consuming an orchestrator attempt at all. What reaches the executor has already
been shown to start, train, score, and produce a verifiable result.json.

`feedback` from the orchestrator is used, not ignored: when it is not None this
is a retry of an idea whose *full* run failed, so the last attempt's source and
that failure are both put in the repair prompt.

WHAT `Diff.diff_path` MEANS HERE -- WORTH A LOOK IN REVIEW
----------------------------------------------------------
agents.py documents `Diff.diff_path` as "where the change is recorded (patch
file, commit ref, ...)", but orchestrator.py passes it straight to
`Executor.run_seeds(...)` as the config path. The executable meaning wins, so
diff_path is the config file. A real unified diff against the baseline is
still written, as `changes.patch` in the solution dir, and referenced from
`attempt.json` -- it just can't live in the field named after it. Renaming the
field, or giving Diff a separate `config_path`, would be the clean fix and
needs an orchestrator.py change.
"""
from __future__ import annotations

import ast
import difflib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agent.agents import Diff, Idea
from agent.coding import prompts
from agent.coding.llm import LLMClient, LLMResponse, UsageLog, default_client
from agent.verification import Status as VerifyStatus, verify_result

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDORED = REPO_ROOT / "harness"
VENDORED_FILES = ("evaluate.py", "data.py")

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# Modules a solution may import. Everything else is rejected before it costs a
# training run -- the environment has numpy and the stdlib, nothing more.
ALLOWED_IMPORTS = frozenset({
    "argparse", "collections", "copy", "csv", "dataclasses", "enum", "functools",
    "heapq", "itertools", "json", "math", "os", "pathlib", "random", "re",
    "statistics", "sys", "time", "typing", "warnings",
    "numpy", "yaml",
    "evaluate", "data",   # the vendored harness, sitting next to train.py
})

# Named separately from "not in ALLOWED_IMPORTS" so the failure message can say
# *why* -- "torch isn't installed in this environment" is a far more actionable
# repair prompt than "unexpected import".
KNOWN_UNAVAILABLE = {
    "torch": "PyTorch is not installed; the environment is numpy-only",
    "pandas": "pandas is not installed; use csv/numpy via the vendored data.py",
    "sklearn": "scikit-learn is not installed; implement the metric-free parts yourself",
    "scipy": "scipy is not installed; numpy only",
    "lightgbm": "lightgbm is not installed; numpy only",
    "xgboost": "xgboost is not installed; numpy only",
    "requests": "there is no network access in the sandbox",
    "urllib": "there is no network access in the sandbox",
    "socket": "there is no network access in the sandbox",
    "subprocess": "spawning subprocesses is not permitted",
}

FORBIDDEN_RESULT_KEYS = ("test_primary", "test_gauc", "test_ndcg5", "test_metrics", "hidden_test")


class CodeExtractionError(RuntimeError):
    pass


@dataclass
class AttemptOutcome:
    """One generate-or-repair cycle inside implement()."""
    ok: bool
    stage: str          # "extract" | "static" | "smoke" | "verify" | "done"
    detail: str
    source: Optional[str] = None


def extract_code(text: str) -> str:
    """Pull the python file out of a model response.

    Takes the LONGEST fenced block, not the first: models sometimes open with a
    short illustrative snippet before the real file, and taking the first block
    silently ships the snippet.
    """
    blocks = CODE_FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    stripped = text.strip()
    if stripped.startswith(("import ", "from ", '"""', "#!")):
        return stripped + "\n"      # unfenced but plausibly a whole file
    raise CodeExtractionError(
        "no ```python code block in the response; first 300 chars: " + stripped[:300]
    )


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def static_check(source: str) -> list[str]:
    """Cheap contract violations, found without running anything.

    Every one of these would otherwise surface only after a full training run
    -- minutes of CPU and, at the orchestrator level, one of three attempts.
    """
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"train.py does not parse: line {e.lineno}: {e.msg}"]

    for root in sorted(_imported_roots(tree)):
        if root in KNOWN_UNAVAILABLE:
            problems.append(f"imports {root!r}: {KNOWN_UNAVAILABLE[root]}")
        elif root not in ALLOWED_IMPORTS:
            problems.append(
                f"imports {root!r}, which is not available. Allowed: numpy, the stdlib, "
                f"and the vendored `evaluate`/`data` modules."
            )

    for flag in ("--config", "--seed", "--out"):
        if flag not in source:
            problems.append(f"never references the required CLI flag {flag}")

    if "TEST_METRICS" not in source:
        problems.append(
            "never prints the `TEST_METRICS: {...}` stdout line carrying the test-split metrics"
        )
    if "val_predictions" not in source:
        problems.append(
            "never writes val_predictions.npz. The executor re-scores those arrays to verify "
            "result.json; without them the run cannot be verified."
        )

    calls_evaluate = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "evaluate"
        for n in ast.walk(tree)
    )
    if not calls_evaluate:
        problems.append("never calls evaluate() -- metrics must come from the vendored evaluate.py")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("evaluate", "auc", "ndcg_at_k"):
            problems.append(
                f"defines its own {node.name}() -- evaluate.py is the sole scoring authority "
                f"and must be imported, not reimplemented. The executor re-scores and will "
                f"fail the run on any disagreement."
            )

    for key in FORBIDDEN_RESULT_KEYS:
        if f'"{key}"' in source or f"'{key}'" in source:
            problems.append(
                f"mentions the key {key!r}. Test-split numbers may appear ONLY on the "
                f"TEST_METRICS stdout line, never in result.json."
            )

    return problems


@dataclass
class LLMCodingAgent:
    """A real CodingAgent. Satisfies agent.agents.CodingAgent structurally."""

    work_dir: Path
    data_dir: Optional[str] = None
    llm: Optional[LLMClient] = None
    usage_log_path: Optional[Path] = None

    # Inner repair loop. Independent of (and much cheaper than) the
    # orchestrator's max_fix_attempts -- see the module docstring.
    max_repair_attempts: int = 2

    # Smoke run: enough to prove it starts, trains, scores and verifies.
    run_smoke_test: bool = True
    smoke_max_train_rows: int = 20_000
    smoke_epochs: int = 1
    smoke_timeout_s: float = 600.0

    base_config: dict[str, Any] = field(default_factory=dict)

    _counter: int = field(default=0, init=False)
    last_usage: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.llm is None:
            self.llm = default_client()
        if self.usage_log_path is None:
            self.usage_log_path = REPO_ROOT / "logs" / "coding_agent_usage.jsonl"
        self.usage = UsageLog(Path(self.usage_log_path))

    # -- the Protocol method -------------------------------------------------

    def implement(self, idea: Idea, feedback: Optional[str]) -> Diff:
        """Always returns a Diff, even when every repair failed.

        Returning rather than raising is deliberate: the orchestrator has no
        try/except around this call, and a raise would take down the whole run
        instead of being recorded as one failed iteration. A solution dir that
        is known-bad still gets handed over, the executor fails it honestly,
        and tier-1 retry/abandonment does its job.
        """
        t0 = time.time()
        n = self._counter
        self._counter += 1
        sol_dir = self.work_dir / f"attempt_{n:03d}"
        if sol_dir.exists():
            shutil.rmtree(sol_dir)
        sol_dir.mkdir(parents=True)
        self._scaffold(sol_dir)

        baseline = prompts.load_baseline_source()
        system = prompts.SYSTEM_PROMPT
        calls: list[LLMResponse] = []
        history: list[AttemptOutcome] = []
        source: Optional[str] = None

        if feedback:
            # A retry from the orchestrator: an earlier full run of this idea
            # failed. Repair the last thing we shipped rather than starting over.
            previous = self._last_shipped_source() or baseline
            user = prompts.build_repair_prompt(
                idea.hypothesis, previous, prompts.format_failure_feedback(feedback)
            )
            purpose = "repair_after_orchestrator_feedback"
        else:
            user = prompts.build_generate_prompt(idea.hypothesis, baseline, prompts.load_ideas())
            purpose = "generate"

        for attempt in range(self.max_repair_attempts + 1):
            response = self.llm.complete(system, user, purpose=purpose)
            calls.append(response)
            self.usage.record(response, purpose=purpose, idea=idea.hypothesis, attempt=attempt)

            outcome = self._try_candidate(response.text, sol_dir)
            history.append(outcome)
            if outcome.ok:
                source = outcome.source
                break

            source = outcome.source or source
            if attempt == self.max_repair_attempts:
                break
            user = prompts.build_repair_prompt(
                idea.hypothesis, source or baseline, f"[{outcome.stage}] {outcome.detail}"
            )
            purpose = "repair"

        # Even a failed candidate is written out: the executor should be the
        # thing that records the failure, in a RunRecord, on the audit trail.
        if source is not None:
            (sol_dir / "train.py").write_text(source)
            self._write_patch(sol_dir, baseline, source)

        config_path = self._write_config(sol_dir)
        self.last_usage = {
            "tokens_in": sum(c.tokens_in for c in calls),
            "tokens_out": sum(c.tokens_out for c in calls),
            "cost_usd": round(sum(c.cost_usd for c in calls), 6),
            "llm_calls": len(calls),
            "real_model_calls": sum(1 for c in calls if c.is_real_model_call),
        }
        self._write_manifest(sol_dir, idea, feedback, history, calls, time.time() - t0)
        return Diff(diff_path=str(config_path), solution_dir=str(sol_dir))

    # -- internals -----------------------------------------------------------

    def _scaffold(self, sol_dir: Path) -> None:
        """Copy the vendored harness in so the dir is self-contained.

        The executor runs train.py with cwd=<solution_dir>, so sys.path[0] is
        this directory and `from evaluate import evaluate` resolves to the
        hash-pinned copy -- no sys.path manipulation in generated code, and no
        dependence on the starter kit still existing at a fixed path.
        """
        for name in VENDORED_FILES:
            shutil.copy(VENDORED / name, sol_dir / name)

    def _try_candidate(self, response_text: str, sol_dir: Path) -> AttemptOutcome:
        try:
            source = extract_code(response_text)
        except CodeExtractionError as e:
            return AttemptOutcome(False, "extract", str(e))

        if response_text.strip().startswith("NO_TEMPLATE"):
            return AttemptOutcome(False, "extract", response_text.strip(), source=None)

        problems = static_check(source)
        if problems:
            return AttemptOutcome(
                False, "static",
                "the generated train.py violates the contract:\n"
                + "\n".join(f"  - {p}" for p in problems),
                source=source,
            )

        (sol_dir / "train.py").write_text(source)
        if not self.run_smoke_test:
            return AttemptOutcome(True, "done", "static checks passed (smoke test disabled)", source)

        return self._smoke(sol_dir, source)

    def _smoke(self, sol_dir: Path, source: str) -> AttemptOutcome:
        """Train on a small subsample for one epoch and check the contract for
        real: does it start, does it write a parseable result.json, do the
        metrics survive re-scoring, is the TEST_METRICS line there."""
        smoke_dir = sol_dir / "_smoke"
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
        smoke_dir.mkdir()
        cfg = dict(self._config_dict())
        cfg.update({
            "max_train_rows": self.smoke_max_train_rows,
            "epochs": self.smoke_epochs,
            "patience": 1,
        })
        cfg_path = smoke_dir / "smoke_config.json"
        cfg_path.write_text(json.dumps(cfg))
        out_path = smoke_dir / "result.json"

        try:
            proc = subprocess.run(
                [sys.executable, str(sol_dir / "train.py"), "--config", str(cfg_path),
                 "--seed", "0", "--out", str(out_path)],
                cwd=sol_dir, capture_output=True, text=True, timeout=self.smoke_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return AttemptOutcome(
                False, "smoke",
                f"the smoke run (1 epoch on {self.smoke_max_train_rows} rows) did not finish "
                f"within {self.smoke_timeout_s:.0f}s. A full run trains on 1.14M rows for up "
                f"to 40 epochs, so this has to be fast. Vectorise with numpy -- no Python "
                f"loops over rows.",
                source=source,
            )

        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-40:])
            return AttemptOutcome(False, "smoke", f"the smoke run crashed:\n{tail}", source=source)

        if not out_path.exists():
            return AttemptOutcome(
                False, "smoke", "the smoke run exited 0 but wrote no result.json at --out",
                source=source,
            )
        try:
            claimed = json.loads(out_path.read_text())
        except json.JSONDecodeError as e:
            return AttemptOutcome(False, "smoke", f"result.json is not valid JSON: {e}", source=source)

        missing = {"primary", "gauc", "ndcg5", "epochs_run"} - set(claimed)
        if missing:
            return AttemptOutcome(
                False, "smoke",
                f"result.json is missing required keys: {sorted(missing)}. "
                f"It had: {sorted(claimed)}",
                source=source,
            )

        leaked = [k for k in claimed if k.lower() in FORBIDDEN_RESULT_KEYS]
        if leaked:
            return AttemptOutcome(
                False, "smoke",
                f"result.json contains test-split key(s) {leaked}. Test metrics belong ONLY "
                f"on the TEST_METRICS stdout line.",
                source=source,
            )

        if "TEST_METRICS:" not in proc.stdout:
            return AttemptOutcome(
                False, "smoke",
                "no `TEST_METRICS: {...}` line on stdout. The executor quarantines that line; "
                "without it the run has no hidden-test record.",
                source=source,
            )

        outcome = verify_result(smoke_dir, claimed)
        if outcome.status != VerifyStatus.OK:
            return AttemptOutcome(
                False, "verify",
                f"metric verification came back {outcome.status.value}: {outcome.detail}",
                source=source,
            )

        return AttemptOutcome(
            True, "done",
            f"smoke run passed: primary={claimed['primary']:.4f} on a "
            f"{self.smoke_max_train_rows}-row subsample, metrics verified",
            source=source,
        )

    def _config_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if self.data_dir:
            cfg["data_dir"] = str(self.data_dir)
        cfg.update(self.base_config)
        return cfg

    def _write_config(self, sol_dir: Path) -> Path:
        """Config for the real run.

        Deliberately minimal: the generated train.py is required to default
        every hyperparameter it invents, so this only pins what the *harness*
        needs to know (where the data is), plus any explicit overrides. That
        keeps the model from having to emit two files in sync.
        """
        path = sol_dir / "config.json"
        path.write_text(json.dumps(self._config_dict(), indent=2))
        return path

    def _write_patch(self, sol_dir: Path, baseline: str, source: str) -> None:
        diff = difflib.unified_diff(
            baseline.splitlines(keepends=True), source.splitlines(keepends=True),
            fromfile="a/solution/train.py", tofile="b/train.py",
        )
        (sol_dir / "changes.patch").write_text("".join(diff))

    def _last_shipped_source(self) -> Optional[str]:
        prior = sorted(self.work_dir.glob("attempt_*/train.py"))
        return prior[-1].read_text() if prior else None

    def _write_manifest(self, sol_dir, idea, feedback, history, calls, wall_s) -> None:
        (sol_dir / "attempt.json").write_text(json.dumps({
            "hypothesis": idea.hypothesis,
            "parent_iteration": idea.parent_iteration,
            "had_orchestrator_feedback": feedback is not None,
            "wall_s": round(wall_s, 2),
            "succeeded": bool(history and history[-1].ok),
            "patch": "changes.patch",
            "cycles": [
                {"stage": h.stage, "ok": h.ok, "detail": h.detail[:2000]} for h in history
            ],
            "usage": self.last_usage,
            "model": calls[0].model if calls else None,
        }, indent=2))
