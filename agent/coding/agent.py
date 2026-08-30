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

WHAT THIS RETURNS
-----------------
A `Diff` with `config_path` (what the executor runs), `solution_dir`,
`patch_path` (a real unified diff against the source this was built from), and
`usage` (tokens/cost for this implement() call, which orchestrator.py folds
into RunRecord.resources).

`Diff` used to have a single `diff_path` documented as a patch file but
consumed by orchestrator.py as the config path. That is now split into two
explicitly named fields and the ambiguous name is gone. Note that
RunRecord.diff_path (records.py) still holds the CONFIG path -- see
Orchestrator._record_diff_path for why that was left alone, and why
_current_best_source() below depends on it.
"""
from __future__ import annotations

import ast
import importlib.util
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

from agent.agents import AgentUsage, Diff, Idea
from agent.coding import prompts
from agent.config import TEST_METRICS_SENTINEL
from agent.coding.llm import LLMClient, LLMResponse, UsageLog, default_client
from agent.verification import Status as VerifyStatus, verify_result

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDORED = REPO_ROOT / "harness"
VENDORED_FILES = ("evaluate.py", "data.py")

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# Modules a solution may import. Everything else is rejected before it costs a
# training run -- the environment has numpy and the stdlib, nothing more.
# IMPORT POLICY
# -------------
# External open-source libraries and frameworks are PERMITTED -- see
# docs/coding-agent.md, which is the authoritative statement of this. What used
# to sit here was a hardcoded numpy-only allowlist plus a hand-maintained
# "unavailable" dict, and both had drifted from reality: the dict claimed
# pandas, sklearn and scipy were "not installed" when all three are, while the
# allowlist rejected them anyway.
#
# So availability is now MEASURED rather than declared. Two rules:
#
#   1. FORBIDDEN_IMPORTS -- refused however the environment is provisioned,
#      because they are sandbox rules rather than packaging facts.
#   2. everything else -- allowed if and only if it can actually be imported
#      here. That is exactly docs/coding-agent.md's rule that a dependency must
#      be installed before a solution may rely on it, and it needs no edit when
#      one is later provisioned: pip install torch, and torch becomes legal.
#
# Measuring in THIS process is valid because executor.py launches solutions with
# sys.executable -- the same interpreter -- so what imports here imports there.

# Sit next to the generated train.py, so they are importable in the executor's
# cwd=<solution_dir> subprocess but not from this process. Availability cannot
# be measured for them; they are always legal.
VENDORED_IMPORTS = frozenset({"evaluate", "data"})

# Sandbox rules. These stay refused no matter what is installed -- every one is
# in the stdlib and would therefore pass an availability test.
FORBIDDEN_IMPORTS = {
    "subprocess": "spawning subprocesses is not permitted",
    "socket": "there is no network access in the sandbox",
    "requests": "there is no network access in the sandbox",
    "urllib": "there is no network access in the sandbox",
}


def module_available(root: str) -> bool:
    """Whether `root` can actually be imported in the interpreter that will run
    the solution.

    find_spec, not import: it resolves the module without executing it, so
    probing a heavy framework costs nothing and cannot run its side effects.
    Any failure is treated as unavailable -- a module whose parent package
    explodes on lookup is not one a generated solution should depend on.
    """
    if root in VENDORED_IMPORTS:
        return True
    try:
        return importlib.util.find_spec(root) is not None
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        return False


def available_third_party() -> list[str]:
    """Installed, non-stdlib libraries a solution may use, for the prompt.

    Naming what IS available matters as much as rejecting what is not: without
    it the model has to guess, and guesses cost a whole generate/check cycle.
    """
    candidates = (
        "numpy", "scipy", "pandas", "sklearn", "torch", "jax", "tensorflow",
        "lightgbm", "xgboost", "numba",
    )
    return [name for name in candidates if module_available(name)]


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


def _scrub_test_metrics(text: str) -> str:
    """Drop any TEST_METRICS: line from captured subprocess output.

    Hidden-test-split metrics reach stdout so executor.py can quarantine them.
    Anything this module builds out of that output becomes
    AttemptOutcome.detail, which goes into a repair prompt and from there into
    a real, billable LLM call -- so it has to be scrubbed BEFORE it is
    truncated into a tail, not after. Truncating first can keep the sentinel
    line and drop everything else.

    Mirrors executor.py's _strip_test_metrics_lines, which does this for its
    own traceback tails. Duplicated rather than imported: that helper is
    private to the executor and the coding agent should not reach into another
    module's internals. The sentinel itself comes from agent.config, so there
    is still exactly one definition of what the line looks like.
    """
    return "\n".join(
        line for line in text.splitlines()
        if not line.strip().startswith(TEST_METRICS_SENTINEL)
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
        if root in FORBIDDEN_IMPORTS:
            problems.append(f"imports {root!r}: {FORBIDDEN_IMPORTS[root]}")
        elif not module_available(root):
            # Caught here rather than at runtime on purpose: an ImportError
            # inside the executor costs a full multi-seed training run and
            # surfaces as a CRASH traceback, where this costs nothing and says
            # what to do instead.
            problems.append(
                f"imports {root!r}, which is not installed in this environment. "
                f"External libraries are permitted, but only once provisioned. "
                f"Installed and available: {', '.join(available_third_party())}, "
                f"plus the stdlib and the vendored `evaluate`/`data` modules."
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

    # ACCUMULATION: where to look up "the current best solution" so a new idea
    # builds on what has been accepted rather than always on the static
    # solution/train.py. Both default to None, which keeps exactly the old
    # behaviour for callers (and tests) that don't pass them.
    registry_path: Optional[Path] = None
    run_log_path: Optional[Path] = None

    _counter: int = field(default=0, init=False)
    last_usage: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Resume numbering after whatever is already on disk. _counter used to
        # start at 0 in every new process, so a second run against the same
        # work_dir reused attempt_000 -- see _next_solution_dir.
        self._counter = self._highest_existing_attempt() + 1
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
        sol_dir = self._next_solution_dir()
        self._scaffold(sol_dir)

        baseline, provenance = self._current_best_source()
        # Rendered per call against what is actually importable, so the model is
        # told the real environment rather than a constant that can drift from it.
        system = prompts.system_prompt(available_third_party())
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
        self._write_manifest(sol_dir, idea, feedback, history, calls, time.time() - t0,
                             provenance=provenance)
        # Hand usage back so orchestrator.py can put it in RunRecord.resources.
        # This is per-implement() totals -- one iteration's worth -- which is
        # the granularity a RunRecord wants. It includes the inner repair
        # cycles, since those are part of what this iteration cost.
        return Diff(
            config_path=str(config_path),
            solution_dir=str(sol_dir),
            patch_path=str(sol_dir / "changes.patch") if source is not None else None,
            usage=AgentUsage(
                tokens_in=self.last_usage["tokens_in"],
                tokens_out=self.last_usage["tokens_out"],
                cost_usd=self.last_usage["cost_usd"],
            ),
        )

    # -- internals -----------------------------------------------------------

    def _current_best_source(self) -> tuple[str, str]:
        """The source a new idea should be built from, plus a provenance label.

        Without this the agent restarted from the static solution/train.py on
        every idea, so improvements never compounded: iteration 5 was built on
        the baseline, not on whatever iterations 1-4 had established.

        The lookup chains three things the harness already maintains:

            registry.best()      -> the accepted iteration with the best
                                    validation primary
            runs.jsonl           -> that iteration's RunRecord
            record.diff_path     -> the config the executor was pointed at,
                                    whose sibling train.py is the source that
                                    actually produced the score

        Note this reads RunRecord.diff_path, which orchestrator.py populates
        from the config path -- see the naming discussion in this module's
        docstring. The adjacency of config and train.py inside a solution dir
        is what makes the last hop work, and both the CodingAgent's own
        attempt dirs and the seeded solution/ satisfy it.

        Every failure mode falls back to the static baseline rather than
        raising: a missing registry, an empty one, a record that can't be
        found, a solution dir that has been cleaned up. implement() must not
        crash the whole run because provenance couldn't be resolved.
        """
        fallback = (prompts.load_baseline_source(), "solution/train.py (static baseline)")
        if self.registry_path is None or self.run_log_path is None:
            return fallback

        try:
            from agent.records import RunLog
            from agent.registry import CheckpointRegistry

            best = CheckpointRegistry(Path(self.registry_path)).best()
            if best is None:
                return fallback

            record = next(
                (r for r in RunLog(Path(self.run_log_path)).read_all()
                 if r.iteration == best.iteration),
                None,
            )
            if record is None or not record.diff_path:
                return fallback

            train_py = Path(record.diff_path).parent / "train.py"
            if not train_py.exists():
                return fallback

            return (
                train_py.read_text(),
                f"iteration {best.iteration} (val primary {best.val_primary:.4f}) "
                f"via {train_py}",
            )
        except Exception as e:  # noqa: BLE001 - provenance is best-effort
            print(f"[coding-agent] could not resolve the current best, "
                  f"falling back to the static baseline: {type(e).__name__}: {e}", flush=True)
            return fallback

    ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")

    def _highest_existing_attempt(self) -> int:
        """Largest attempt number already in work_dir, or -1 if there are none.

        Parsed from the directory names rather than tracked in a file: the
        directories are the fact, and any sidecar counter could disagree with
        them after a manual move or a partial cleanup.
        """
        highest = -1
        for child in self.work_dir.iterdir():
            if not child.is_dir():
                continue
            m = self.ATTEMPT_RE.match(child.name)
            if m:
                highest = max(highest, int(m.group(1)))
        return highest

    def _next_solution_dir(self) -> Path:
        """Create and return a fresh attempt directory. Never reuses one.

        This previously took the next number from an in-memory counter that
        reset to 0 in every process, then `shutil.rmtree`'d the directory if it
        already existed. A second run against the same work_dir therefore
        deleted the first run's attempt_000 -- while runs.jsonl still recorded
        that path as the iteration's diff_path/patch_path, and
        _current_best_source() still resolved the incumbent's train.py from it.
        The graded artifact silently described code that was no longer there,
        and the Coding agent built on the wrong source. Nothing raised.

        So: numbering continues past whatever is on disk, and an existing
        directory is skipped rather than destroyed. Attempt directories are
        append-only, like the run log that points at them.
        """
        while True:
            sol_dir = self.work_dir / f"attempt_{self._counter:03d}"
            self._counter += 1
            try:
                sol_dir.mkdir(parents=True)
                return sol_dir
            except FileExistsError:
                # Raced, or appeared since __post_init__ scanned. Take the next.
                continue

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
            # Scrub before truncating. A candidate can print its TEST_METRICS
            # line and then exit non-zero WITHOUT raising -- sys.exit(1) leaves
            # stderr empty -- so `proc.stderr or proc.stdout` falls through to
            # stdout, and the last lines of stdout still carry the hidden-test
            # metrics. That detail becomes a repair prompt and goes out to a
            # live LLM call, which is exactly what the quarantine exists to
            # prevent.
            tail = "\n".join(
                _scrub_test_metrics(proc.stderr or proc.stdout).splitlines()[-40:]
            )
            return AttemptOutcome(
                False, "smoke",
                f"the smoke run crashed:\n{tail}" if tail.strip() else
                "the smoke run exited non-zero with no diagnostic output on stdout or stderr",
                source=source,
            )

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
        """The most recent attempt's source, by attempt NUMBER.

        Sorted numerically, not lexically: zero-padding only orders correctly
        below attempt_999, and numbering now accumulates across runs rather
        than restarting, so that ceiling is reachable in a way it wasn't before.
        """
        numbered = []
        for train_py in self.work_dir.glob("attempt_*/train.py"):
            m = self.ATTEMPT_RE.match(train_py.parent.name)
            if m:
                numbered.append((int(m.group(1)), train_py))
        if not numbered:
            return None
        return max(numbered)[1].read_text()

    def _write_manifest(self, sol_dir, idea, feedback, history, calls, wall_s,
                        provenance: str = "") -> None:
        (sol_dir / "attempt.json").write_text(json.dumps({
            "hypothesis": idea.hypothesis,
            # What this attempt was built ON -- the static baseline, or the
            # accepted iteration it accumulated from.
            "built_from": provenance,
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
