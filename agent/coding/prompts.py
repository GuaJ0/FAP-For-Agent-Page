"""Prompts for the Coding agent.

The system prompt is the contract. Everything the executor will later reject a
solution for is stated here as a hard rule, because a violation costs a full
training run to discover -- and, under the orchestrator's tier-1 policy, one of
only three attempts before the idea is abandoned. Cheap to say up front.

Two of the rules are non-obvious enough to be worth their prose:

  - result.json must contain validation metrics only. A model that "helpfully"
    adds a test-split score would trip executor.py's forbidden-key guard and
    fail the run outright.
  - the reported metrics must come from evaluate.py over exactly the arrays
    persisted to val_predictions.npz. executor.py re-scores them and fails on
    a mismatch, so a reimplemented metric is not a small stylistic deviation,
    it is a hard failure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEM_PROMPT = """\
You are the Coding agent in an automated ML research loop working on the \
KuaiRand-Pure within-user ranking task. You are given one hypothesis and you \
return one complete `train.py` that tests it.

TASK
  Rank each user's own logged impressions. Label is `long_view` (0/1).
  Metrics are GAUC and nDCG@5; the primary score is their mean.
  Ranking is *within* a user -- scores are never compared across users, so any
  term that is constant within a user cannot change the score.

ENVIRONMENT (hard constraints -- violating these fails the run)
  - Python 3.9+ with numpy. NOTHING ELSE. No torch, pandas, sklearn, scipy.
  - No network access. No downloads. No subprocesses.
  - Single CPU core. Your run is killed at the executor's timeout, so keep a
    full training run to a few minutes.
  - `evaluate.py` and `data.py` are already sitting next to your train.py.
    Import them as top-level modules: `from evaluate import evaluate` and
    `from data import load, encode, FIELDS`.

INVOCATION CONTRACT (exact -- the executor drives this)
  python train.py --config <path> --seed <int> --out <path/to/result.json>

  `--config` is a YAML or JSON file of flat key/value pairs. Read every
  hyperparameter from it with a sensible default; never hardcode a path.
  Resolve the data directory as: config `data_dir`, else the KUAIRAND_PATH
  environment variable, else "./KuaiRand-Pure/data".

OUTPUTS -- write all of these into the SAME directory as --out
  1. `--out` itself: JSON with exactly these keys, VALIDATION SPLIT ONLY:
        {"primary": float, "gauc": float, "ndcg5": float, "epochs_run": int}
     Extra descriptive keys are fine. Cast numpy scalars with float()/int() --
     json.dumps refuses np.float32 and the resulting crash looks like a bug in
     your model rather than a serialisation slip.
     NEVER put a test-split number in this file, under any key name. The
     executor scans for that and fails the run.

  2. `val_predictions.npz`: the exact arrays you passed to evaluate() for the
     numbers in result.json.
        np.savez_compressed(out_dir / "val_predictions.npz",
                            user_ids=np.asarray(users),
                            labels=np.asarray(labels, dtype=np.float64),
                            scores=np.asarray(scores, dtype=np.float64))
     The executor independently re-scores these through evaluate.py and FAILS
     the run if they disagree with result.json by more than 1e-5. So:
     compute your reported metrics by calling evaluate() on exactly these
     arrays. Do not reimplement GAUC or nDCG. Do not round. Do not report a
     number from a different epoch than the arrays you save.

  3. `checkpoint.npz`: your best-epoch weights.

  4. On stdout, exactly one line:
        TEST_METRICS: {"primary": ..., "gauc": ..., "ndcg5": ...}
     with the TEST-split metrics. The executor intercepts this line and
     quarantines it; it is the ONLY place a test-split number may appear.

METHOD RULES
  - Select on validation, never on test. Compute test metrics once, at the
    end, from the model you already selected on validation.
  - Do not train on the validation or test split.
  - Early-stop on validation primary.
  - Change only what the hypothesis calls for. Everything else should stay as
    it is in the current baseline so the comparison is attributable.

OUTPUT FORMAT
  Return ONE python file in a single ```python fenced block. No commentary
  before or after it. It must run top to bottom as written -- no TODOs, no
  placeholders, no "..." elisions.
"""


def _read(path: Path, limit: int = 24_000) -> str:
    text = path.read_text()
    if len(text) > limit:
        text = text[:limit] + "\n# ... (truncated)\n"
    return text


def build_generate_prompt(
    hypothesis: str,
    baseline_source: str,
    ideas_md: Optional[str] = None,
) -> str:
    """`baseline_source` is whatever the agent resolved as the current best --
    the static solution/train.py early on, and the best accepted iteration's
    train.py once something has been accepted. See
    LLMCodingAgent._current_best_source()."""
    parts = [
        "## Hypothesis to implement\n",
        hypothesis.strip(),
        "\n\n## Current best train.py (the code you are modifying)\n",
        "```python\n" + baseline_source + "\n```\n",
    ]
    if ideas_md:
        parts += [
            "\n## Solution log -- what has already been tried\n",
            "Read the dead ends before choosing an approach; re-testing one wastes "
            "an iteration.\n\n",
            ideas_md,
        ]
    parts.append(
        "\n\n## Your task\n"
        "Return the complete `train.py` implementing the hypothesis above, honouring "
        "every rule in the system prompt. Keep everything the hypothesis does not "
        "require identical to the baseline."
    )
    return "".join(parts)


def build_repair_prompt(
    hypothesis: str,
    previous_source: str,
    failure: str,
) -> str:
    """Prompt for a retry.

    Carries the previous attempt's *source* as well as the error. Without it
    the model rewrites from scratch and tends to reintroduce whatever it just
    got right, so a two-line fix becomes a fresh set of bugs.
    """
    return (
        "## Hypothesis being implemented\n"
        f"{hypothesis.strip()}\n\n"
        "## Your previous attempt\n"
        "```python\n" + previous_source + "\n```\n\n"
        "## How it failed\n"
        "```\n" + failure.strip() + "\n```\n\n"
        "## Your task\n"
        "Diagnose the failure above and return the COMPLETE corrected `train.py` "
        "in one ```python block -- not a patch, not a diff, not just the changed "
        "function. Keep everything that was already working. Re-read the output "
        "and environment rules in the system prompt before answering: a mismatch "
        "between result.json and val_predictions.npz, an import outside numpy, or "
        "a test-split number in result.json will each fail the run again."
    )


def format_failure_feedback(feedback: str, limit: int = 4000) -> str:
    """The orchestrator's `feedback` argument, trimmed for a prompt.

    It arrives as "<failure_kind>: <traceback tail>". The tail is the useful
    half and the tail *end* is the useful half of that, so trim from the front.
    """
    feedback = feedback.strip()
    if len(feedback) <= limit:
        return feedback
    return "... (earlier output trimmed)\n" + feedback[-limit:]


def load_baseline_source(path: Optional[Path] = None) -> str:
    return _read(path or (REPO_ROOT / "solution" / "train.py"))


def load_ideas(path: Optional[Path] = None) -> Optional[str]:
    p = path or (REPO_ROOT / "solution" / "ideas.md")
    return _read(p, limit=8000) if p.exists() else None
