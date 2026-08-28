"""Independent re-scoring of a solution's claimed validation metrics.

WHY THIS EXISTS
---------------
The brief names `evaluate.py` as the sole scoring authority, but until now
nothing enforced that. `executor.py` read `result.json`, checked that
primary/gauc/ndcg5 were finite numbers, and believed them. A `train.py` that
reimplemented GAUC slightly wrong -- or, once an LLM is writing these files,
one that simply printed a number it never computed -- would sail through and
poison every downstream decision: `delta_vs_current_best`, the Evaluator's
ACCEPT/REVERT, the checkpoint registry, and the convergence rule all key off
`primary`, so a single fabricated metric doesn't just add noise, it can
permanently pin the "best so far" line to a model that was never that good.

So: `train.py` also persists the exact `(user_id, label, score)` arrays it fed
to `evaluate()`, next to `result.json` in the artifact dir executor.py already
creates. This module re-runs the *vendored, unmodified* `harness/evaluate.py`
over those arrays and cross-checks the answer against what result.json
claimed. A meaningful mismatch is a run failure
(`FailureKind.METRIC_MISMATCH`), not a silently-accepted result.

WHAT IT DOES AND DOESN'T CATCH
------------------------------
It catches: a reimplemented/miscomputed metric, a fabricated number, and
metrics computed over a different set of rows than the ones persisted.

It does NOT catch a solution that persists arrays consistent with a number it
scored on the wrong split -- e.g. one that trains on validation, or that
writes test-split arrays here. That is a different property (data hygiene, not
arithmetic honesty) and would need the executor to know the split definitions.
Flagged for whoever picks this up next.

BACKWARDS COMPATIBILITY
-----------------------
Verification is strictly opt-in from the solution's side: no raw-predictions
file means `Status.SKIPPED`, which the executor treats as "fine, carry on".
`fixtures/fake_train.py` and every existing test therefore keep passing
untouched. Only a solution that *does* persist arrays gets held to them.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# Filenames a train.py may use to persist its raw validation predictions, in
# preference order. Keep in sync with solution/train.py's RAW_PREDICTIONS_NAME
# and with the contract text in agent/coding/prompts.py.
RAW_PREDICTIONS_NAMES = ("val_predictions.npz", "val_predictions.json")

# Max absolute difference between a claimed metric and the independently
# recomputed one before the run is failed.
#
# This is float-noise tolerance, not a fudge factor. Recomputation is *nearly*
# bit-identical -- same evaluate.py, same arrays, same row order -- but not
# exactly: evaluate() sums the label array to get `npos`, so a train.py that
# scored float32 labels in-process and persisted them as float64 (which
# solution/train.py does) re-scores a hair differently. Measured on the real
# 124,909-row validation split, that residual is 7.6e-7 on primary.
#
# 1e-5 sits ~13x above that measured noise floor and far below anything that
# could matter: the FM baseline's seed-to-seed std is 8e-4 (80x this) and the
# convergence epsilon is 2e-3 (200x this). So nothing this tolerance hides can
# move a delta, a decision, or the stopping rule.
#
# It lives here rather than in agent/config.py because config.py is a shared,
# tested file this change is not authorised to touch; fold it in there when
# this lands. Override for an experiment with METRIC_VERIFICATION_TOLERANCE.
DEFAULT_TOLERANCE = 1e-5

# Metrics cross-checked, mapping result.json's key -> evaluate()'s key.
CHECKED_METRICS = {"primary": "primary", "gauc": "GAUC", "ndcg5": "nDCG@5"}

VENDORED_EVALUATE = Path(__file__).resolve().parent.parent / "harness" / "evaluate.py"


class Status(str, Enum):
    OK = "ok"                # recomputed and matched
    MISMATCH = "mismatch"    # recomputed and did NOT match -> run fails
    SKIPPED = "skipped"      # no raw-predictions file; solution opted out
    UNREADABLE = "unreadable"  # file present but unusable -> run fails


@dataclass(frozen=True)
class VerificationOutcome:
    status: Status
    detail: str
    claimed: Optional[dict[str, float]] = None
    recomputed: Optional[dict[str, float]] = None

    @property
    def failed(self) -> bool:
        return self.status in (Status.MISMATCH, Status.UNREADABLE)


def tolerance() -> float:
    raw = os.environ.get("METRIC_VERIFICATION_TOLERANCE")
    if not raw:
        return DEFAULT_TOLERANCE
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_TOLERANCE


def load_evaluate() -> Callable[..., dict]:
    """Import the vendored evaluate.py by path.

    By path, not by `sys.path` insertion: `harness/` is a verbatim copy of the
    starter kit, whose modules are flat top-level names (`evaluate`, `data`).
    Putting that directory on sys.path would shadow anything else called
    `data` in the parent process. This keeps it contained.
    """
    if not VENDORED_EVALUATE.exists():
        raise FileNotFoundError(f"vendored scorer missing at {VENDORED_EVALUATE}")
    cached = sys.modules.get("_vendored_evaluate")
    if cached is not None:
        return cached.evaluate
    spec = importlib.util.spec_from_file_location("_vendored_evaluate", VENDORED_EVALUATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_vendored_evaluate"] = module
    spec.loader.exec_module(module)
    return module.evaluate


def find_raw_predictions(out_dir: Path) -> Optional[Path]:
    for name in RAW_PREDICTIONS_NAMES:
        p = Path(out_dir) / name
        if p.exists():
            return p
    return None


def _load_arrays(path: Path) -> tuple[list, list, list]:
    """Returns (user_ids, labels, scores) as plain Python lists."""
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        return list(d["user_ids"]), list(d["labels"]), list(d["scores"])

    import numpy as np  # only needed for the .npz path
    with np.load(path, allow_pickle=False) as z:
        users = z["user_ids"]
        labels = z["labels"]
        scores = z["scores"]
    # evaluate() groups by user_id via a dict key, so numpy scalars would work,
    # but .tolist() makes the grouping keys plain str/int and the arithmetic
    # plain floats -- matching what a train.py passes in directly.
    return users.tolist(), labels.tolist(), scores.tolist()


def verify_result(out_dir: Path, claimed: dict[str, Any], tol: Optional[float] = None) -> VerificationOutcome:
    """Re-score `out_dir`'s persisted arrays and compare against `claimed`.

    Never raises: any problem becomes an UNREADABLE outcome, because this runs
    inside executor.py's already-uniform "always return a SeedMetrics" path.
    """
    tol = tolerance() if tol is None else tol
    path = find_raw_predictions(Path(out_dir))
    if path is None:
        return VerificationOutcome(
            Status.SKIPPED,
            f"no raw-predictions file ({'/'.join(RAW_PREDICTIONS_NAMES)}) in {out_dir}; "
            "metric verification skipped",
        )

    try:
        users, labels, scores = _load_arrays(path)
    except Exception as e:  # noqa: BLE001 - any read problem is the same outcome
        return VerificationOutcome(
            Status.UNREADABLE, f"could not read {path.name}: {type(e).__name__}: {e}"
        )

    if not (len(users) == len(labels) == len(scores)):
        return VerificationOutcome(
            Status.UNREADABLE,
            f"{path.name} arrays are ragged: user_ids={len(users)} "
            f"labels={len(labels)} scores={len(scores)}",
        )
    if not users:
        return VerificationOutcome(Status.UNREADABLE, f"{path.name} is empty")
    if any(s != s or math.isinf(s) for s in scores):
        return VerificationOutcome(Status.UNREADABLE, f"{path.name} contains NaN/Inf scores")

    try:
        recomputed_raw = load_evaluate()(users, labels, scores)
    except Exception as e:  # noqa: BLE001
        return VerificationOutcome(
            Status.UNREADABLE, f"vendored evaluate() failed on {path.name}: {type(e).__name__}: {e}"
        )

    recomputed = {ours: float(recomputed_raw[theirs]) for ours, theirs in CHECKED_METRICS.items()}
    claimed_f = {k: float(claimed[k]) for k in CHECKED_METRICS if k in claimed}

    bad = {
        k: (claimed_f[k], recomputed[k])
        for k in claimed_f
        if abs(claimed_f[k] - recomputed[k]) > tol
    }
    if bad:
        parts = ", ".join(f"{k}: claimed {c:.6f} vs recomputed {r:.6f} (diff {abs(c - r):.2e})"
                          for k, (c, r) in sorted(bad.items()))
        return VerificationOutcome(
            Status.MISMATCH,
            f"result.json disagrees with an independent re-score of {path.name} "
            f"through the vendored evaluate.py (tolerance {tol:g}): {parts}. "
            "Compute the reported metrics with evaluate.py over exactly the arrays "
            "you persist -- do not reimplement the metric or hardcode a number.",
            claimed=claimed_f,
            recomputed=recomputed,
        )

    return VerificationOutcome(
        Status.OK,
        f"metrics reproduced from {path.name} via vendored evaluate.py "
        f"({len(users)} rows, tolerance {tol:g})",
        claimed=claimed_f,
        recomputed=recomputed,
    )
