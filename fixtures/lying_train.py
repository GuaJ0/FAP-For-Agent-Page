"""A train.py that can be told to misreport its own metrics.

Exists to prove the scoring-integrity check in executor.py actually bites.
`fixtures/fake_train.py` is the honest stand-in for the *invocation* contract;
this one is the adversary for the *scoring* contract.

    python lying_train.py --config <cfg> --seed <n> --out <result.json>

`<cfg>` is JSON, with:

    mode: one of ("invert" ranks backwards: sub-random GAUC, honest reporting)
      "honest"    -- persist predictions, report exactly what evaluate.py says
      "inflate"   -- persist predictions, report `claimed` instead (the case
                     that matters: well-formed result.json, false numbers)
      "no_raw"    -- report honestly but persist nothing; verification should
                     SKIP, not fail (this is the fixtures/fake_train.py case)
      "ragged"    -- persist arrays of mismatched length
      "nan_score" -- persist a NaN among the scores
    claimed: {"primary":…, "gauc":…, "ndcg5":…}  used by "inflate"
    n_users / n_per_user / raw_format ("npz" | "json")

Generates a small deterministic synthetic ranking problem -- no KuaiRand data,
so this stays a fast unit-test fixture. It scores that problem with the real
vendored evaluate.py, which the test copies in next to this file exactly the
way the CodingAgent copies it into a real solution dir.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from evaluate import evaluate  # vendored copy, sitting next to this file

TEST_METRICS_SENTINEL = "TEST_METRICS:"
RAW_PREDICTIONS_NPZ = "val_predictions.npz"
RAW_PREDICTIONS_JSON = "val_predictions.json"


def synth(seed, n_users, n_per_user):
    """A ranking problem with real signal, so GAUC/nDCG aren't degenerate."""
    rng = random.Random(seed)
    users, labels, scores = [], [], []
    for u in range(n_users):
        for _ in range(n_per_user):
            y = 1 if rng.random() < 0.4 else 0
            users.append(f"u{u}")
            labels.append(y)
            scores.append(rng.gauss(1.0 if y else 0.0, 1.0))
    return users, labels, scores


def write_raw(out_dir, users, labels, scores, fmt, mode):
    if mode == "ragged":
        scores = scores[:-1]
    elif mode == "nan_score":
        scores = list(scores)
        scores[0] = float("nan")

    if fmt == "json":
        (out_dir / RAW_PREDICTIONS_JSON).write_text(json.dumps(
            {"user_ids": users, "labels": labels, "scores": scores}
        ))
    else:
        import numpy as np
        np.savez_compressed(
            out_dir / RAW_PREDICTIONS_NPZ,
            user_ids=np.asarray(users),
            labels=np.asarray(labels, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    mode = cfg.get("mode", "honest")
    fmt = cfg.get("raw_format", "npz")

    users, labels, scores = synth(a.seed, cfg.get("n_users", 40), cfg.get("n_per_user", 12))
    if mode == "invert":
        # Rank every user's items backwards. The reported metrics stay perfectly
        # consistent with the persisted predictions, so verification passes --
        # this is exactly the failure metric verification cannot see, and what
        # the sub-random GAUC check exists for.
        scores = [-s for s in scores]
    truth = evaluate(users, labels, scores)

    out_path = Path(a.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode != "no_raw":
        write_raw(out_dir, users, labels, scores, fmt, mode)

    if mode == "inflate":
        claimed = cfg.get("claimed", {"primary": 0.95, "gauc": 0.97, "ndcg5": 0.93})
    else:
        claimed = {
            "primary": float(truth["primary"]),
            "gauc": float(truth["GAUC"]),
            "ndcg5": float(truth["nDCG@5"]),
        }

    out_path.write_text(json.dumps({**claimed, "epochs_run": int(cfg.get("epochs", 3))}))

    print(f"{TEST_METRICS_SENTINEL} " + json.dumps(
        {"primary": claimed["primary"], "gauc": claimed["gauc"], "ndcg5": claimed["ndcg5"]}
    ))


if __name__ == "__main__":
    sys.exit(main())
