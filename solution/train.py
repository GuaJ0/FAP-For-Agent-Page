"""Iteration-0 solution: the official FM baseline, wrapped in the executor's
invocation contract.

This is a straight port of `harness/baseline.py`'s `run_fm` -- same model, same
optimizer, same defaults -- so it reproduces the published FM numbers
(test primary 0.5946 +- 0.0008). It exists so the agent loop has a real
starting point to branch from: the Research agent's "current model state" and
the Evaluator's "current best" both need an iteration 0 that actually ran.

Invocation contract (agent/executor.py drives this; fixtures/fake_train.py is
the minimal reference implementation):

    python train.py --config <cfg> --seed <n> --out <path/to/result.json>

`--config` is YAML or JSON, flat key/value. Every key is optional:

    data_dir            path to KuaiRand-Pure/data (default: $KUAIRAND_PATH)
    loss                "logloss" (default) | see note below
    k                   FM embedding dim (default 16)
    lr                  Adam learning rate (default 0.001)
    l2                  L2 penalty (default 1e-6)
    epochs              max epochs (default 40)
    batch_size          (default 8192)
    patience            early-stop patience on valid primary (default 4)
    max_train_rows      subsample train to N rows; null = all. Smoke-test hook.
    emit_test_metrics   print the quarantined TEST_METRICS line (default true)
    write_raw_predictions  dump val_predictions.npz for executor-side
                        verification (default true)

Outputs, all into the directory holding `--out` (executor.py guarantees that
directory is persistent, not a tempdir, so a checkpoint here survives the
subprocess and can be registered):

    result.json          {"primary","gauc","ndcg5","epochs_run", ...}
    val_predictions.npz  (user_ids, labels, scores) actually used to compute
                         the numbers in result.json -- see the note below
    checkpoint.npz       best-epoch FM weights

stdout carries one `TEST_METRICS: {...}` line with hidden-test-split metrics.
executor.py greps that off stdout into logs/quarantine/ and it never reaches
any agent. Nothing here ever writes a test-split number into result.json.

WHY val_predictions.npz EXISTS
------------------------------
`evaluate.py` is the competition's sole scoring authority, but result.json is
just a file this process wrote -- nothing stops a future (LLM-generated)
train.py from reimplementing the metric slightly wrong, or from reporting a
number it never computed. So this file also persists the exact three arrays it
fed to `evaluate()`, and executor.py independently re-scores them through the
vendored `evaluate.py` and cross-checks the result. See agent/verification.py.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# The executor runs us with cwd=<solution_dir>, and the CodingAgent drops a
# copy of the vendored evaluate.py/data.py right next to train.py, so the flat
# import is the normal path. The fallback is for running this file in place
# from the repo (`python solution/train.py ...`), where the vendored copies
# live in ../harness instead.
try:
    from evaluate import evaluate
    from data import load, encode, FIELDS
except ImportError:  # pragma: no cover - exercised by running from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
    from evaluate import evaluate
    from data import load, encode, FIELDS

TEST_METRICS_SENTINEL = "TEST_METRICS:"  # keep in sync with agent/config.py
RAW_PREDICTIONS_NAME = "val_predictions.npz"  # keep in sync with agent/verification.py

DEFAULTS = {
    "data_dir": None,
    "loss": "logloss",
    "k": 16,
    "lr": 0.001,
    "l2": 1e-6,
    "epochs": 40,
    "batch_size": 8192,
    "patience": 4,
    "max_train_rows": None,
    "emit_test_metrics": True,
    "write_raw_predictions": True,
}


def load_config(path):
    """YAML or JSON, flat. PyYAML if available, else a small flat-YAML reader
    so a solution dir stays runnable on a numpy-only environment."""
    text = Path(path).read_text()
    if path.endswith((".json",)):
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        return _parse_flat_yaml(text)


def _parse_flat_yaml(text):
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        out[key.strip()] = _coerce(raw.strip())
    return out


def _coerce(raw):
    if raw in ("", "null", "~", "None"):
        return None
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw.strip("'\"")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Verbatim port of harness/baseline.py's FM (Adam, no bias in the
    interaction term). Kept identical so iteration 0 reproduces the published
    number rather than being a near-miss re-implementation."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self._adam(gV, gW)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def _adam(self, gV, gW):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def state(self):
        return {"V": self.V, "W": self.W, "b": np.float32(self.b)}

    def load_state(self, st):
        self.V, self.W, self.b = st["V"], st["W"], np.float32(st["b"])


def resolve_data_dir(cfg):
    d = cfg.get("data_dir") or os.environ.get("KUAIRAND_PATH") or "./KuaiRand-Pure/data"
    d = str(d)
    if not Path(d).is_dir():
        raise SystemExit(
            f"data_dir {d!r} is not a directory. Set `data_dir` in the config or "
            f"export KUAIRAND_PATH=/path/to/KuaiRand-Pure/data"
        )
    return d


def train(cfg, seed):
    """Returns (val_metrics, test_metrics, epochs_run, raw_val, state).

    `raw_val` is the exact (user_ids, labels, scores) triple handed to
    evaluate() for the reported validation metrics -- persisted so the
    executor can re-derive the score independently.
    """
    splits = load(resolve_data_dir(cfg))
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]

    rng = np.random.default_rng(seed)

    max_rows = cfg.get("max_train_rows")
    if max_rows and max_rows < len(ytr):
        keep = rng.choice(len(ytr), size=int(max_rows), replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]

    m = FM(dim, k=int(cfg["k"]), lr=float(cfg["lr"]), l2=float(cfg["l2"]), seed=seed)
    bs = int(cfg["batch_size"])
    best, best_state, bad, epochs_run = -1.0, None, 0, 0

    for ep in range(1, int(cfg["epochs"]) + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        epochs_run = ep
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} "
              f"| {time.time() - t0:.1f}s", flush=True)
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = {k: v.copy() for k, v in m.state().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                print(f"  early stop at epoch {ep}", flush=True)
                break

    if best_state is not None:
        m.load_state(best_state)

    val_scores = m.predict(Xva)
    val_metrics = evaluate(uva, yva, val_scores)
    test_metrics = evaluate(ute, yte, m.predict(Xte))
    raw_val = (uva, yva, val_scores)
    return val_metrics, test_metrics, epochs_run, raw_val, m.state()


def write_outputs(out_path, cfg, val_metrics, test_metrics, epochs_run, raw_val, state, seed, wall_s):
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("write_raw_predictions", True):
        users, labels, scores = raw_val
        np.savez_compressed(
            out_dir / RAW_PREDICTIONS_NAME,
            user_ids=np.asarray(users),
            labels=np.asarray(labels, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
        )

    np.savez_compressed(out_dir / "checkpoint.npz", **state)

    # result.json holds validation metrics ONLY. There is deliberately no
    # test-split key here -- that channel is the quarantined stdout line below.
    # float()/int() casts are load-bearing, not cosmetic: evaluate() sums the
    # float32 label array, so GAUC comes back as np.float32 and json.dumps
    # refuses it. executor.py would classify that crash as CRASH, not as the
    # "your metrics are wrong" signal it looks like.
    out_path.write_text(json.dumps({
        "primary": float(val_metrics["primary"]),
        "gauc": float(val_metrics["GAUC"]),
        "ndcg5": float(val_metrics["nDCG@5"]),
        "epochs_run": int(epochs_run),
        "seed": int(seed),
        "wall_s": float(wall_s),
        "loss": cfg.get("loss", "logloss"),
        "n_val_rows": int(val_metrics["rows"]),
        "n_val_users": int(val_metrics["users"]),
    }, indent=2))

    if cfg.get("emit_test_metrics", True):
        print(f"{TEST_METRICS_SENTINEL} " + json.dumps({
            "primary": float(test_metrics["primary"]),
            "gauc": float(test_metrics["GAUC"]),
            "ndcg5": float(test_metrics["nDCG@5"]),
        }), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(load_config(a.config) or {})

    if cfg.get("loss", "logloss") != "logloss":
        raise SystemExit(
            f"solution/train.py is the iteration-0 FM baseline and only implements "
            f"loss='logloss'; got {cfg['loss']!r}. A ranking loss is a CodingAgent change."
        )

    t0 = time.time()
    val_metrics, test_metrics, epochs_run, raw_val, state = train(cfg, a.seed)
    wall_s = time.time() - t0

    write_outputs(a.out, cfg, val_metrics, test_metrics, epochs_run, raw_val, state, a.seed, wall_s)
    print(f"valid GAUC {val_metrics['GAUC']:.4f} | nDCG@5 {val_metrics['nDCG@5']:.4f} "
          f"| primary {val_metrics['primary']:.4f} | {wall_s:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
