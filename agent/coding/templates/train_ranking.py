"""Ranking-loss FM: same features and same model as the iteration-0 baseline,
optimised for the metric instead of for calibrated probabilities.

Hypothesis: GAUC and nDCG@5 are within-user *ranking* metrics, but the
baseline trains pointwise logloss, which spends capacity on getting absolute
click probabilities right -- effort that cannot move a within-user ordering.
Optimising a within-user ranking objective should align the two.

Two losses, selected by `loss` in the config:

  bpr       Pairwise. For each user, sample (positive, negative) pairs from
            that user's own impressions and maximise
            log sigmoid(z_pos - z_neg).
  listwise  Within-user softmax cross-entropy over the user's sampled
            impressions: -log( exp(z_pos) / sum_j exp(z_j) ).

Both sample *within a user*, which is the point: the metric never compares
scores across users, so neither should the loss. A consequence worth knowing
is that any term constant within a user (a pure user-side bias) has zero
gradient under both -- consistent with the organisers' observed result that
pure user-side first-order features contribute exactly nothing.

Everything else -- feature encoding, FM scorer, Adam, early stopping on
validation primary, artifact layout -- is unchanged from solution/train.py,
so a difference in the score is attributable to the loss.

Honours the executor's contract:
    python train.py --config <cfg> --seed <n> --out <result.json>
result.json carries validation metrics only; hidden-test numbers go to the
quarantined TEST_METRICS stdout line; the raw (user_id, label, score) arrays
behind the reported metrics are persisted for executor-side verification.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from evaluate import evaluate
    from data import load, encode, FIELDS
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
    from evaluate import evaluate
    from data import load, encode, FIELDS

TEST_METRICS_SENTINEL = "TEST_METRICS:"
RAW_PREDICTIONS_NAME = "val_predictions.npz"

DEFAULTS = {
    "data_dir": None,
    "loss": "bpr",
    "k": 16,
    "lr": 0.001,
    "l2": 1e-6,
    "epochs": 40,
    "batch_size": 8192,
    "patience": 4,
    "pairs_per_epoch": None,   # default: as many pairs as there are train rows
    "list_size": 8,            # listwise: candidates per sampled list
    "max_train_rows": None,
    "emit_test_metrics": True,
    "write_raw_predictions": True,
}


def load_config(path):
    text = Path(path).read_text()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
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
    """Identical scorer to solution/train.py's. Only `step` differs -- the
    gradient of the loss w.r.t. the logits changes, the backward pass from
    logits into (V, W) does not."""

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
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _accumulate(self, X, g, E, S, gV, gW):
        """Scatter dL/dz for a block of rows back into the parameter grads.

        The `b` (global bias) gradient is deliberately not accumulated: under
        both ranking losses the per-row dL/dz sums to zero within every
        comparison group, so b has no gradient at all. It also cannot affect
        a within-user ordering. Keeping it fixed makes that explicit.
        """
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))

    def _adam(self, gV, gW):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def step_bpr(self, Xp, Xn):
        """-log sigmoid(z_pos - z_neg), averaged over the batch."""
        B = len(Xp)
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn)
        d = zp - zn
        # dL/d(zp) = -sigmoid(-d); dL/d(zn) = +sigmoid(-d)
        g = (-sigmoid(-d) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        self._accumulate(Xp, g, Ep, Sp, gV, gW)
        self._accumulate(Xn, -g, En, Sn, gV, gW)
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self._adam(gV, gW)
        return float(np.mean(-np.log(sigmoid(d) + 1e-9)))

    def step_listwise(self, Xl):
        """Within-list softmax cross-entropy. Xl is (B, L, F); column 0 of each
        list is the positive."""
        B, L, F = Xl.shape
        flat = Xl.reshape(B * L, F)
        z, E, S = self.logits(flat)
        z = z.reshape(B, L)
        z = z - z.max(axis=1, keepdims=True)          # stabilise before exp
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        # target is index 0, so dL/dz = p - onehot(0)
        g = p.copy()
        g[:, 0] -= 1.0
        g = (g.reshape(B * L) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        self._accumulate(flat, g, E, S, gV, gW)
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self._adam(gV, gW)
        return float(np.mean(-np.log(p[:, 0] + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def state(self):
        return {"V": self.V, "W": self.W, "b": np.float32(self.b)}

    def load_state(self, st):
        self.V, self.W, self.b = st["V"], st["W"], np.float32(st["b"])


class UserGroups:
    """Per-user positive/negative row indices in a flat CSR-style layout.

    Ragged per-user lists as a dict-of-arrays would force a Python loop per
    sampled pair -- ~1.1M iterations an epoch, which dominates the actual
    training. Flattening to (values, offsets, counts) makes sampling a couple
    of vectorised gathers instead.
    """

    def __init__(self, users, y):
        pos, neg = defaultdict(list), defaultdict(list)
        for i, (u, label) in enumerate(zip(users, y)):
            (pos if label > 0 else neg)[u].append(i)

        # Only users with at least one of each offer a within-user comparison
        # -- the same set the GAUC denominator counts.
        keys = sorted(u for u in pos if u in neg)
        self.n_users = len(keys)
        self.pos_flat, self.pos_off, self.pos_cnt = self._flatten(pos, keys)
        self.neg_flat, self.neg_off, self.neg_cnt = self._flatten(neg, keys)

    @staticmethod
    def _flatten(d, keys):
        counts = np.array([len(d[k]) for k in keys], dtype=np.int64)
        offsets = np.zeros(len(keys), dtype=np.int64)
        if len(keys):
            offsets[1:] = np.cumsum(counts)[:-1]
        flat = np.concatenate([np.asarray(d[k], dtype=np.int64) for k in keys]) if keys \
            else np.empty(0, dtype=np.int64)
        return flat, offsets, counts

    def __len__(self):
        return self.n_users

    def _gather(self, which, user_idx, rng, size=None):
        flat, off, cnt = (
            (self.pos_flat, self.pos_off, self.pos_cnt) if which == "pos"
            else (self.neg_flat, self.neg_off, self.neg_cnt)
        )
        shape = user_idx.shape if size is None else (len(user_idx), size)
        c = cnt[user_idx] if size is None else cnt[user_idx][:, None]
        o = off[user_idx] if size is None else off[user_idx][:, None]
        # rng.random * count, floored: one uniform draw per slot, no rejection.
        j = (rng.random(shape) * c).astype(np.int64)
        return flat[o + np.minimum(j, c - 1)]

    def sample_pairs(self, n_pairs, rng):
        """n_pairs (positive, negative) row indices, users drawn uniformly.

        Uniform over *users*, not over rows: GAUC weights users by positive
        count but nDCG weights every user equally, and per-row sampling would
        over-serve heavy users relative to both.
        """
        if not self.n_users:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        u = rng.integers(0, self.n_users, size=n_pairs)
        return self._gather("pos", u, rng), self._gather("neg", u, rng)

    def sample_lists(self, n_lists, list_size, rng):
        """n_lists row-index arrays, each [positive, negative x (list_size-1)].

        Negatives are drawn with replacement: many users have fewer than
        list_size-1 negatives, and resampling is cheaper than variable-length
        lists while leaving the softmax unbiased.
        """
        if not self.n_users:
            return np.empty((0, list_size), dtype=np.int64)
        u = rng.integers(0, self.n_users, size=n_lists)
        out = np.empty((n_lists, list_size), dtype=np.int64)
        out[:, 0] = self._gather("pos", u, rng)
        out[:, 1:] = self._gather("neg", u, rng, size=list_size - 1)
        return out


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
    loss_name = cfg.get("loss", "bpr")
    if loss_name not in ("bpr", "listwise"):
        raise SystemExit(f"unsupported loss {loss_name!r}; expected 'bpr' or 'listwise'")

    splits = load(resolve_data_dir(cfg))
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]

    rng = np.random.default_rng(seed)

    max_rows = cfg.get("max_train_rows")
    if max_rows and max_rows < len(ytr):
        keep = np.sort(rng.choice(len(ytr), size=int(max_rows), replace=False))
        Xtr, ytr = Xtr[keep], ytr[keep]
        utr = [utr[i] for i in keep]

    groups = UserGroups(utr, ytr)
    if not len(groups):
        raise SystemExit(
            "no user in the training split has both a positive and a negative "
            "impression, so no within-user pair can be formed"
        )

    m = FM(dim, k=int(cfg["k"]), lr=float(cfg["lr"]), l2=float(cfg["l2"]), seed=seed)
    bs = int(cfg["batch_size"])
    n_per_epoch = int(cfg.get("pairs_per_epoch") or len(ytr))
    list_size = max(2, int(cfg.get("list_size", 8)))
    best, best_state, bad, epochs_run = -1.0, None, 0, 0

    print(f"  loss={loss_name} usable_users={len(groups)} samples/epoch={n_per_epoch}", flush=True)

    for ep in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        losses = []
        if loss_name == "bpr":
            pi, ni = groups.sample_pairs(n_per_epoch, rng)
            for i in range(0, n_per_epoch, bs):
                losses.append(m.step_bpr(Xtr[pi[i:i + bs]], Xtr[ni[i:i + bs]]))
        else:
            n_lists = max(1, n_per_epoch // list_size)
            idx = groups.sample_lists(n_lists, list_size, rng)
            for i in range(0, n_lists, bs):
                losses.append(m.step_listwise(Xtr[idx[i:i + bs]]))

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
    return val_metrics, test_metrics, epochs_run, (uva, yva, val_scores), m.state()


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

    out_path.write_text(json.dumps({
        "primary": float(val_metrics["primary"]),
        "gauc": float(val_metrics["GAUC"]),
        "ndcg5": float(val_metrics["nDCG@5"]),
        "epochs_run": int(epochs_run),
        "seed": int(seed),
        "wall_s": float(wall_s),
        "loss": cfg.get("loss", "bpr"),
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

    t0 = time.time()
    val_metrics, test_metrics, epochs_run, raw_val, state = train(cfg, a.seed)
    wall_s = time.time() - t0

    write_outputs(a.out, cfg, val_metrics, test_metrics, epochs_run, raw_val, state, a.seed, wall_s)
    print(f"valid GAUC {val_metrics['GAUC']:.4f} | nDCG@5 {val_metrics['nDCG@5']:.4f} "
          f"| primary {val_metrics['primary']:.4f} | {wall_s:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
