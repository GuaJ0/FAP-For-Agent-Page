"""Multi-task FM: same 5 baseline fields, same FM interaction structure, but
trained with auxiliary heads on is_click / is_like alongside the main
long_view head, sharing the same embeddings.

Hand-written, not agent-generated -- this is a deliberately careful
implementation after the earlier gradient-sign bug in an agent-generated BPR
candidate. Kept as close to solution/train.py as possible (same fields, same
FM formula, same k/lr/optimizer) so any improvement is attributable to the
auxiliary supervision, not a confound from a different architecture.

WHY is_follow / is_comment / is_forward are excluded: measured base rates on
the real train split are 0.10%, 0.26%, 0.10% positive -- extreme enough that
a naive BCE auxiliary head on them is more likely to inject noisy gradients
than useful signal. is_click (46.3%) and is_like (1.9%) are dense/moderate
enough to plausibly help and are the two used here.

Invocation contract, identical to solution/train.py:
    python train.py --config <cfg> --seed <n> --out <path/to/result.json>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)  # stay inside the single-core budget, per project convention

try:
    from evaluate import evaluate
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harness"))
    from evaluate import evaluate

TEST_METRICS_SENTINEL = "TEST_METRICS:"
RAW_PREDICTIONS_NAME = "val_predictions.npz"

SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428), "test": (20220429, 20220508)}
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]

DEFAULTS = {
    "data_dir": None,
    "k": 16,
    "lr": 0.001,
    "l2": 1e-6,
    "epochs": 40,
    "batch_size": 8192,
    "patience": 4,
    "max_train_rows": None,
    "click_weight": 0.2,
    "like_weight": 0.2,
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
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
        return out


def resolve_data_dir(cfg):
    d = cfg.get("data_dir") or os.environ.get("KUAIRAND_PATH") or "./KuaiRand-Pure/data"
    if not Path(d).is_dir():
        raise SystemExit(f"data_dir {d!r} is not a directory; set KUAIRAND_PATH or config.data_dir")
    return str(d)


def load_raw(data_dir):
    """Loads (date, user_id, video_id, author_id, tab, duration_ms, long_view,
    is_click, is_like) tuples from the two log files, joining author_id from
    video_features_basic_pure.csv -- same join solution/train.py's harness
    uses, plus the two auxiliary labels that vendored harness/data.py drops."""
    import csv
    vid2author = {}
    with open(Path(data_dir) / "video_features_basic_pure.csv") as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]

    rows = []
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(Path(data_dir) / f) as fh:
            for r in csv.DictReader(fh):
                rows.append((
                    int(r["date"]), r["user_id"], r["video_id"],
                    vid2author.get(r["video_id"], "UNK"), r["tab"],
                    float(r["duration_ms"]), 1 if r["long_view"] != "0" else 0,
                    1 if r["is_click"] != "0" else 0, 1 if r["is_like"] != "0" else 0,
                ))
    return {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


def encode(splits, max_train_rows=None, seed=0):
    train = splits["train"]
    if max_train_rows and max_train_rows < len(train):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(train), size=int(max_train_rows), replace=False)
        train = [train[i] for i in keep]

    edges = np.quantile([x[5] for x in train], np.linspace(0, 1, 11)[1:-1])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in train:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    enc = {}
    for name, rws in (("train", train), ("valid", splits["valid"]), ("test", splits["test"])):
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int64)
        y = np.empty(len(rws), dtype=np.float32)
        y_click = np.empty(len(rws), dtype=np.float32)
        y_like = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            y_click[n] = x[7]
            y_like[n] = x[8]
            users.append(x[1])
        enc[name] = (X, y, y_click, y_like, users)
    return enc, int(sum(dims))


class MultiTaskFM(nn.Module):
    """Same FM interaction as solution/train.py's FM class (bias + first-order
    + second-order over the 5 fields), plus two small linear auxiliary heads
    reading the same summed embedding -- shared representation, separate
    supervision. Main head's FM structure is untouched by the auxiliary
    heads: they read S (the summed embeddings) but do not feed back into the
    main FM formula itself, only into the shared embedding gradients."""

    def __init__(self, dim, k=16):
        super().__init__()
        self.V = nn.Parameter(torch.randn(dim, k) * 0.01)
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.click_head = nn.Linear(k, 1)
        self.like_head = nn.Linear(k, 1)

    def forward(self, X):
        E = self.V[X]                       # (B, F, k)
        S = E.sum(1)                         # (B, k)
        interaction = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        main_logit = self.b + self.W[X].sum(1) + interaction
        click_logit = self.click_head(S).squeeze(-1)
        like_logit = self.like_head(S).squeeze(-1)
        return main_logit, click_logit, like_logit


def train(cfg, seed):
    torch.manual_seed(seed)
    data_dir = resolve_data_dir(cfg)
    splits = load_raw(data_dir)
    enc, dim = encode(splits, max_train_rows=cfg.get("max_train_rows"), seed=seed)
    Xtr, ytr, ytr_click, ytr_like, _ = enc["train"]
    Xva, yva, _, _, uva = enc["valid"]
    Xte, yte, _, _, ute = enc["test"]

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    ytr_click_t = torch.from_numpy(ytr_click)
    ytr_like_t = torch.from_numpy(ytr_like)
    Xva_t = torch.from_numpy(Xva)
    Xte_t = torch.from_numpy(Xte)

    model = MultiTaskFM(dim, k=int(cfg["k"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["l2"]))
    bce = nn.BCEWithLogitsLoss()
    bs = int(cfg["batch_size"])
    w_click, w_like = float(cfg["click_weight"]), float(cfg["like_weight"])

    best, best_state, bad, epochs_run = -1.0, None, 0, 0
    rng = np.random.default_rng(seed)

    for ep in range(1, int(cfg["epochs"]) + 1):
        model.train()
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            main_logit, click_logit, like_logit = model(Xtr_t[b])
            loss = (
                bce(main_logit, ytr_t[b])
                + w_click * bce(click_logit, ytr_click_t[b])
                + w_like * bce(like_logit, ytr_like_t[b])
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_scores = model(Xva_t)[0].numpy()
        va = evaluate(uva, yva, va_scores)
        epochs_run = ep
        print(f"  epoch {ep:2d} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                print(f"  early stop at epoch {ep}", flush=True)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_scores = model(Xva_t)[0].numpy()
        test_scores = model(Xte_t)[0].numpy()
    val_metrics = evaluate(uva, yva, val_scores)
    test_metrics = evaluate(ute, yte, test_scores)
    return val_metrics, test_metrics, epochs_run, (uva, yva, val_scores), model.state_dict()


def write_outputs(out_path, cfg, val_metrics, test_metrics, epochs_run, raw_val, state, seed, wall_s):
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("write_raw_predictions", True):
        users, labels, scores = raw_val
        np.savez_compressed(
            out_dir / RAW_PREDICTIONS_NAME,
            user_ids=np.asarray(users), labels=np.asarray(labels, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
        )
    np.savez_compressed(out_dir / "checkpoint.npz", **{k: v.numpy() for k, v in state.items()})

    out_path.write_text(json.dumps({
        "primary": float(val_metrics["primary"]), "gauc": float(val_metrics["GAUC"]),
        "ndcg5": float(val_metrics["nDCG@5"]), "epochs_run": int(epochs_run),
        "seed": int(seed), "wall_s": float(wall_s), "loss": "multitask_fm",
        "n_val_rows": int(val_metrics["rows"]), "n_val_users": int(val_metrics["users"]),
    }, indent=2))

    if cfg.get("emit_test_metrics", True):
        print(f"{TEST_METRICS_SENTINEL} " + json.dumps({
            "primary": float(test_metrics["primary"]), "gauc": float(test_metrics["GAUC"]),
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
