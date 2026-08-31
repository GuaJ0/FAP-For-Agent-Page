"""Combined: sequential attention (experiments/sequential_attention) +
multi-task auxiliary heads (experiments/multitask), in one model.

Each idea was validated independently first:
  - multitask alone:            valid primary 0.6042 vs baseline 0.6016 (+0.0026)
  - sequential_attention alone: valid primary 0.6014 vs baseline 0.6016 (~flat)

Rationale for combining rather than picking the winner: the sequence encoder
is a data-hungry component (attention weights, positional embeddings) being
trained from pointwise long_view BCE alone, which is a comparatively weak,
sparse gradient signal. The auxiliary is_click head in particular is dense
(46% positive rate) and may give the sequence encoder more signal to learn a
useful history representation from, even though it did not help on its own.

Architecture: identical to sequential_attention's SequentialAttentionFM (same
self-attention encoder, same target-attention pooling, same FM fusion of the
pooled history as a 6th field) with multitask's click_head/like_head added,
reading off S (the FM sum INCLUDING the sequence contribution) -- so the
auxiliary gradients flow into the sequence encoder too, not just the 5 static
fields.

Invocation contract, identical to solution/train.py:
    python train.py --config <cfg> --seed <n> --out <path/to/result.json>
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

try:
    from evaluate import evaluate
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harness"))
    from evaluate import evaluate

TEST_METRICS_SENTINEL = "TEST_METRICS:"
RAW_PREDICTIONS_NAME = "val_predictions.npz"

SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428), "test": (20220429, 20220508)}
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
MAX_SEQ_LEN = 50

DEFAULTS = {
    "data_dir": None,
    "k": 16,
    "seq_dim": 16,
    "n_heads": 2,
    "n_layers": 2,
    "lr": 0.001,
    "l2": 1e-6,
    "epochs": 40,
    "batch_size": 4096,
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
    """(date, user_id, video_id, author_id, tab, duration_ms, long_view,
    time_ms, is_click, is_like) -- union of what sequential_attention (time_ms)
    and multitask (is_click/is_like) each needed separately."""
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
                    int(r["time_ms"]), 1 if r["is_click"] != "0" else 0, 1 if r["is_like"] != "0" else 0,
                ))
    return {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


def build_histories(train_rows):
    video_vocab: dict = {}
    by_user = defaultdict(list)
    for (_, uid, vid, _author, _tab, _dur, _label, t_ms, _c, _l) in sorted(train_rows, key=lambda r: r[7]):
        if vid not in video_vocab:
            video_vocab[vid] = len(video_vocab) + 1
        by_user[uid].append(video_vocab[vid])

    histories = {}
    for uid, seq in by_user.items():
        seq = seq[-MAX_SEQ_LEN:]
        pad = [0] * (MAX_SEQ_LEN - len(seq))
        histories[uid] = np.array(pad + seq, dtype=np.int64)
    return histories, video_vocab


def encode(splits, histories, video_vocab, max_train_rows=None, seed=0):
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

    empty_hist = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    target_unk = len(video_vocab) + 1

    enc = {}
    for name, rws in (("train", train), ("valid", splits["valid"]), ("test", splits["test"])):
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int64)
        seq = np.empty((len(rws), MAX_SEQ_LEN), dtype=np.int64)
        target_item = np.empty(len(rws), dtype=np.int64)
        y = np.empty(len(rws), dtype=np.float32)
        y_click = np.empty(len(rws), dtype=np.float32)
        y_like = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            seq[n] = histories.get(x[1], empty_hist)
            target_item[n] = video_vocab.get(x[2], target_unk)
            y[n] = x[6]
            y_click[n] = x[8]
            y_like[n] = x[9]
            users.append(x[1])
        enc[name] = (X, seq, target_item, y, y_click, y_like, users)
    return enc, int(sum(dims)), target_unk + 1


class CombinedModel(nn.Module):
    """SequentialAttentionFM's architecture (self-attention over history,
    target-attention pooling, fused as a 6th FM field) plus multitask's
    click/like heads, reading off S (the full 6-field sum) rather than just
    the 5 static fields -- so auxiliary gradients reach the sequence encoder."""

    def __init__(self, fm_dim, item_vocab_size, k=16, seq_dim=16, n_heads=2, n_layers=2):
        super().__init__()
        self.V = nn.Parameter(torch.randn(fm_dim, k) * 0.01)
        self.W = nn.Parameter(torch.zeros(fm_dim))
        self.b = nn.Parameter(torch.zeros(1))

        self.item_embed = nn.Embedding(item_vocab_size, seq_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN, seq_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=seq_dim, nhead=n_heads, dim_feedforward=seq_dim * 2,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.seq_to_fm = nn.Linear(seq_dim, k)
        self.seq_first_order = nn.Linear(seq_dim, 1)

        self.click_head = nn.Linear(k, 1)
        self.like_head = nn.Linear(k, 1)

    def _sequence_repr(self, seq_ids, target_item_ids):
        pad_mask = seq_ids == 0
        H = self.item_embed(seq_ids)
        positions = torch.arange(seq_ids.shape[1], device=seq_ids.device)
        H = H + self.pos_embed(positions).unsqueeze(0)
        all_padded = pad_mask.all(dim=1)
        safe_mask = pad_mask.clone()
        safe_mask[all_padded, 0] = False
        H_enc = self.encoder(H, src_key_padding_mask=safe_mask)

        target_h = self.item_embed(target_item_ids)
        scores = (H_enc * target_h.unsqueeze(1)).sum(-1) / (H_enc.shape[-1] ** 0.5)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        scores = scores.masked_fill(all_padded.unsqueeze(1), 0.0)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * H_enc).sum(1)
        pooled = pooled * (~all_padded).unsqueeze(-1)
        return pooled

    def forward(self, X, seq_ids, target_item_ids):
        E = self.V[X]
        S5 = E.sum(1)

        pooled = self._sequence_repr(seq_ids, target_item_ids)
        u = self.seq_to_fm(pooled)

        S = S5 + u
        sq_sum = (E ** 2).sum((1, 2)) + (u ** 2).sum(1)
        interaction = 0.5 * ((S ** 2).sum(1) - sq_sum)
        first_order = self.W[X].sum(1) + self.seq_first_order(pooled).squeeze(-1)
        main_logit = self.b + first_order + interaction

        click_logit = self.click_head(S).squeeze(-1)
        like_logit = self.like_head(S).squeeze(-1)
        return main_logit, click_logit, like_logit


def predict_in_batches(model, X, seq, target, bs=8192):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            chunks.append(model(X[i:i + bs], seq[i:i + bs], target[i:i + bs])[0])
    return torch.cat(chunks).numpy()


def train(cfg, seed):
    torch.manual_seed(seed)
    data_dir = resolve_data_dir(cfg)
    splits = load_raw(data_dir)
    histories, video_vocab = build_histories(splits["train"])
    enc, fm_dim, item_vocab_size = encode(
        splits, histories, video_vocab, max_train_rows=cfg.get("max_train_rows"), seed=seed,
    )
    Xtr, seqtr, ttr, ytr, ytr_click, ytr_like, _ = enc["train"]
    Xva, seqva, tva, yva, _, _, uva = enc["valid"]
    Xte, seqte, tte, yte, _, _, ute = enc["test"]

    def to_t(*arrs):
        return [torch.from_numpy(a) for a in arrs]

    Xtr_t, seqtr_t, ttr_t, ytr_t, ytr_click_t, ytr_like_t = to_t(Xtr, seqtr, ttr, ytr, ytr_click, ytr_like)
    Xva_t, seqva_t, tva_t = to_t(Xva, seqva, tva)
    Xte_t, seqte_t, tte_t = to_t(Xte, seqte, tte)

    model = CombinedModel(
        fm_dim, item_vocab_size, k=int(cfg["k"]), seq_dim=int(cfg["seq_dim"]),
        n_heads=int(cfg["n_heads"]), n_layers=int(cfg["n_layers"]),
    )
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
            main_logit, click_logit, like_logit = model(Xtr_t[b], seqtr_t[b], ttr_t[b])
            loss = (
                bce(main_logit, ytr_t[b])
                + w_click * bce(click_logit, ytr_click_t[b])
                + w_like * bce(like_logit, ytr_like_t[b])
            )
            loss.backward()
            opt.step()

        va_scores = predict_in_batches(model, Xva_t, seqva_t, tva_t, bs=bs)
        va = evaluate(uva, yva, va_scores)
        epochs_run = ep
        print(f"  epoch {ep:2d} | valid primary {va['primary']:.4f} gauc {va['GAUC']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                print(f"  early stop at epoch {ep}", flush=True)
                break

    model.load_state_dict(best_state)
    val_scores = predict_in_batches(model, Xva_t, seqva_t, tva_t, bs=bs)
    test_scores = predict_in_batches(model, Xte_t, seqte_t, tte_t, bs=bs)
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
        "seed": int(seed), "wall_s": float(wall_s), "loss": "combined_seqattn_multitask",
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
