# Vendored KuaiRand-Pure harness

These files are copied **verbatim** from the official starter kit at
`kuairand-starter-kit` and must not be modified. `evaluate.py` in particular is
the sole scoring authority for this competition -- every number that reaches a
`RunRecord` has to come out of *this* file, not out of a reimplementation.

They live in this repo (rather than being imported from the starter kit) so a
`solution_dir` handed to `agent/executor.py` is self-contained: the executor
runs `train.py` in a subprocess with `cwd=solution_dir`, so a copy of
`evaluate.py` / `data.py` sitting next to `train.py` is importable as a
top-level module with no `sys.path` juggling and no dependency on the starter
kit still being present at a fixed path.

The KuaiRand-Pure CSVs themselves are **not** vendored -- they are ~200 MB and
stay in the starter kit checkout. Point `KUAIRAND_PATH` at them (see
`.env.example`).

| file | role | may edit? |
|---|---|---|
| `evaluate.py` | GAUC / nDCG@5 / primary. Scoring authority. | **no** |
| `data.py` | Loading, official date splits, feature encoding. | no (vendored copy) |
| `baseline.py` | pop / FM / random baselines. FM is the bar to beat. | no (vendored copy) |
| `baseline_scores.json` | Official published scores, seed std, convergence params. | no |

## Provenance

Copied on 2026-08-28 from `kuairand-starter-kit` @ working tree.

    sha256(evaluate.py)          = ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de
    sha256(data.py)              = 1bf54f5f3a9f590eab2f87f09a3c27422031867a20a5328d56cbd8c7db36e541
    sha256(baseline.py)          = c8f7fc60178413e247e78bb231e7550eeef52101b6493fcf1a4d2b0e5fe18f8a
    sha256(baseline_scores.json) = 950f98181770c030a68bdddab7be3c0abbf060531f54455a6a6f81a4cb003324

`tests/test_vendored_harness.py` re-checks these hashes, so an accidental edit
to a vendored file fails the suite rather than silently changing what "score"
means.

## Verified on vendoring day

Both numbers reproduced locally against the freshly downloaded dataset:

| check | expected | got |
|---|---|---|
| `baseline.py --model random` test primary | ~0.475 (+-0.001) | **0.4757** |
| `baseline.py --model fm --seed 0` test primary | 0.5946 (+-0.0008 seed std) | **0.5953** |
| `baseline.py --model fm --seed 0` valid primary | 0.6016 | **0.6015** |
