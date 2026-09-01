#!/usr/bin/env python3
"""Assemble submission/ from the checkpoint registry's validation-best iteration.

There is no separate "starter kit submission schema" beyond what
harness/evaluate.py consumes and what a real train.py already produces next
to its --out result.json (see solution/train.py's RAW_PREDICTIONS_NAME
convention): result.json, val_predictions.npz, and checkpoint.npz. This
script assembles exactly that, plus the code that produced it, from the two
places those pieces live on disk after a run:

  - logs/registry.json's best_iteration -> logs/runs.jsonl's matching
    RunRecord -> the winning seed's SeedMetrics.artifact_dir
    (logs/artifacts/iter_N/seed_M/), which holds result.json /
    val_predictions.npz / checkpoint.npz.
  - that same RunRecord's diff_path -> its parent directory
    (solutions/attempt_NNN/), which holds train.py / config.json.

Usage:
    python scripts/build_submission.py [--registry logs/registry.json]
        [--runs logs/runs.jsonl] [--out submission] [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.records import RunLog, RunRecord  # noqa: E402
from agent.registry import CheckpointRegistry  # noqa: E402

ARTIFACT_FILES = ["result.json", "val_predictions.npz", "checkpoint.npz"]
# train.py is always this name by convention (agent/executor.py invokes it by
# that exact path). The config file's name is NOT fixed the same way: a
# CodingAgent-generated solution always names it config.json, but the
# bootstrap baseline's config.path is solution/config.yaml (see
# solution/config.yaml) -- discovered by dry-running this script rather than
# assumed, so it is resolved from RunRecord.diff_path itself below, not
# hardcoded here.
SOLUTION_FILES = ["train.py"]


def find_run_record(runs: list[RunRecord], iteration: int) -> RunRecord:
    for r in runs:
        if r.iteration == iteration:
            return r
    raise SystemExit(f"no RunRecord for iteration {iteration} in runs.jsonl")


def winning_seed(record: RunRecord):
    ok = [s for s in record.seeds if s.failure_kind is None and s.artifact_dir]
    if not ok:
        raise SystemExit(f"iteration {record.iteration} has no seed with a usable artifact_dir")
    return max(ok, key=lambda s: s.primary)


def solution_dir_from_diff_path(diff_path) -> Path | None:
    if not diff_path:
        return None
    return Path(diff_path).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="logs/registry.json")
    ap.add_argument("--runs", default="logs/runs.jsonl")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out/checkpoint directory")
    a = ap.parse_args()

    registry_path, runs_path, out_dir = Path(a.registry), Path(a.runs), Path(a.out)

    if not registry_path.exists():
        raise SystemExit(f"missing {registry_path} -- has any iteration been accepted yet?")
    if not runs_path.exists():
        raise SystemExit(f"missing {runs_path}")

    registry = CheckpointRegistry(registry_path)
    best = registry.best()
    if best is None:
        raise SystemExit("registry has no best_iteration -- nothing accepted yet")

    record = find_run_record(RunLog(runs_path).read_all(), best.iteration)
    seed = winning_seed(record)

    artifact_dir = Path(seed.artifact_dir)
    sol_dir = solution_dir_from_diff_path(record.diff_path)

    missing = []
    if not artifact_dir.exists():
        missing.append(f"artifact_dir does not exist: {artifact_dir}")
    else:
        for name in ARTIFACT_FILES:
            if not (artifact_dir / name).exists():
                if name == "checkpoint.npz":
                    continue  # optional: a model may have no separate weights file
                missing.append(f"missing {name} in {artifact_dir}")

    config_src = Path(record.diff_path) if record.diff_path else None
    if sol_dir is None:
        missing.append(
            "RunRecord.diff_path is None -- can't locate train.py/config "
            "(this iteration's Diff had no config path recorded, e.g. a bootstrap "
            "baseline record produced without going through CodingAgent)"
        )
    elif not sol_dir.exists():
        missing.append(f"solution dir does not exist: {sol_dir}")
    else:
        for name in SOLUTION_FILES:
            if not (sol_dir / name).exists():
                missing.append(f"missing {name} in {sol_dir}")
        if config_src is None or not config_src.exists():
            missing.append(f"config file does not exist: {config_src}")

    checkpoint_out = out_dir / "checkpoint"
    existing_result = checkpoint_out / "result.json"
    if existing_result.exists():
        prior = json.loads(existing_result.read_text()).get("primary")
        if prior is not None and seed.primary is not None and seed.primary < prior:
            print(
                f"WARNING: {existing_result} already holds a BETTER result "
                f"(primary={prior}) than this iteration's winner (primary={seed.primary}). "
                "Refusing to overwrite -- pass --force if you really mean to.",
                file=sys.stderr,
            )
            if not a.force:
                return 2
        elif not a.force:
            print(
                f"{checkpoint_out} already exists -- pass --force to overwrite",
                file=sys.stderr,
            )
            return 2

    checkpoint_out.mkdir(parents=True, exist_ok=True)

    copied = []
    if artifact_dir.exists():
        for name in ARTIFACT_FILES:
            src = artifact_dir / name
            if src.exists():
                shutil.copy2(src, checkpoint_out / name)
                copied.append(str(checkpoint_out / name))
    if sol_dir and sol_dir.exists():
        for name in SOLUTION_FILES:
            src = sol_dir / name
            if src.exists():
                shutil.copy2(src, checkpoint_out / name)
                copied.append(str(checkpoint_out / name))
        if config_src and config_src.exists():
            # Copy under its own real name (config.json or config.yaml) --
            # renaming it would desync it from what --config actually expects.
            dest = checkpoint_out / config_src.name
            shutil.copy2(config_src, dest)
            copied.append(str(dest))

    result_path = artifact_dir / "result.json"
    result_json = json.loads(result_path.read_text()) if result_path.exists() else {}

    manifest = {
        "winning_iteration": best.iteration,
        "winning_seed": seed.seed,
        "checkpoint_path_in_registry": best.checkpoint_path,
        "hypothesis": record.hypothesis,
        "decision": record.decision.value if record.decision else None,
        "validation_metrics": result_json or {
            "primary": seed.primary, "gauc": seed.gauc, "ndcg5": seed.ndcg5,
        },
        "aggregate_across_seeds": record.aggregate.to_json() if record.aggregate else None,
        "source": {
            "artifact_dir": str(artifact_dir),
            "solution_dir": str(sol_dir) if sol_dir else None,
        },
        "missing": missing,
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"winning iteration: {best.iteration} (seed {seed.seed}), val primary={seed.primary}")
    print(f"copied {len(copied)} files into {checkpoint_out}:")
    for c in copied:
        print(f"  {c}")
    print(f"manifest written to {manifest_path}")
    if missing:
        print("\nMISSING (not fatal, but flagged):")
        for m in missing:
            print(f"  - {m}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
