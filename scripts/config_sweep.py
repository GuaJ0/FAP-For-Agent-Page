"""Rerun ONE already-generated implementation across several config points.

This is the hyperparameter-sensitivity mode: the code is held fixed and only
the config changes, so a difference between points is attributable to the
setting rather than to the Coding Agent having written different code. It is
deliberately NOT part of the main loop -- see agent/sweep.py's docstring for
why -- and it never registers a checkpoint, so it cannot move the incumbent.

Use it when a direction's single attempt lost and you want to know whether the
mechanism failed or the setting was simply wrong. That distinction is the
difference between a real "Don't" and a false one.

    # sweep one key across three values against an accepted solution
    python scripts/config_sweep.py \\
        --solution-dir runs/exploration/solutions/attempt_004 \\
        --base-config  runs/exploration/solutions/attempt_004/config.yaml \\
        --sweep lambda_bpr=0.05,0.1,0.2 \\
        --incumbent-primary 0.6015

    # several keys per point
    python scripts/config_sweep.py --solution-dir ... --base-config ... \\
        --point "wide:k=32,lr=0.003" --point "narrow:k=8,lr=0.001"

    # no LLM judge, no spend (deterministic margin rule instead)
    python scripts/config_sweep.py ... --offline

Results land in <root>/logs/runs.jsonl as ordinary RunRecords, and the
generated configs under <root>/logs/sweeps/. Nothing is written into the
solution dir being swept.

Reads KUAIRAND_PATH (and OPENAI_API_KEY) from .env if present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agent.agents import FakeEvaluatorAgent  # noqa: E402
from agent.coding import OpenAIClient  # noqa: E402
from agent.evaluator import LLMEvaluatorAgent  # noqa: E402
from agent.executor import Executor  # noqa: E402
from agent.records import RunLog  # noqa: E402
from agent.sweep import ConfigPoint, ConfigSweep, ConfigSweepError  # noqa: E402
from run_loop import build_config, load_dotenv  # noqa: E402


def _coerce(raw: str):
    """Parse an override value. JSON first so ints, floats, booleans and null
    keep their type; anything else stays the literal string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_points(sweeps: list[str], points: list[str]) -> list[ConfigPoint]:
    """Build the sweep's points from --sweep and --point.

    --sweep key=a,b,c   one point per value, single key varied
    --point label:k=v,k=v   one explicitly labelled multi-key point
    """
    out: list[ConfigPoint] = []
    for spec in sweeps:
        if "=" not in spec:
            raise ConfigSweepError(f"--sweep needs key=v1,v2,...; got {spec!r}")
        key, _, values = spec.partition("=")
        key = key.strip()
        for value in [v.strip() for v in values.split(",") if v.strip()]:
            out.append(ConfigPoint(f"{key}={value}", {key: _coerce(value)}))
    for spec in points:
        if ":" not in spec:
            raise ConfigSweepError(f"--point needs label:k=v,...; got {spec!r}")
        label, _, assignments = spec.partition(":")
        overrides = {}
        for pair in [p.strip() for p in assignments.split(",") if p.strip()]:
            if "=" not in pair:
                raise ConfigSweepError(f"--point assignment needs k=v; got {pair!r}")
            k, _, v = pair.partition("=")
            overrides[k.strip()] = _coerce(v.strip())
        out.append(ConfigPoint(label.strip(), overrides))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution-dir", required=True, help="the fixed implementation to rerun")
    ap.add_argument("--base-config", required=True, help="config the points override")
    ap.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2",
                    help="one point per value of a single key; repeatable")
    ap.add_argument("--point", action="append", default=[], metavar="LABEL:K=V,K=V",
                    help="one explicitly labelled multi-key point; repeatable")
    ap.add_argument("--incumbent-primary", type=float, default=None,
                    help="validation primary each point's delta is measured against. Passed in "
                         "rather than read from the registry: a sweep is a measurement, not a "
                         "promotion path, and must not be able to move the incumbent.")
    ap.add_argument("--root", default=str(REPO_ROOT / "runs" / "sweeps"),
                    help="scratch dir for logs/ (default: runs/sweeps)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--start-iteration", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--model", default=os.environ.get("CODING_AGENT_MODEL", "gpt-5"))
    ap.add_argument("--offline", action="store_true",
                    help="use the deterministic margin rule instead of an LLM judge (no spend)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--loss", default="bpr")
    ap.add_argument("--max-iterations", type=int, default=1)   # unused; build_config wants it
    ap.add_argument("--max-wall-s", type=float, default=24 * 3600.0)
    ap.add_argument("--hypothesis", default=None,
                    help="text recorded on every point's RunRecord")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    data_dir = os.environ.get("KUAIRAND_PATH")
    if data_dir:
        os.environ["KUAIRAND_PATH"] = data_dir

    try:
        points = parse_points(args.sweep, args.point)
        if not points:
            ap.error("give at least one --sweep or --point")

        root = Path(args.root)
        cfg = build_config(root, args)
        evaluator = FakeEvaluatorAgent() if args.offline else LLMEvaluatorAgent(
            llm=OpenAIClient(model=args.model),
            usage_log_path=cfg.paths.logs_dir / "evaluator_usage.jsonl",
        )
        print(f"[judge] {'deterministic margin rule -- no spend' if args.offline else args.model}")

        sweep = ConfigSweep(
            executor=Executor(cfg=cfg),
            evaluator=evaluator,
            run_log=RunLog(cfg.paths.runs_jsonl),
            cfg=cfg,
            seeds=list(range(args.seeds)),
        )
        solution_dir = Path(args.solution_dir)
        hypothesis = args.hypothesis or (
            f"Config-only sensitivity sweep over {solution_dir.name}: "
            + "; ".join(p.label for p in points)
        )

        print(f"[sweep] {len(points)} point(s) over {solution_dir}")
        results = sweep.run(
            solution_dir=solution_dir,
            base_config=Path(args.base_config),
            points=points,
            hypothesis=hypothesis,
            incumbent_primary=args.incumbent_primary,
            start_iteration=args.start_iteration,
            timeout_s=args.timeout_s,
        )
    except ConfigSweepError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"\n=== {len(results)} point(s) ===")
    for r in results:
        primary = f"{r.primary:.4f}" if r.primary is not None else "FAILED"
        delta = r.record.delta_vs_current_best
        decision = r.record.decision.value if r.record.decision else "unjudged"
        print(f"  {r.point.label:<28} primary={primary:<8} "
              f"delta={delta if delta is None else format(delta, '+.4f'):<9} {decision}")
    print(f"\nrun log: {cfg.paths.runs_jsonl}")
    print(f"configs: {sweep.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
