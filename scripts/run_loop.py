"""Run the agent loop end to end.

Wires the real CodingAgent into the EXISTING Orchestrator. Research and
Evaluator are still the Fakes from agent/agents.py -- there are no real
implementations of those yet -- so this drives one fixed hypothesis rather
than proposing its own.

    # offline: no API key, no spend, uses the hand-written template library
    python scripts/run_loop.py --offline

    # live: real OpenAI generation (needs OPENAI_API_KEY in .env)
    python scripts/run_loop.py --hypothesis "..." --model gpt-5

Reads KUAIRAND_PATH (and OPENAI_API_KEY) from .env if present.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.agents import Diff, FakeEvaluatorAgent, FakeResearchAgent, Idea  # noqa: E402
from agent.coding import LLMCodingAgent, OpenAIClient, TemplateLibraryClient  # noqa: E402
from agent.config import (  # noqa: E402
    Config,
    ConvergenceConfig,
    ExecutorConfig,
    Paths,
    RetryConfig,
    SeedingConfig,
)
from agent.executor import Executor  # noqa: E402
from agent.orchestrator import BootstrapError, Orchestrator  # noqa: E402
from agent.records import RunLog  # noqa: E402
from agent.registry import CheckpointRegistry  # noqa: E402
from agent.state import StateStore  # noqa: E402

BASELINE_HYPOTHESIS = (
    "Baseline: factorization machine over [user_id, video_id, author_id, tab, "
    "dur_bucket] trained with pointwise logloss -- the seeded solution/train.py, "
    "run as iteration 0 to establish the incumbent."
)

DEFAULT_HYPOTHESIS = (
    "Replace the pointwise logloss objective with a pairwise BPR ranking loss "
    "computed within each user, so the training objective matches the GAUC/nDCG@5 "
    "ranking metric instead of optimising calibrated click probabilities."
)


def load_dotenv(path: Path) -> None:
    """Minimal .env reader -- avoids a python-dotenv dependency. Never prints
    a value; only the variable names are ever echoed."""
    if not path.exists():
        return
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    if loaded:
        print(f"[env] loaded from .env: {', '.join(loaded)}")


def build_config(root: Path, args) -> Config:
    logs = root / "logs"
    # No baseline compensation here: convergence.should_stop() excludes the
    # bootstrap iteration from its max_iterations count, so this is passed
    # straight through and means exactly what it says. It used to need a +1.
    return Config(
        convergence=ConvergenceConfig(max_iterations=args.max_iterations, max_wall_s=args.max_wall_s),
        retry=RetryConfig(),
        executor=ExecutorConfig(per_run_timeout_s=args.timeout_s),
        seeding=SeedingConfig(max_seeds=args.seeds, min_seeds=1),
        paths=Paths(
            logs_dir=logs,
            runs_jsonl=logs / "runs.jsonl",
            quarantine_dir=logs / "quarantine",
            test_metrics_jsonl=logs / "quarantine" / "test_metrics.jsonl",
            orchestrator_state=logs / "orchestrator_state.json",
            registry_json=logs / "registry.json",
            artifacts_dir=logs / "artifacts",
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypothesis", default=DEFAULT_HYPOTHESIS)
    ap.add_argument("--offline", action="store_true",
                    help="use the hand-written template library instead of an LLM (no spend)")
    ap.add_argument("--model", default=os.environ.get("CODING_AGENT_MODEL", "gpt-5"))
    ap.add_argument("--data-dir", default=None, help="defaults to $KUAIRAND_PATH")
    ap.add_argument("--root", default=str(REPO_ROOT), help="where logs/ and solutions/ go")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--loss", default="bpr")
    ap.add_argument("--max-iterations", type=int, default=1,
                    help="number of research iterations. The bootstrap baseline is not "
                         "one of them and does not consume a slot.")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="do not run solution/ as iteration 0. The registry then starts "
                         "empty, so the first result becomes the incumbent whatever it scores.")
    ap.add_argument("--max-wall-s", type=float, default=6 * 3600.0)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    data_dir = args.data_dir or os.environ.get("KUAIRAND_PATH")
    if not data_dir or not Path(data_dir).is_dir():
        print("error: set KUAIRAND_PATH (in .env) or pass --data-dir; it must point at "
              "the directory holding the KuaiRand-Pure CSVs.", file=sys.stderr)
        return 2

    # solution/config.yaml leaves data_dir empty and falls back to this, and
    # the executor's subprocess inherits the environment.
    os.environ["KUAIRAND_PATH"] = str(data_dir)

    root = Path(args.root)
    cfg = build_config(root, args)

    if args.offline:
        client = TemplateLibraryClient()
        print("[llm] offline template library -- no API calls, no spend")
    else:
        client = OpenAIClient(model=args.model)
        print(f"[llm] OpenAI, model={args.model} -- this run costs money")

    coding = LLMCodingAgent(
        work_dir=root / "solutions",
        data_dir=data_dir,
        llm=client,
        usage_log_path=cfg.paths.logs_dir / "coding_agent_usage.jsonl",
        base_config={"loss": args.loss, "epochs": args.epochs, "patience": 3},
    )
    orc = Orchestrator(
        research=FakeResearchAgent([args.hypothesis]),
        coding=coding,
        evaluator=FakeEvaluatorAgent(),
        executor=Executor(cfg=cfg),
        run_log=RunLog(cfg.paths.runs_jsonl),
        registry=CheckpointRegistry(cfg.paths.registry_json),
        state_store=StateStore(cfg.paths.orchestrator_state),
        cfg=cfg,
    )

    t0 = time.time()

    if not args.skip_baseline:
        # Establish solution/ as a real iteration 0 before any hypothesis runs,
        # so deltas, ACCEPT/REVERT and convergence all have an incumbent to
        # measure against. Idempotent: a no-op if a previous run already did it.
        #
        # diff_path is solution/config.yaml, a sibling of solution/train.py --
        # that adjacency is what lets a registry entry be resolved back to the
        # source that produced it.
        baseline = Diff(
            diff_path=str(REPO_ROOT / "solution" / "config.yaml"),
            solution_dir=str(REPO_ROOT / "solution"),
        )
        try:
            record = orc.bootstrap_baseline(Idea(BASELINE_HYPOTHESIS, None), baseline)
        except BootstrapError as e:
            print(f"\nerror: could not establish the baseline as iteration 0:\n  {e}", file=sys.stderr)
            return 3
        agg = record.aggregate
        print(f"[baseline] iteration 0: valid primary={agg.primary_mean:.4f} "
              f"(std {agg.primary_std:.4f}) over {agg.n_seeds} seed(s) -- this is the bar")

    history = orc.run()
    print(f"\n=== finished in {time.time() - t0:.0f}s, {len(history)} record(s) ===")
    for r in history:
        print(f"\niteration {r.iteration}  status={r.status.value}  "
              f"decision={r.decision.value if r.decision else None}")
        if r.aggregate:
            a = r.aggregate
            print(f"  VALID primary={a.primary_mean:.4f} (std {a.primary_std:.4f}) "
                  f"gauc={a.gauc_mean:.4f} ndcg5={a.ndcg5_mean:.4f} over {a.n_seeds} seed(s)")
        for s in r.seeds:
            print(f"    seed {s.seed}: primary={s.primary} kind="
                  f"{s.failure_kind.value if s.failure_kind else None} wall={s.wall_s:.0f}s")
            if s.failure_kind:
                print(f"      {(s.traceback_tail or '')[:400]}")

    print(f"\nrun log:      {cfg.paths.runs_jsonl}")
    print(f"LLM usage:    {cfg.paths.logs_dir / 'coding_agent_usage.jsonl'}")
    print(f"quarantine:   {cfg.paths.test_metrics_jsonl}  (agents never read this)")
    print(f"LLM totals:   {coding.usage.totals()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
