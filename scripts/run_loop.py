"""Run the agent loop end to end.

Wires the Research, Coding, and Evaluator agents into the existing
Orchestrator. OfflineResearchAgent is the default Research implementation and
selects a history-aware proposal from its validated offline backlog.
--live-research instead enables LLMResearchAgent using the same live client as
Coding and Evaluator. Here "live Research" means LLM-generated proposals
grounded in the bundled citation catalog; it does not perform scholarly/web
retrieval.

--offline keeps Research deterministic and additionally uses
FakeEvaluatorAgent (the same deterministic margin rule LLMEvaluatorAgent falls
back to internally) and the Coding agent's hand-written template library, so it
makes no API calls.

    # offline: no API key, no spend, uses the hand-written template library
    python scripts/run_loop.py --offline

    # live Coding/Evaluator, deterministic offline Research (the default)
    # (needs OPENAI_API_KEY in .env)
    python scripts/run_loop.py --model gpt-5

    # live Research/Coding/Evaluator sharing one client; bundled citations only
    python scripts/run_loop.py --live-research --model gpt-5

The legacy --hypothesis flag remains accepted for CLI compatibility, but the
selected Research agent produces the proposal.

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

from agent.agents import Diff, FakeEvaluatorAgent, Idea  # noqa: E402
from agent.coding import LLMCodingAgent, OpenAIClient, TemplateLibraryClient  # noqa: E402
from agent.config import (  # noqa: E402
    Config,
    ConvergenceConfig,
    ExecutorConfig,
    Paths,
    RetryConfig,
    SeedingConfig,
)
from agent.evaluator import LLMEvaluatorAgent  # noqa: E402
from agent.executor import Executor  # noqa: E402
from agent.orchestrator import BootstrapError, Orchestrator  # noqa: E402
from agent.records import RunLog  # noqa: E402
from agent.research import LLMResearchAgent, OfflineResearchAgent  # noqa: E402
from agent.research.findings import DEFAULT_FINDINGS_PATH, FindingsLedger  # noqa: E402
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
        # getattr for the stall rule, as for retry below: build_config is shared
        # with scripts/config_sweep.py, which declares neither flag.
        convergence=ConvergenceConfig(
            max_iterations=args.max_iterations,
            max_wall_s=args.max_wall_s,
            epsilon=getattr(args, "epsilon", ConvergenceConfig.epsilon),
            n_window=getattr(args, "stall_window", ConvergenceConfig.n_window),
        ),
        # Overridable so an exploration pass can buy more repair attempts and a
        # longer per-idea backstop than the graded run, without editing the
        # defaults in agent/config.py that the graded run relies on.
        # getattr, not args.x: build_config is shared with scripts/config_sweep.py,
        # which runs no retry loop and so defines neither flag. A caller that
        # does not care about the repair policy gets exactly RetryConfig's
        # defaults rather than having to declare arguments it never uses.
        retry=RetryConfig(
            max_fix_attempts=getattr(args, "max_fix_attempts", RetryConfig.max_fix_attempts),
            idea_time_backstop_s=getattr(args, "idea_backstop_s", RetryConfig.idea_time_backstop_s),
        ),
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
    """Entry point. Restores os.environ on the way out.

    _main() exports KUAIRAND_PATH so the executor's training subprocesses
    inherit it, and load_dotenv() injects .env values. Both are process-global.
    Those subprocesses are spawned *inside* _main(), so they still see
    everything they need -- the restore happens only after it returns.

    For the CLI that is a no-op: the process exits immediately afterwards. It
    matters for a caller that runs main() in-process, which the tests do:
    without this, one run's data dir stays exported and later, unrelated work
    resolves KUAIRAND_PATH to a directory that run had created and finished
    with. The failure surfaces far from its cause, looking like a broken
    solution rather than a leaked global.
    """
    saved_env = dict(os.environ)
    try:
        return _main()
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypothesis", default=DEFAULT_HYPOTHESIS)
    ap.add_argument("--offline", action="store_true",
                    help="use the hand-written template library instead of an LLM (no spend)")
    ap.add_argument(
        "--live-research",
        action="store_true",
        help=("use LLM-backed Research proposal generation grounded in the bundled "
              "citation catalog; requires live Coding/Evaluator mode"),
    )
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
    # RetryConfig overrides. Defaults mirror agent/config.RetryConfig exactly, so
    # omitting both flags reproduces the graded run's policy byte for byte.
    ap.add_argument("--max-fix-attempts", type=int, default=RetryConfig.max_fix_attempts,
                    help="tier-1 repair attempts on one idea before abandoning it. Raise for an "
                         "exploration pass so a complex idea is not abandoned over a fixable bug.")
    ap.add_argument("--idea-backstop-s", type=float, default=RetryConfig.idea_time_backstop_s,
                    help="abandon an idea past this wall time regardless of attempt count.")
    ap.add_argument("--epsilon", type=float, default=ConvergenceConfig.epsilon,
                    help="minimum validation-primary improvement that does not count as a stall.")
    ap.add_argument("--stall-window", type=int, default=ConvergenceConfig.n_window,
                    help="consecutive stalled iterations before stopping. Raise it above the "
                         "number of ideas to be covered when the goal is COVERAGE rather than "
                         "convergence -- an exploration pass expects most directions to fail, so "
                         "the stall rule would end it long before the backlog is exhausted.")
    ap.add_argument("--findings-path", default=None,
                    help="cross-run Do/Don't ledger to write. Defaults to the committed "
                         "agent/research/findings.jsonl. Point this at a scratch file to keep an "
                         "exploration pass out of the ledger the graded run reads.")
    args = ap.parse_args()

    if args.offline and args.live_research:
        ap.error("--offline and --live-research are incompatible; live Research requires a live LLM client")

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

    # All live agents share this exact client instance. Offline Research stays
    # the default and uses no LLM even when Coding/Evaluator are live.
    if args.live_research:
        research = LLMResearchAgent(
            llm=client,
            usage_log_path=cfg.paths.logs_dir / "research_agent_usage.jsonl",
            convergence=cfg.convergence,
        )
    else:
        research = OfflineResearchAgent(convergence=cfg.convergence)

    coding = LLMCodingAgent(
        work_dir=root / "solutions",
        data_dir=data_dir,
        llm=client,
        usage_log_path=cfg.paths.logs_dir / "coding_agent_usage.jsonl",
        base_config={"loss": args.loss, "epochs": args.epochs, "patience": 3},
        # ACCUMULATION: build each new idea on the best accepted solution so
        # far rather than always on the static solution/train.py. Right after
        # bootstrapping these resolve to iteration 0, i.e. solution/ itself.
        registry_path=cfg.paths.registry_json,
        run_log_path=cfg.paths.runs_jsonl,
    )
    # Same client as the CodingAgent when live: one model, one API key, no
    # separate --evaluator-model flag to keep in sync. --offline keeps the
    # deterministic margin rule instead of spending on judge calls too.
    evaluator = FakeEvaluatorAgent() if args.offline else LLMEvaluatorAgent(
        llm=client, usage_log_path=cfg.paths.logs_dir / "evaluator_usage.jsonl",
    )
    # Cross-run Do/Don't ledger. Deliberately NOT under --root by default: a
    # reset here archives logs/ wholesale, and this has to outlive that. An
    # exploration pass overrides it so its findings can be reviewed before they
    # reach the ledger the graded run actually reads.
    findings = FindingsLedger(Path(args.findings_path) if args.findings_path else DEFAULT_FINDINGS_PATH)
    print(f"[findings] ledger: {findings.path}")

    orc = Orchestrator(
        research=research,
        coding=coding,
        evaluator=evaluator,
        executor=Executor(cfg=cfg),
        run_log=RunLog(cfg.paths.runs_jsonl),
        registry=CheckpointRegistry(cfg.paths.registry_json),
        state_store=StateStore(cfg.paths.orchestrator_state),
        cfg=cfg,
        findings=findings,
    )

    t0 = time.time()

    if not args.skip_baseline:
        # Establish solution/ as a real iteration 0 before any hypothesis runs,
        # so deltas, ACCEPT/REVERT and convergence all have an incumbent to
        # measure against. Idempotent: a no-op if a previous run already did it.
        #
        # config_path is solution/config.yaml, a sibling of solution/train.py --
        # that adjacency is what lets a registry entry be resolved back to the
        # source that produced it (see LLMCodingAgent._current_best_source).
        # No patch_path: the baseline is a pre-existing solution, not an edit.
        baseline = Diff(
            config_path=str(REPO_ROOT / "solution" / "config.yaml"),
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
    # getattr: the wiring tests substitute a minimal Orchestrator stub whose
    # only contract is run(). Reporting why a run stopped must not require
    # every stand-in to grow internal state to stay callable.
    if getattr(getattr(orc, "state", None), "research_exhausted", False):
        # The natural end of an exploration pass: every idea the backlog holds
        # has been tried. Orchestrator stops the loop cleanly for this (see
        # _handle_research_exhausted) rather than recording it as a failure to
        # propose, so there is nothing to catch here -- only to report.
        print(f"\n[research] nothing left to propose -- stopped cleanly: "
              f"{orc.state.research_exhausted_reason}")

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
        for e in r.events:
            if e.type == "evaluator_commentary":
                print(f"  evaluator: {e.detail}")

    print(f"\nrun log:          {cfg.paths.runs_jsonl}")
    print(f"coding LLM usage: {cfg.paths.logs_dir / 'coding_agent_usage.jsonl'}")
    print(f"quarantine:       {cfg.paths.test_metrics_jsonl}  (agents never read this)")
    print(f"coding totals:    {coding.usage.totals()}")
    research_usage = getattr(research, "usage", None)  # OfflineResearchAgent has none
    if research_usage is not None:
        print(f"research usage:   {cfg.paths.logs_dir / 'research_agent_usage.jsonl'}")
        print(f"research totals:  {research_usage.totals()}")
    evaluator_usage = getattr(evaluator, "usage", None)  # FakeEvaluatorAgent (--offline) has none
    if evaluator_usage is not None:
        print(f"evaluator usage:  {cfg.paths.logs_dir / 'evaluator_usage.jsonl'}")
        print(f"evaluator totals: {evaluator_usage.totals()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
