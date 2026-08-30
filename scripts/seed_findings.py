"""One-time exploration campaign: measure the 7 unexplored directions for real,
so the graded run starts with an honest Do/Don't ledger instead of an empty one.

WHAT THIS IS FOR
----------------
agent/research/findings.jsonl is what stops a fresh graded run from re-spending
iterations on directions this pipeline has already measured. On a clean
checkout it is empty, so the rule in the Research prompt points at nothing.
This script fills it the only way that is honest: by actually running each
direction through the real Coding -> Executor -> Evaluator loop and letting the
resulting RunRecords produce the entries. Nothing here fabricates an outcome.

HOW TO RUN IT
-------------
This needs the REAL Coding and Evaluator agents. It is deliberately NOT the
--offline mode: that mode substitutes a hand-written template library which can
only express ranking-loss variants, so it structurally cannot implement DIN
sequences, multi-task heads, DeepFM, watch-time regression, or time features.
An --offline campaign would produce real-looking records for ideas that were
never actually built.

    # the campaign (needs OPENAI_API_KEY in .env; costs real money)
    python scripts/seed_findings.py --model gpt-5

    # see what it would run, spend nothing
    python scripts/seed_findings.py --dry-run

    # after reviewing the scratch ledger, merge it into the committed one
    python scripts/seed_findings.py --promote

Research stays deterministic (OfflineResearchAgent walks DEFAULT_BACKLOG in
rank order) because the campaign's whole point is covering a known list of
directions rather than letting a model pick. Coding and Evaluator are live.

ISOLATION
---------
Everything this writes is scratch, and none of it counts against the graded run:

  - --root defaults to runs/exploration/, so logs/, solutions/, the registry,
    the orchestrator state and the artifacts all live there. The graded run
    reads repo-root logs/ and never sees any of it.
  - the ledger defaults to <root>/findings.jsonl, NOT the committed
    agent/research/findings.jsonl. --promote is the separate, explicit step
    that merges reviewed findings across. Nothing this run measures reaches
    the graded run's memory until you run it.
  - the iteration count, wall-clock and convergence window are per-run state in
    that scratch logs/, so the campaign cannot consume the graded run's budget.

BUDGET
------
Looser than the graded run on purpose, and only here: a complex idea should be
abandoned because the mechanism failed, not because it ran out of repair
attempts. See EXPLORATION_* below for the values and RetryConfig for the
defaults they deliberately depart from.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_loop  # noqa: E402
from agent.research.findings import (  # noqa: E402
    DEFAULT_FINDINGS_PATH,
    FindingsLedger,
    findings_for_prompt,
)
from agent.research.offline import DEFAULT_BACKLOG  # noqa: E402

# Scratch by default. Nothing the graded run reads.
DEFAULT_EXPLORATION_ROOT = REPO_ROOT / "runs" / "exploration"

# Deliberately looser than agent/config.RetryConfig, and applied ONLY here.
# A structurally complex idea (sequence models, multi-task heads) routinely
# needs more than the graded run's 3 repair attempts before it even compiles;
# abandoning at 3 would record "the mechanism failed" when what actually
# happened is "we ran out of retries", which is exactly the false Don't the
# whole campaign exists to avoid.
EXPLORATION_MAX_FIX_ATTEMPTS = 5
EXPLORATION_IDEA_BACKSTOP_S = 90 * 60.0     # vs 45 min
EXPLORATION_PER_RUN_TIMEOUT_S = 60 * 60.0   # vs 15 min
EXPLORATION_MAX_WALL_S = 24 * 3600.0


def _describe_campaign() -> str:
    from agent.research.findings import resolve_family

    families: dict[str, list] = {}
    for entry in sorted(DEFAULT_BACKLOG, key=lambda e: e.rank):
        families.setdefault(resolve_family(entry.key), []).append(entry)

    lines = [f"{len(DEFAULT_BACKLOG)} rounds across {len(families)} directions:"]
    for family, entries in families.items():
        budget = sum(e.expected_wall_s for e in entries) / 3600.0
        lines.append(f"  {family:<22} {len(entries)} variant(s)  ~{budget:.1f}h")
        for e in entries:
            lines.append(f"      rank {e.rank:>2}  {e.key:<26} {e.title}")
    total = sum(e.expected_wall_s for e in DEFAULT_BACKLOG) / 3600.0
    lines.append(f"  estimated training wall-clock: ~{total:.1f}h (excludes LLM latency)")
    return "\n".join(lines)


def _promote(scratch: Path, target: Path) -> int:
    """Merge a reviewed scratch ledger into the committed one.

    Goes through FindingsLedger.record() rather than concatenating files, so a
    direction already present in the target accumulates attempts instead of
    being overwritten, and every entry is re-scanned for hidden-test material
    on the way in.
    """
    if not scratch.exists():
        print(f"error: no scratch ledger at {scratch}; run the campaign first.", file=sys.stderr)
        return 2

    source = FindingsLedger(scratch).load()
    if not source:
        print(f"error: {scratch} holds no findings to promote.", file=sys.stderr)
        return 2

    ledger = FindingsLedger(target)
    print(f"promoting {len(source)} finding(s)\n  from {scratch}\n  into {target}\n")
    for finding in source:
        merged = ledger.record(finding)
        if merged is None:
            print(f"  SKIPPED  {finding.direction} (failed the validation-only scan)")
        else:
            print(f"  {merged.verdict:<5} {merged.direction:<22} "
                  f"attempts={merged.attempts} confidence={merged.confidence}")
    print(f"\n{target} now holds {len(ledger.load())} finding(s).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_EXPLORATION_ROOT),
                    help="scratch dir for logs/ and solutions/ (default: runs/exploration)")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-iterations", type=int, default=len(DEFAULT_BACKLOG),
                    help="one per backlog entry by default, so every direction gets its round")
    ap.add_argument("--findings-path", default=None,
                    help="scratch ledger to write (default: <root>/findings.jsonl)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the campaign plan and the exact run_loop invocation, then exit")
    ap.add_argument("--promote", action="store_true",
                    help="merge the scratch ledger into the committed one and exit; runs nothing")
    ap.add_argument("--promote-into", default=str(DEFAULT_FINDINGS_PATH),
                    help="ledger --promote writes into (default: the committed findings.jsonl)")
    args, passthrough = ap.parse_known_args()

    root = Path(args.root)
    findings_path = Path(args.findings_path) if args.findings_path else root / "findings.jsonl"

    if args.promote:
        return _promote(findings_path, Path(args.promote_into))

    argv = [
        "--root", str(root),
        "--model", args.model,
        "--seeds", str(args.seeds),
        "--max-iterations", str(args.max_iterations),
        "--findings-path", str(findings_path),
        "--timeout-s", str(EXPLORATION_PER_RUN_TIMEOUT_S),
        "--max-fix-attempts", str(EXPLORATION_MAX_FIX_ATTEMPTS),
        "--idea-backstop-s", str(EXPLORATION_IDEA_BACKSTOP_S),
        "--max-wall-s", str(EXPLORATION_MAX_WALL_S),
        *passthrough,
    ]

    print(_describe_campaign())
    print(f"\n  scratch root:   {root}")
    print(f"  scratch ledger: {findings_path}")
    print(f"  graded ledger:  {DEFAULT_FINDINGS_PATH}  (untouched until --promote)")
    print(f"\n  run_loop argv:  {' '.join(argv)}\n")

    if args.dry_run:
        print("dry run -- nothing executed, nothing spent.")
        return 0

    # Live Coding + Evaluator, deterministic backlog Research. NOT --offline:
    # the template library cannot implement most of these directions.
    sys.argv = ["run_loop.py", *argv]
    status = run_loop.main()

    ledger = FindingsLedger(findings_path)
    stored = ledger.load()
    print(f"\n=== campaign ledger: {len(stored)} direction(s) measured ===")
    for entry in findings_for_prompt(stored):
        print(f"  {entry['verdict']:<5} {entry['direction']:<22} "
              f"attempts={entry['attempts']} confidence={entry['confidence']} "
              f"best_delta={entry['delta_vs_incumbent']}")
    print(f"\nreview {findings_path}, then:  python scripts/seed_findings.py --promote")
    return status


if __name__ == "__main__":
    sys.exit(main())
