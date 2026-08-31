"""Config-only sweeps: rerun ONE already-generated implementation at several
config points.

WHY THIS IS NOT THE ORCHESTRATOR
--------------------------------
Orchestrator._step() runs research.propose() -> coding.implement() ->
executor.run_seeds() -> evaluator.judge(). Every pass generates NEW code, so
two iterations of the same direction differ in both implementation and
settings and you cannot attribute the difference to either.

This module deliberately removes the first two stages. The implementation is
fixed -- one solution_dir, generated once -- and only the config changes. That
isolates hyperparameter sensitivity within a single implementation, which is
the one question the main loop structurally cannot answer.

It is a separate mode, not a flag on the main loop: no proposal is made, no
code is written, no idea is abandoned or retried, and the tier-1/tier-2 policy
does not apply because there is nothing to repair. Folding it into _step()
would mean threading "but skip these two stages" through a control path whose
entire job is running them.

WHAT IT REUSES, UNCHANGED
-------------------------
Executor.run_seeds(), EvaluatorAgent.judge(), RunLog.append() and RunRecord --
all called through their public interfaces, none modified. A sweep RunRecord
is a real RunRecord written by the real Evaluator through the normal path; it
is not a synthetic or downgraded record.

WHAT IT DELIBERATELY DOES NOT TOUCH
-----------------------------------
  - CheckpointRegistry: a sweep never registers a checkpoint and so can never
    move the incumbent. Deltas are measured against an incumbent passed in by
    the caller. A config sweep is a measurement, not a promotion path.
  - StateStore / OrchestratorState: no resumable idea state exists here.
  - The solution dir: generated configs are written under the sweep's own
    output directory, so the implementation being swept stays byte-identical
    across every point.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agent.agents import EvaluatorAgent
from agent.config import Config, DEFAULT_CONFIG
from agent.executor import Executor
from agent.records import (
    Decision,
    Event,
    ResourceUsage,
    RunLog,
    RunRecord,
    Status,
)


class ConfigSweepError(RuntimeError):
    """The sweep as specified cannot be run."""


def load_flat_config(path: Path) -> dict[str, Any]:
    """Read a flat solution config. Mirrors solution/train.py's load_config.

    Kept as a small local reader rather than importing from solution/train.py:
    that file is the thing under test and gets replaced by generated code, so
    importing it would couple the sweep driver to whatever the Coding Agent
    most recently wrote.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        out: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, raw = line.partition(":")
            out[key.strip()] = _coerce(raw.strip())
        return out


def _coerce(raw: str) -> Any:
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


@dataclass(frozen=True)
class ConfigPoint:
    """One point in the sweep: a label and the keys it overrides."""

    label: str                       # short, e.g. "lr=0.003"
    overrides: Mapping[str, Any]     # merged over the base config

    def describe(self) -> str:
        return ", ".join(f"{k}={v!r}" for k, v in sorted(self.overrides.items()))


@dataclass(frozen=True)
class SweepPointResult:
    point: ConfigPoint
    record: RunRecord
    config_path: Path

    @property
    def primary(self) -> Optional[float]:
        return self.record.aggregate.primary_mean if self.record.aggregate else None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConfigSweep:
    """Rerun one fixed implementation across several config points.

    Every point produces a real RunRecord judged by the real Evaluator and
    appended to the real RunLog -- the same artifacts the main loop produces,
    minus the propose/implement stages.
    """

    executor: Executor
    evaluator: EvaluatorAgent
    run_log: RunLog
    cfg: Config = DEFAULT_CONFIG
    out_dir: Optional[Path] = None      # where generated configs go; defaults under logs/
    seeds: Sequence[int] = (0, 1)
    _points: list[SweepPointResult] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.out_dir is None:
            self.out_dir = self.cfg.paths.logs_dir / "sweeps"
        self.out_dir = Path(self.out_dir)

    def materialise_config(self, base_config: Path, point: ConfigPoint, dest_dir: Path) -> Path:
        """Write base_config + point.overrides to a fresh config.json.

        JSON rather than YAML on purpose: solution/train.py's load_config()
        dispatches on the .json suffix and parses it with the stdlib, so a
        generated config needs no PyYAML and cannot be mangled by the fallback
        flat-YAML reader.
        """
        merged = dict(load_flat_config(Path(base_config)))
        unknown = set(point.overrides) - set(merged)
        if unknown:
            # Not fatal -- a generated implementation may legitimately read
            # keys the base config never set -- but worth saying out loud,
            # because a typo'd key is otherwise a silently ignored sweep point.
            print(f"[sweep] note: {point.label} sets key(s) absent from the base "
                  f"config: {sorted(unknown)}", flush=True)
        merged.update(point.overrides)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "config.json"
        dest.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return dest

    def run(
        self,
        *,
        solution_dir: Path,
        base_config: Path,
        points: Sequence[ConfigPoint],
        hypothesis: str,
        incumbent_primary: Optional[float] = None,
        start_iteration: int = 1,
        parent_iteration: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> list[SweepPointResult]:
        """Run every point. Returns one result per point, in order.

        `incumbent_primary` is the bar deltas are measured against. Passed in
        rather than read from a CheckpointRegistry so a sweep can never be the
        thing that moves the incumbent -- see the module docstring.
        """
        solution_dir = Path(solution_dir)
        if not (solution_dir / "train.py").is_file():
            raise ConfigSweepError(f"no train.py in solution_dir {solution_dir}")
        if not points:
            raise ConfigSweepError("sweep needs at least one ConfigPoint")

        results: list[SweepPointResult] = []
        for offset, point in enumerate(points):
            iteration = start_iteration + offset
            dest_dir = Path(self.out_dir) / f"point_{iteration:03d}_{_slug(point.label)}"
            config_path = self.materialise_config(base_config, point, dest_dir)

            seeds, agg = self.executor.run_seeds(
                solution_dir, config_path, iteration, list(self.seeds), timeout_s,
            )
            record = self._build_record(
                iteration=iteration,
                parent_iteration=parent_iteration,
                hypothesis=hypothesis,
                point=point,
                config_path=config_path,
                seeds=seeds,
                agg=agg,
                incumbent_primary=incumbent_primary,
            )
            # A sweep point that crashed on every seed is a technical failure,
            # exactly as in the main loop: no aggregate means nothing for the
            # Evaluator to judge, so it is logged unjudged rather than being
            # handed a record with no metrics in it.
            if agg is not None:
                self._judge(record)
            self.run_log.append(record)

            result = SweepPointResult(point=point, record=record, config_path=config_path)
            results.append(result)
            self._points.append(result)
        return results

    def _build_record(
        self,
        *,
        iteration: int,
        parent_iteration: Optional[int],
        hypothesis: str,
        point: ConfigPoint,
        config_path: Path,
        seeds: list,
        agg,
        incumbent_primary: Optional[float],
    ) -> RunRecord:
        if agg is None:
            status, delta = Status.FAILED, None
        else:
            status = Status.SUCCESS
            delta = (
                agg.primary_mean - incumbent_primary
                if incumbent_primary is not None else agg.primary_mean
            )
        events = [Event(
            type="config_sweep_point",
            detail=f"label={point.label} overrides={point.describe()}",
            agent_action="orchestrator",
        )]
        if agg is not None:
            events.append(Event(
                type="eval_finished", detail=f"primary={agg.primary_mean:.4f}",
                agent_action="evaluator",
            ))
        return RunRecord(
            iteration=iteration,
            parent_iteration=parent_iteration,
            timestamp=_iso_now(),
            hypothesis=hypothesis,
            diff_path=str(config_path),
            status=status,
            seeds=seeds,
            aggregate=agg,
            delta_vs_current_best=delta,
            decision=None,
            events=events,
            resources=ResourceUsage(
                wall_s=sum(s.wall_s for s in seeds),
                cpu_hours=sum(s.cpu_s for s in seeds) / 3600.0,
            ),
        )

    def _judge(self, record: RunRecord) -> None:
        """Run the real Evaluator over this point and fold its verdict in.

        Mirrors Orchestrator._handle_successful_run's evaluator block, including
        the ResourceUsage rebuild (ResourceUsage is frozen), so a sweep record
        carries the same fields a loop record does.
        """
        history = [r.record for r in self._points]
        verdict = self.evaluator.judge(record, history)
        record.decision = verdict.decision
        if verdict.commentary:
            record.events.append(Event(
                type="evaluator_commentary", detail=verdict.commentary, agent_action="evaluator",
            ))
        if verdict.usage is not None:
            r = record.resources
            record.resources = ResourceUsage(
                wall_s=r.wall_s, cpu_hours=r.cpu_hours,
                tokens_in=r.tokens_in + verdict.usage.tokens_in,
                tokens_out=r.tokens_out + verdict.usage.tokens_out,
            )


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in text]
    return "".join(keep).strip("-")[:40] or "point"
