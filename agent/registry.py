"""Checkpoint registry: iteration -> checkpoint path + validation score.

"Validation-best" needs to be an O(1) lookup the orchestrator can hit on
every iteration (for delta_vs_current_best and for adaptive decisions),
not a scan over logs/runs.jsonl. Persisted to JSON so it survives a crash
alongside the orchestrator's own state file.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class RegistryEntry:
    iteration: int
    checkpoint_path: str
    val_primary: float


class CheckpointRegistry:
    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[int, RegistryEntry] = {}
        self._best_iteration: Optional[int] = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        for d in data["entries"]:
            e = RegistryEntry(**d)
            self._entries[e.iteration] = e
        self._best_iteration = data.get("best_iteration")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": [asdict(e) for e in self._entries.values()],
            "best_iteration": self._best_iteration,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)  # atomic on POSIX: a crash mid-write can't corrupt the live file

    def register(self, iteration: int, checkpoint_path: str, val_primary: float) -> None:
        entry = RegistryEntry(iteration, checkpoint_path, val_primary)
        self._entries[iteration] = entry
        if self._best_iteration is None or val_primary > self._entries[self._best_iteration].val_primary:
            self._best_iteration = iteration
        self._save()

    def get(self, iteration: int) -> Optional[RegistryEntry]:
        return self._entries.get(iteration)

    def best(self) -> Optional[RegistryEntry]:
        if self._best_iteration is None:
            return None
        return self._entries[self._best_iteration]
