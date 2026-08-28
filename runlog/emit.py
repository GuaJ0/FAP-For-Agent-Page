"""Generic append-only JSONL helpers.

Deliberately dict-in/dict-out and independent of agent/records.py: this keeps
the dependency direction one-way (agent.records -> logging.emit, never the
reverse) so logging/ has no idea what a RunRecord is.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator


def append_line(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object as a line. Flushes and fsyncs before returning
    so a crash immediately after this call can't lose the write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, separators=(",", ":"))
    with open(path, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each line as a dict. Missing file yields nothing (empty log is
    the normal state before the first iteration)."""
    if not path.exists():
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
