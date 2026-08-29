"""Load the versioned, train/validation-safe KuaiRand Research profile."""
from __future__ import annotations

import json
from pathlib import Path

from agent.research.retrieval.models import DatasetProfile, RetrievalValidationError
from agent.research.retrieval.safety import ResearchSafetyScanner


DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent / "profiles" / "kuairand_pure.json"
)


def load_dataset_profile(
    path: Path = DEFAULT_PROFILE_PATH,
    *,
    scanner: ResearchSafetyScanner = ResearchSafetyScanner(),
) -> DatasetProfile:
    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalValidationError(
            f"could not load Research dataset profile {profile_path}: {exc}"
        ) from exc
    scanner.scan_value(raw, origin=f"dataset profile {profile_path.name}")
    return DatasetProfile.from_dict(raw)
