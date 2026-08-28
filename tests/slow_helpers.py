"""Gates for tests that need the real dataset or a real (billable) API call.

The default suite stays deterministic, offline and fast -- these opt in:

    RUN_SLOW_TESTS=1 KUAIRAND_PATH=/path/to/KuaiRand-Pure/data pytest -m slow
    RUN_LLM_TESTS=1  OPENAI_API_KEY=sk-...                     pytest -m llm
"""
import os
from pathlib import Path

import pytest

REQUIRED_CSVS = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "video_features_basic_pure.csv",
)


def kuairand_path():
    p = os.environ.get("KUAIRAND_PATH")
    return Path(p) if p else None


def _data_ready():
    p = kuairand_path()
    return p is not None and p.is_dir() and all((p / c).exists() for c in REQUIRED_CSVS)


requires_data = pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1" or not _data_ready(),
    reason="needs RUN_SLOW_TESTS=1 and KUAIRAND_PATH pointing at the KuaiRand-Pure CSVs",
)

requires_openai = pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="needs RUN_LLM_TESTS=1 and OPENAI_API_KEY (makes a real, billable call)",
)
