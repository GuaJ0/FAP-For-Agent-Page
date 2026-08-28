"""The vendored starter-kit files must stay byte-identical to what was copied.

`harness/evaluate.py` is the competition's sole scoring authority and
`agent/verification.py` re-scores every run through it. If someone "fixes" a
line in there, every number this repo has ever produced silently changes
meaning. Pinning the hashes makes that a failing test instead of a mystery.

Update a hash here only when deliberately re-vendoring from the starter kit,
and update harness/VENDORED.md in the same commit.
"""
import hashlib
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent / "harness"

EXPECTED = {
    "evaluate.py": "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de",
    "data.py": "1bf54f5f3a9f590eab2f87f09a3c27422031867a20a5328d56cbd8c7db36e541",
    "baseline.py": "c8f7fc60178413e247e78bb231e7550eeef52101b6493fcf1a4d2b0e5fe18f8a",
    "baseline_scores.json": "950f98181770c030a68bdddab7be3c0abbf060531f54455a6a6f81a4cb003324",
}


@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_vendored_file_is_unmodified(name, expected):
    path = HARNESS / name
    assert path.exists(), f"vendored {name} is missing from harness/"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (
        f"harness/{name} has been modified. These files are verbatim copies of the "
        f"official starter kit and must not be edited -- evaluate.py in particular "
        f"defines what every score in this repo means. If the change is a deliberate "
        f"re-vendor, update EXPECTED here and harness/VENDORED.md together."
    )


def test_published_baseline_numbers_are_available_to_tests():
    """Guards against harness/baseline_scores.json being dropped: the FM bar
    (0.5946) and the seed std (0.0008) are what the slow end-to-end tests
    assert iteration 0 against."""
    import json

    scores = json.loads((HARNESS / "baseline_scores.json").read_text())
    fm = scores["scores"]["fm_official"]
    assert fm["test"]["primary"] == pytest.approx(0.5946)
    assert fm["valid"]["primary"] == pytest.approx(0.6016)
    assert fm["std_over_5_seeds"]["test_primary"] == pytest.approx(0.0008)
