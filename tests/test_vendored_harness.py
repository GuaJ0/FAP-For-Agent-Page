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


# ---------------------------------------------------------------------------
# dataset.py: the extra log columns, added from OUTSIDE the vendored files.
#
# The gap this closes: data.load() exposes 7 of the log's 19 columns. In the
# first exploration campaign the watch-time and multi-task solutions looked for
# play_time_ms / is_click, found nothing, silently skipped their auxiliary head
# and reported the unchanged baseline's score -- which the ledger recorded as
# the direction having failed. Three watch-time variants rolled up to a
# "well_tested" dead end on evidence that no watch-time model had ever run.
# ---------------------------------------------------------------------------

def test_the_extension_does_not_live_in_a_vendored_file():
    """It must add from outside, or the hash guard above becomes unmaintainable
    and the line between the competition's code and ours disappears."""
    assert "load_full" not in (HARNESS / "data.py").read_text()
    assert (HARNESS / "dataset.py").exists()


def test_the_extension_reuses_the_official_split_definition():
    """Not a copy of SPLITS -- an import of it. Two definitions of where the
    held-out boundary sits is exactly how a solution ends up training on
    validation rows, winning on validation, and collapsing on test."""
    src = (HARNESS / "dataset.py").read_text()
    assert "from data import" in src and "SPLITS" in src
    assert "20220408" not in src, "dataset.py restates split dates instead of importing them"


def test_every_extra_column_exists_in_the_log_header():
    """A typo'd column name would parse as 0 for every row and look like real
    data -- the failure mode is silent, so it is pinned here."""
    import sys
    sys.path.insert(0, str(HARNESS))
    import dataset

    header = {
        "user_id", "video_id", "date", "hourmin", "time_ms", "is_click", "is_like",
        "is_follow", "is_comment", "is_forward", "is_hate", "long_view",
        "play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time",
        "is_profile_enter", "is_rand", "tab",
    }
    assert set(dataset.EXTRA_COLUMNS) <= header
    # and it must not re-expose what load() already returns
    assert not set(dataset.EXTRA_COLUMNS) & set(dataset.BASE_FIELDS)


def test_the_positional_row_contract_is_preserved():
    """encode() and every generated train.py index rows positionally: x[5] is
    duration_ms, x[6] is the label. Extras append after that, never insert."""
    import sys
    sys.path.insert(0, str(HARNESS))
    import dataset

    assert dataset.BASE_FIELDS[:7] == (
        "date", "user_id", "video_id", "author_id", "tab", "duration_ms", "long_view")


def test_an_unknown_column_is_rejected_rather_than_silently_ignored(tmp_path):
    import sys
    sys.path.insert(0, str(HARNESS))
    import dataset

    with pytest.raises(ValueError, match="unknown column"):
        dataset.load_full(str(tmp_path), columns=("play_tiem_ms",))   # typo


def test_malformed_cells_do_not_take_down_a_run():
    import sys
    sys.path.insert(0, str(HARNESS))
    import dataset

    assert dataset._parse("", int) == 0
    assert dataset._parse(None, float) == 0.0
    assert dataset._parse("not a number", float) == 0.0
    assert dataset._parse("42", int) == 42


def test_the_coding_agent_vendors_and_permits_the_extension():
    """It has to reach the solution dir and pass the import check, or a
    solution that uses it fails before it runs."""
    from agent.coding.agent import VENDORED_FILES, VENDORED_IMPORTS

    assert "dataset.py" in VENDORED_FILES
    assert "dataset" in VENDORED_IMPORTS
