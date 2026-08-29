"""RunRecord's on-disk contract: round-tripping and forward/backward
compatibility for logs/runs.jsonl.

runs.jsonl is append-only and permanent -- lines written months apart by
different versions of this code all have to keep reading. records.py had no
dedicated test for that, so adding a field meant hoping nothing broke rather
than knowing. These tests cover the round trip and, specifically, that a line
written before a field existed still loads.

The precedent this follows is `manual_intervention`, which from_json() already
reads with d.get(..., False) for exactly this reason.
"""
import json
from pathlib import Path

import pytest

from agent.records import (
    AggregateMetrics,
    Decision,
    Event,
    FailureKind,
    ResourceUsage,
    RunLog,
    RunRecord,
    SeedMetrics,
    Status,
)


def _full_record(**over):
    """A record with every field populated, so the round trip is not passing
    because most of the shape is None."""
    kwargs = dict(
        iteration=3,
        parent_iteration=2,
        timestamp="2026-01-01T00:00:00+00:00",
        hypothesis="use a pairwise BPR loss",
        diff_path="/runs/attempt_003/config.json",
        status=Status.SUCCESS_AFTER_RETRY,
        seeds=[
            SeedMetrics(seed=0, primary=0.61, gauc=0.66, ndcg5=0.55, epochs_run=7,
                        wall_s=12.5, artifact_dir="/runs/artifacts/iter_3/seed_0"),
            SeedMetrics(seed=1, primary=None, gauc=None, ndcg5=None, epochs_run=None,
                        wall_s=1.0, failure_kind=FailureKind.CRASH,
                        traceback_tail="ValueError: boom"),
        ],
        aggregate=AggregateMetrics(0.61, 0.001, 0.66, 0.55, 1),
        delta_vs_current_best=0.0085,
        decision=Decision.ACCEPT,
        events=[Event(type="eval_finished", detail="primary=0.6100",
                      agent_action="evaluator", timestamp=1.0)],
        resources=ResourceUsage(wall_s=13.5, gpu_s=0.0, tokens_in=1200, tokens_out=800),
        manual_intervention=True,
        patch_path="/runs/attempt_003/changes.patch",
    )
    kwargs.update(over)
    return RunRecord(**kwargs)


# ---------------------------------------------------------------------------
# Round trip.
# ---------------------------------------------------------------------------

def test_full_record_survives_a_json_round_trip():
    original = _full_record()

    restored = RunRecord.from_json(json.loads(json.dumps(original.to_json())))

    assert restored == original


def test_round_trip_through_an_actual_jsonl_file(tmp_path):
    log = RunLog(tmp_path / "runs.jsonl")
    original = _full_record()

    log.append(original)

    assert log.read_all() == [original]


def test_every_dataclass_field_appears_in_to_json():
    """Guards the failure mode where a field is added to the dataclass but not
    to to_json(), so it silently vanishes on write."""
    import dataclasses

    payload = _full_record().to_json()
    for f in dataclasses.fields(RunRecord):
        assert f.name in payload, f"{f.name} is missing from RunRecord.to_json()"


def test_enums_serialize_as_their_values_not_repr():
    payload = _full_record().to_json()

    assert payload["status"] == "success_after_retry"
    assert payload["decision"] == "accept"
    assert payload["seeds"][1]["failure_kind"] == "crash"
    json.dumps(payload)  # must be plain-JSON serialisable


# ---------------------------------------------------------------------------
# Backward compatibility: lines written before a field existed.
# ---------------------------------------------------------------------------

def test_a_line_written_before_patch_path_existed_still_reads():
    """The case that matters for this change. runs.jsonl is append-only, so a
    log from before the field was added must not blow up on read."""
    payload = _full_record().to_json()
    del payload["patch_path"]

    restored = RunRecord.from_json(payload)

    assert restored.patch_path is None
    assert restored.diff_path == "/runs/attempt_003/config.json"   # untouched


def test_a_line_written_before_manual_intervention_existed_still_reads():
    """The precedent patch_path follows -- pinned so it can't regress."""
    payload = _full_record().to_json()
    del payload["manual_intervention"]

    assert RunRecord.from_json(payload).manual_intervention is False


def test_a_line_missing_both_optional_fields_reads():
    payload = _full_record().to_json()
    del payload["patch_path"]
    del payload["manual_intervention"]

    restored = RunRecord.from_json(payload)

    assert restored.patch_path is None
    assert restored.manual_intervention is False


def test_a_seed_written_before_artifact_dir_existed_still_reads():
    """SeedMetrics has the same pattern via from_json's setdefault."""
    payload = _full_record().to_json()
    del payload["seeds"][0]["artifact_dir"]

    assert RunRecord.from_json(payload).seeds[0].artifact_dir is None


def test_an_old_style_log_file_reads_end_to_end(tmp_path):
    """The realistic shape: a whole file of pre-existing lines, none of which
    carry the new key."""
    path = tmp_path / "runs.jsonl"
    lines = []
    for i in (0, 1, 2):
        payload = _full_record(iteration=i).to_json()
        del payload["patch_path"]
        lines.append(json.dumps(payload))
    path.write_text("\n".join(lines) + "\n")

    records = RunLog(path).read_all()

    assert len(records) == 3
    assert all(r.patch_path is None for r in records)


# ---------------------------------------------------------------------------
# patch_path's own semantics.
# ---------------------------------------------------------------------------

def test_patch_path_defaults_to_none_when_not_supplied():
    record = _full_record()
    minimal = RunRecord(
        iteration=1, parent_iteration=None, timestamp=record.timestamp,
        hypothesis="h", diff_path="/c.json", status=Status.SUCCESS,
        seeds=[], aggregate=None, delta_vs_current_best=None, decision=None,
        events=[], resources=ResourceUsage(wall_s=1.0),
    )

    assert minimal.patch_path is None
    assert minimal.to_json()["patch_path"] is None


def test_diff_path_and_patch_path_are_distinct_and_both_preserved():
    """The whole point of the field: the record remembers the settings AND the
    code change, and they are different files."""
    restored = RunRecord.from_json(_full_record().to_json())

    assert restored.diff_path.endswith("config.json")
    assert restored.patch_path.endswith("changes.patch")
    assert restored.diff_path != restored.patch_path
