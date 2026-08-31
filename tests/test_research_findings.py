"""Cross-run Do/Don't ledger (agent/research/findings.py).

The gap this closes: build_research_context reads only the current run's
RunRecord history, so a run whose logs/runs.jsonl starts empty has no memory of
directions the pipeline already measured. In the graded practice run that cost
an iteration re-proposing a ranking loss we had already measured as a loser.
"""
import json
from pathlib import Path

import pytest

from agent.records import (
    AggregateMetrics,
    Decision,
    Event,
    ResourceUsage,
    RunRecord,
    Status,
)
from agent.research.context import build_research_context
from agent.research.findings import (
    DEFAULT_FINDINGS_PATH,
    VERDICT_DO,
    VERDICT_DONT,
    Finding,
    FindingsLedger,
    build_finding,
    findings_for_prompt,
)

HANDOFF = """[RESEARCH_PROPOSAL v1]
ID: {hid}
TITLE: {title}
PARENT ITERATION: 0

HYPOTHESIS:
{hypothesis}

WHY THIS SHOULD HELP:
mechanism
"""


def _record(iteration=1, hid="RP-001", decision=Decision.REVERT, primary=0.5989,
            delta=-0.0027, commentary="Lost to the incumbent on validation primary.",
            hypothesis="Replace pointwise logloss with a pairwise BPR ranking loss."):
    events = []
    if commentary:
        events.append(Event(type="evaluator_commentary", detail=commentary,
                            agent_action="evaluator"))
    return RunRecord(
        iteration=iteration, parent_iteration=0, timestamp="2026-01-01T00:00:00+00:00",
        hypothesis=HANDOFF.format(hid=hid, title="a title", hypothesis=hypothesis),
        diff_path="/c.json", status=Status.SUCCESS, seeds=[],
        aggregate=AggregateMetrics(primary, 0.0, primary, primary, 1) if primary else None,
        delta_vs_current_best=delta, decision=decision, events=events,
        resources=ResourceUsage(wall_s=1.0),
    )


def _finding(direction="RP-001", verdict=VERDICT_DONT, delta=-0.0027, iteration=1):
    return Finding(
        direction=direction, title="a direction", verdict=verdict,
        decision="revert" if verdict == VERDICT_DONT else "accept",
        delta_vs_incumbent=delta, validation_primary=0.5989,
        why="why it lost", iteration=iteration,
    )


# ---------------------------------------------------------------------------
# Building an entry from a judged record -- no LLM, all extraction.
# ---------------------------------------------------------------------------

def test_a_reverted_iteration_becomes_a_dont_with_its_delta_and_reason():
    f = build_finding(_record(), direction="RP-001", title="pairwise BPR ranking loss")

    assert f.verdict == VERDICT_DONT
    assert f.decision == "revert"
    assert f.delta_vs_incumbent == pytest.approx(-0.0027)
    assert f.validation_primary == pytest.approx(0.5989)
    assert "Lost to the incumbent" in f.why       # the Evaluator's own words
    assert f.iteration == 1


def test_an_accepted_iteration_becomes_a_do():
    f = build_finding(_record(decision=Decision.ACCEPT, delta=0.0185),
                      direction="RP-002", title="t")
    assert f.verdict == VERDICT_DO


def test_an_abandoned_iteration_becomes_a_dont():
    f = build_finding(_record(decision=Decision.ABANDON), direction="RP-003", title="t")
    assert f.verdict == VERDICT_DONT


def test_an_unjudged_record_produces_nothing():
    """Technical abandons have no Evaluator judgment. A crash means the Coding
    agent couldn't build the idea -- not evidence the direction is dead."""
    assert build_finding(_record(decision=None), direction="RP-004", title="t") is None


def test_missing_commentary_is_tolerated():
    f = build_finding(_record(commentary=""), direction="RP-005", title="t")
    assert f.why == ""


def test_long_text_is_truncated_so_entries_stay_small():
    f = build_finding(_record(commentary="x" * 5000), direction="RP-006", title="y" * 5000)
    assert len(f.why) <= 280
    assert len(f.title) <= 120


# ---------------------------------------------------------------------------
# Persistence, dedup, cap.
# ---------------------------------------------------------------------------

def test_a_finding_survives_a_new_ledger_over_the_same_file(tmp_path):
    """The whole point: it outlives the process, and the run's logs/."""
    path = tmp_path / "findings.jsonl"
    FindingsLedger(path).record(_finding())

    assert [f.direction for f in FindingsLedger(path).load()] == ["RP-001"]


def test_one_entry_per_direction_updates_rather_than_appends(tmp_path):
    """Retesting a direction deepens its single entry instead of appending.

    The headline delta is the family's BEST attempt, not whichever attempt ran
    last. Once an entry rolls several attempts up (see the family tests below),
    "the delta" has to mean something stable about the direction; last-write
    would make it depend on scheduling order, so a direction would look worse
    purely for having been retried in an unlucky order.
    """
    path = tmp_path / "findings.jsonl"
    ledger = FindingsLedger(path)
    ledger.record(_finding(delta=-0.004, iteration=1))
    ledger.record(_finding(delta=-0.001, iteration=7))    # same direction, retested

    stored = ledger.load()
    assert len(stored) == 1
    assert stored[0].delta_vs_incumbent == pytest.approx(-0.001)   # best, not last
    assert stored[0].iteration == 7                                # most recent


def test_distinct_directions_coexist(tmp_path):
    ledger = FindingsLedger(tmp_path / "f.jsonl")
    ledger.record(_finding(direction="RP-001"))
    ledger.record(_finding(direction="RP-002", verdict=VERDICT_DO, delta=0.02))

    assert {f.direction for f in ledger.load()} == {"RP-001", "RP-002"}


def test_the_cap_keeps_the_strongest_evidence(tmp_path):
    ledger = FindingsLedger(tmp_path / "f.jsonl", max_findings=3)
    for i, delta in enumerate([-0.0001, -0.05, -0.0002, -0.04, -0.03]):
        ledger.record(_finding(direction=f"RP-{i}", delta=delta, iteration=i))

    kept = {f.direction for f in ledger.load()}
    assert len(kept) == 3
    # The three largest |delta| survive; the two marginal ones are evicted.
    assert kept == {"RP-1", "RP-3", "RP-4"}


def test_a_corrupt_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "f.jsonl"
    FindingsLedger(path).record(_finding())
    with path.open("a") as fh:
        fh.write("{not json\n")

    assert len(FindingsLedger(path).load()) == 1


def test_a_missing_file_loads_as_empty(tmp_path):
    assert FindingsLedger(tmp_path / "nope.jsonl").load() == ()


def test_writes_are_atomic_leaving_no_tmp_file(tmp_path):
    path = tmp_path / "f.jsonl"
    FindingsLedger(path).record(_finding())
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Safety: structurally incapable of carrying hidden-test material.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("poison", [
    "test_primary was 0.61",
    "the hidden test split shows improvement",
    "TEST_METRICS: {...}",
])
def test_a_poisoned_finding_is_refused_on_write(tmp_path, poison):
    """An Evaluator only ever sees a RunRecord, so it cannot quote a test
    number -- but the ledger refuses one anyway rather than trusting that."""
    path = tmp_path / "f.jsonl"
    ledger = FindingsLedger(path)

    stored = ledger.record(Finding(
        direction="RP-X", title="t", verdict=VERDICT_DONT, decision="revert",
        delta_vs_incumbent=-0.01, validation_primary=0.59, why=poison, iteration=1,
    ))

    assert stored is None
    assert ledger.load() == ()


def test_a_poisoned_line_already_on_disk_is_not_loaded(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text(json.dumps({
        "direction": "RP-X", "title": "t", "verdict": "dont", "decision": "revert",
        "delta_vs_incumbent": -0.01, "validation_primary": 0.59,
        "why": "test_gauc was 0.66", "iteration": 1,
    }) + "\n")

    assert FindingsLedger(path).load() == ()


def test_findings_pass_the_research_agents_fail_closed_check(tmp_path):
    """Findings ride inside ResearchContext, so the existing prompt-time guard
    re-checks them. This proves the guard actually sees them."""
    from agent.research.agent import ResearchInputError, _assert_validation_only_context

    clean = build_research_context([], prior_findings=findings_for_prompt([_finding()]))
    _assert_validation_only_context(clean)          # must not raise

    poisoned = build_research_context([], prior_findings=(
        {"direction": "RP-X", "why": "hidden test primary 0.61"},
    ))
    with pytest.raises(ResearchInputError):
        _assert_validation_only_context(poisoned)


# ---------------------------------------------------------------------------
# Ordering into the prompt.
# ---------------------------------------------------------------------------

def test_donts_lead_and_strongest_evidence_first():
    rendered = findings_for_prompt([
        _finding(direction="D-weak", verdict=VERDICT_DONT, delta=-0.001),
        _finding(direction="A", verdict=VERDICT_DO, delta=0.02),
        _finding(direction="D-strong", verdict=VERDICT_DONT, delta=-0.05),
    ])

    assert [f["direction"] for f in rendered] == ["D-strong", "D-weak", "A"]


def test_context_carries_findings_into_the_prompt_payload():
    ctx = build_research_context([], prior_findings=findings_for_prompt([_finding()]))
    payload = ctx.to_prompt_dict()

    assert payload["prior_findings"][0]["direction"] == "RP-001"
    assert payload["prior_findings"][0]["verdict"] == VERDICT_DONT


def test_context_without_findings_is_unchanged():
    """Existing callers pass nothing and must be unaffected."""
    assert build_research_context([]).prior_findings == ()


def test_the_shipped_default_path_is_outside_logs():
    """It must survive a reset, and a reset archives logs/ wholesale."""
    assert "logs" not in DEFAULT_FINDINGS_PATH.parts
    assert DEFAULT_FINDINGS_PATH.name == "findings.jsonl"


# ---------------------------------------------------------------------------
# Direction families and confidence: how much of a direction was actually
# tested. The gap this closes: three DIN variants used to land as three
# unrelated one-shot Don'ts, so "tested at three config points and failed all
# three" was not expressible and read identically to a single weak attempt.
# ---------------------------------------------------------------------------

from agent.research.findings import (  # noqa: E402
    CONFIDENCE_INCONCLUSIVE,
    CONFIDENCE_TESTED,
    CONFIDENCE_WELL_TESTED,
    DIRECTION_FAMILIES,
    resolve_family,
)


def _variant_record(hid, delta, iteration, hyperparams="history_length: [20,50]"):
    """A judged record carrying a HYPERPARAMETERS block, as a real handoff does."""
    record = _record(iteration=iteration, hid=hid, delta=delta)
    record.hypothesis = record.hypothesis + f"\n\nHYPERPARAMETERS:\n- {hyperparams}\n"
    return record


def test_variants_of_one_direction_roll_up_into_a_single_family_entry(tmp_path):
    ledger = FindingsLedger(tmp_path / "f.jsonl")
    for hid, delta, it, hp in (
        ("OFFLINE-DIN-SHORT-HISTORY", -0.021, 1, "history_length: [20,50]"),
        ("OFFLINE-DIN-LONG-HISTORY", -0.014, 2, "history_length: [100,200]"),
        ("OFFLINE-DIN-MEAN-POOL", -0.033, 3, "history_length: [50]"),
    ):
        ledger.record(build_finding(_variant_record(hid, delta, it, hp),
                                    direction=hid, title="candidate-conditioned history"))

    stored = ledger.load()
    assert len(stored) == 1                        # one direction, not three
    entry = stored[0]
    assert entry.direction == "DIN-SEQUENCE"
    assert entry.attempts == 3
    assert entry.confidence == CONFIDENCE_WELL_TESTED
    assert set(entry.variants) == {
        "OFFLINE-DIN-SHORT-HISTORY", "OFFLINE-DIN-LONG-HISTORY", "OFFLINE-DIN-MEAN-POOL",
    }
    # The spread of every attempt is preserved, and the headline is the best.
    assert sorted(entry.deltas) == pytest.approx([-0.033, -0.021, -0.014])
    assert entry.delta_vs_incumbent == pytest.approx(-0.014)
    # Coverage names the range actually spanned, so "we tried 20 to 200" is legible.
    assert "100,200" in entry.coverage and "20,50" in entry.coverage


def test_a_single_attempt_is_inconclusive_rather_than_a_flat_dont(tmp_path):
    """The honesty requirement: one attempt rules out one implementation, not
    the mechanism. It is still recorded, but not as a settled dead end."""
    ledger = FindingsLedger(tmp_path / "f.jsonl")
    ledger.record(build_finding(_variant_record("OFFLINE-DEEPFM", -0.02, 1),
                                direction="OFFLINE-DEEPFM", title="DeepFM tower"))

    entry = ledger.load()[0]
    assert entry.verdict == VERDICT_DONT
    assert entry.attempts == 1
    assert entry.confidence == CONFIDENCE_INCONCLUSIVE
    assert not entry.is_conclusive


def test_two_attempts_reach_tested_but_not_well_tested(tmp_path):
    ledger = FindingsLedger(tmp_path / "f.jsonl")
    for hid, delta in (("OFFLINE-HYBRID-BPR", -0.01), ("OFFLINE-GAUC-WEIGHTED-BPR", -0.02)):
        ledger.record(build_finding(_variant_record(hid, delta, 1),
                                    direction=hid, title="ranking loss"))

    entry = ledger.load()[0]
    assert entry.direction == "RANKING-LOSS"
    assert (entry.attempts, entry.confidence) == (2, CONFIDENCE_TESTED)
    assert entry.is_conclusive


def test_rerunning_the_same_variant_does_not_inflate_confidence(tmp_path):
    """Otherwise a resumed run could manufacture 'well_tested' out of one
    measurement repeated three times."""
    ledger = FindingsLedger(tmp_path / "f.jsonl")
    for iteration in (1, 2, 3):
        ledger.record(build_finding(
            _variant_record("OFFLINE-DEEPFM", -0.02, iteration),
            direction="OFFLINE-DEEPFM", title="DeepFM tower"))

    entry = ledger.load()[0]
    assert entry.attempts == 1
    assert entry.confidence == CONFIDENCE_INCONCLUSIVE


def test_one_accepted_variant_makes_the_whole_family_a_do(tmp_path):
    """A direction that demonstrably worked once is a "do", even if other
    implementations of it lost -- the mechanism is not the thing that failed."""
    ledger = FindingsLedger(tmp_path / "f.jsonl")
    ledger.record(build_finding(_variant_record("OFFLINE-WATCHTIME-AUXILIARY", -0.01, 1),
                                direction="OFFLINE-WATCHTIME-AUXILIARY", title="watch time"))
    ledger.record(build_finding(
        _record(iteration=2, hid="OFFLINE-WATCHTIME-RATIO", decision=Decision.ACCEPT, delta=0.006),
        direction="OFFLINE-WATCHTIME-RATIO", title="watch ratio"))

    entry = ledger.load()[0]
    assert entry.direction == "WATCHTIME"
    assert entry.verdict == VERDICT_DO
    assert entry.delta_vs_incumbent == pytest.approx(0.006)
    assert entry.attempts == 2


def test_an_undeclared_direction_forms_its_own_single_member_family():
    """A live-Research proposal nobody declared must still be recordable."""
    assert resolve_family("OFFLINE-SOMETHING-NEW") == "SOMETHING-NEW"
    assert resolve_family("LLM-PROPOSAL-7") == "LLM-PROPOSAL-7"


def test_pre_rollup_entries_still_load_as_single_attempt_findings(tmp_path):
    """findings.jsonl lines written before these fields existed must load
    unchanged -- as the one-attempt, inconclusive findings they always were."""
    path = tmp_path / "f.jsonl"
    path.write_text(json.dumps({
        "direction": "RP-OLD", "title": "an old entry", "verdict": VERDICT_DONT,
        "decision": "revert", "delta_vs_incumbent": -0.003,
        "validation_primary": 0.59, "why": "lost", "iteration": 4,
    }) + "\n")

    entry = FindingsLedger(path).load()[0]
    assert entry.attempts == 1
    assert entry.variants == ()
    assert entry.confidence == CONFIDENCE_INCONCLUSIVE


def test_conclusive_donts_are_ordered_ahead_of_inconclusive_ones():
    """If the prompt is truncated, the firm constraints are the ones to keep."""
    weak = Finding(direction="WEAK", title="t", verdict=VERDICT_DONT, decision="revert",
                   delta_vs_incumbent=-0.09, validation_primary=0.5, why="w", iteration=1,
                   attempts=1, confidence=CONFIDENCE_INCONCLUSIVE)
    firm = Finding(direction="FIRM", title="t", verdict=VERDICT_DONT, decision="revert",
                   delta_vs_incumbent=-0.001, validation_primary=0.6, why="w", iteration=2,
                   attempts=3, confidence=CONFIDENCE_WELL_TESTED)

    rendered = findings_for_prompt([weak, firm])

    assert [f["direction"] for f in rendered] == ["FIRM", "WEAK"]
    assert rendered[0]["confidence"] == CONFIDENCE_WELL_TESTED


def test_every_declared_family_member_resolves_back_to_its_family():
    for family, members in DIRECTION_FAMILIES.items():
        for member in members:
            assert resolve_family(f"OFFLINE-{member}") == family


# ---------------------------------------------------------------------------
# A sub-random result is a broken implementation, not a measured direction.
#
# From the real smoke run: an inverted BPR gradient (`g = sigmoid(-s)` where
# dL/ds is -sigmoid(-s)) scored 0.3937 primary / 0.3704 GAUC against a 0.6016
# incumbent. Adding the missing minus sign took the same code to 0.5864 /
# 0.6457. Without this guard the ledger closes the ranking-loss direction -- the
# highest-expected-value one in solution/ideas.md -- on a one-character bug.
# ---------------------------------------------------------------------------

from agent.research.findings import SUB_RANDOM_GAUC  # noqa: E402


def _record_with_gauc(gauc, primary=0.3937, decision=Decision.REVERT, hid="RP-BPR"):
    record = _record(hid=hid, decision=decision, primary=primary, delta=primary - 0.6016)
    record.aggregate = AggregateMetrics(primary, 0.0019, gauc, 0.4170, 2)
    return record


def test_a_sub_random_result_is_not_recorded_as_a_dont():
    """GAUC 0.3704 means positives were ordered BELOW negatives. That is an
    inverted comparison, not evidence that pairwise ranking objectives fail."""
    assert build_finding(_record_with_gauc(0.3704), direction="RP-BPR", title="BPR") is None


def test_a_sub_random_result_is_not_recorded_as_a_do_either():
    """Nothing was measured in either direction -- an accepted sub-random result
    would be even more misleading than a rejected one."""
    record = _record_with_gauc(0.42, decision=Decision.ACCEPT)
    assert build_finding(record, direction="RP-BPR", title="BPR") is None


def test_a_merely_weak_result_is_still_recorded():
    """The guard must not become a quality gate. A genuinely bad model scores
    near 0.5, not below it, and that IS a real measurement worth keeping."""
    finding = build_finding(_record_with_gauc(0.55, primary=0.52),
                            direction="RP-WEAK", title="a weak idea")

    assert finding is not None
    assert finding.verdict == VERDICT_DONT


def test_the_threshold_is_exactly_random_ordering():
    """0.5 is the score of shuffling. At or above it, the result is weak; below
    it, the ranking is anti-correlated and something is inverted."""
    assert SUB_RANDOM_GAUC == 0.5
    assert build_finding(_record_with_gauc(0.5), direction="D", title="t") is not None
    assert build_finding(_record_with_gauc(0.4999), direction="D", title="t") is None


def test_a_record_with_no_aggregate_is_unaffected_by_the_guard():
    """A crashed run has no GAUC to judge, and was already excluded for having
    no Evaluator decision."""
    record = _record(decision=None, primary=None)
    assert build_finding(record, direction="D", title="t") is None


# ---------------------------------------------------------------------------
# A run that reproduced the incumbent measured nothing.
#
# From the first full campaign: seven of sixteen iterations scored exactly
# 0.6016 across both seeds -- the baseline, to every decimal. The mechanism had
# not run (hyperparameters never reached config.json; the vendored loader has
# no watch-time or engagement columns). Every one was recorded as a "dont", and
# because a whole family shares the cause, three rolled up to "well_tested".
# ---------------------------------------------------------------------------

from agent.research.findings import NO_OP_DELTA  # noqa: E402


def test_a_run_identical_to_the_incumbent_is_not_recorded():
    record = _record(decision=Decision.REVERT, primary=0.6016151905059814, delta=0.0)
    assert build_finding(record, direction="WATCHTIME", title="watch-time aux") is None


def test_a_real_but_tiny_regression_is_still_recorded():
    """The guard must not swallow genuine near-ties -- only bit-identical ones.
    A model that really did run and lost by 1e-4 measured something."""
    record = _record(decision=Decision.REVERT, primary=0.6015, delta=-1e-4)
    finding = build_finding(record, direction="D", title="t")

    assert finding is not None
    assert finding.verdict == VERDICT_DONT


def test_the_no_op_threshold_is_tight_enough_to_mean_identical():
    """1e-9 is 'the same computation ran twice', not 'these were close'."""
    assert NO_OP_DELTA == 1e-9
    assert build_finding(_record(decision=Decision.REVERT, primary=0.6, delta=1e-10),
                         direction="D", title="t") is None
    assert build_finding(_record(decision=Decision.REVERT, primary=0.6, delta=1e-8),
                         direction="D", title="t") is not None


def test_the_six_false_findings_from_the_campaign_would_now_be_refused():
    """Replays the real deltas. WATCHTIME rolled up to well_tested on three
    runs that each reproduced the incumbent exactly."""
    for direction, delta in (("WATCHTIME", 0.0), ("WATCHTIME", 0.0), ("WATCHTIME", 0.0),
                             ("MULTITASK", 0.0), ("TIME-DRIFT", 0.0),
                             ("UNBIASED-VALIDATION", 0.0)):
        record = _record(decision=Decision.REVERT, primary=0.6016151905059814, delta=delta)
        assert build_finding(record, direction=direction, title="t") is None, direction


# ---------------------------------------------------------------------------
# Effect size: how much the metric moved, as opposed to how often it was
# measured. Both axes are needed -- three attempts agreeing on +0.00005 is
# well-measured evidence of approximately nothing, and "do / well_tested" alone
# tells Research to build on a direction nobody has shown moves the metric.
# ---------------------------------------------------------------------------

from agent.research.findings import (  # noqa: E402
    EFFECT_MARGINAL,
    EFFECT_SUBSTANTIVE,
    EFFECT_UNKNOWN,
    EFFECT_WITHIN_NOISE,
    SUBSTANTIVE_DELTA,
    effect_for,
)


def _record_with_std(delta, std, primary=0.6030, decision=Decision.ACCEPT):
    record = _record(decision=decision, primary=primary, delta=delta)
    record.aggregate = AggregateMetrics(primary, std, primary + 0.06, primary - 0.06, 2)
    return record


def test_the_effect_threshold_is_not_the_convergence_epsilon():
    """They answer different questions and must move independently: epsilon is
    the stopping rule, and lowering it makes a graded run harder to declare
    stalled. Reusing it here labelled the best real result 'marginal'."""
    from agent.config import DEFAULT_CONFIG

    assert SUBSTANTIVE_DELTA != DEFAULT_CONFIG.convergence.epsilon
    assert SUBSTANTIVE_DELTA < DEFAULT_CONFIG.convergence.epsilon


@pytest.mark.parametrize("delta,std,expected", [
    (0.00157, 0.00020, EFFECT_SUBSTANTIVE),    # the campaign's one real result
    (-0.00731, 0.00030, EFFECT_SUBSTANTIVE),   # a real regression is substantive too
    (0.00026, 0.00003, EFFECT_MARGINAL),       # above noise, below the bar
    (0.00005, 0.00028, EFFECT_WITHIN_NOISE),   # WATCHTIME: indistinguishable from zero
    (0.00012, 0.00006, EFFECT_WITHIN_NOISE),   # exactly at 2*std
    (None, None, EFFECT_UNKNOWN),
])
def test_effect_tiers_match_the_real_campaign_numbers(delta, std, expected):
    assert effect_for(delta, std) == expected


def test_noise_is_judged_against_the_run_s_own_seed_spread():
    """Not a fixed floor: the seed spread is the only estimate available of what
    this configuration does when nothing changes, so the same delta can be real
    in a quiet run and noise in a jittery one."""
    assert effect_for(0.0004, 0.00002) == EFFECT_MARGINAL        # quiet run
    assert effect_for(0.0004, 0.00050) == EFFECT_WITHIN_NOISE    # jittery run


def test_a_do_that_moved_nothing_is_flagged_as_such():
    finding = build_finding(_record_with_std(0.00005, 0.00028),
                            direction="WATCHTIME", title="watch-time aux")

    assert finding.verdict == VERDICT_DO      # it did not fail...
    assert finding.effect == EFFECT_WITHIN_NOISE
    assert finding.moved_the_metric is False  # ...but it did not work either


def test_effect_follows_the_winning_attempt_through_a_merge(tmp_path):
    """delta_vs_incumbent is the winner's, so the size claim about it has to
    describe the same run or the entry contradicts itself."""
    ledger = FindingsLedger(tmp_path / "f.jsonl")
    ledger.record(build_finding(_record_with_std(0.00004, 0.00002),
                                direction="OFFLINE-A", title="t", family="FAM"))
    ledger.record(build_finding(_record_with_std(0.00157, 0.00020),
                                direction="OFFLINE-B", title="t", family="FAM"))

    entry = ledger.load()[0]
    assert entry.delta_vs_incumbent == pytest.approx(0.00157)
    assert entry.effect == EFFECT_SUBSTANTIVE
    assert entry.attempts == 2


def test_a_do_that_moved_the_metric_leads_one_that_did_not():
    """If the prompt is read top-down, the promising direction should be the one
    that actually is."""
    real = Finding(direction="REAL", title="t", verdict=VERDICT_DO, decision="accept",
                   delta_vs_incumbent=0.00157, validation_primary=0.603, why="w",
                   iteration=2, attempts=2, confidence=CONFIDENCE_TESTED,
                   effect=EFFECT_SUBSTANTIVE)
    hollow = Finding(direction="HOLLOW", title="t", verdict=VERDICT_DO, decision="accept",
                     delta_vs_incumbent=0.00005, validation_primary=0.603, why="w",
                     iteration=5, attempts=3, confidence=CONFIDENCE_WELL_TESTED,
                     effect=EFFECT_WITHIN_NOISE)

    rendered = findings_for_prompt([hollow, real])

    assert [f["direction"] for f in rendered] == ["REAL", "HOLLOW"]


def test_pre_effect_entries_still_load():
    path = Path(__file__).parent / "_tmp_effect.jsonl"
    path.write_text(json.dumps({
        "direction": "OLD", "title": "t", "verdict": VERDICT_DONT, "decision": "revert",
        "delta_vs_incumbent": -0.003, "validation_primary": 0.59, "why": "w", "iteration": 1,
    }) + "\n")
    try:
        entry = FindingsLedger(path).load()[0]
        assert entry.effect == EFFECT_UNKNOWN
        assert entry.moved_the_metric is False
    finally:
        path.unlink()
