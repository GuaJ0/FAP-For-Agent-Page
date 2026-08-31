"""Extra KuaiRand-Pure columns, on request. NOT part of the vendored starter kit.

WHY THIS IS A SEPARATE FILE
---------------------------
`data.py` and `evaluate.py` are verbatim copies of the official starter kit and
must stay byte-identical -- tests/test_vendored_harness.py pins their hashes,
because `evaluate.py` defines what every score in this repo means and `data.py`
defines the official split. This module adds to them from the outside instead
of editing them, so that guard keeps working and there is never a question
about which lines are the competition's and which are ours.

WHY IT EXISTS
-------------
`data.load()` returns 7 fields: date, user_id, video_id, author_id, tab,
duration_ms, long_view. The log actually holds 19 columns, and the missing ones
are exactly what several research directions need -- watch-time modelling needs
`play_time_ms`, multi-task needs `is_click`/`is_like`/..., drift work needs
`hourmin`/`time_ms`.

The first exploration campaign could not test any of them. Those solutions
looked for `play_time_ms` through `data.load()`, found nothing, silently
skipped their auxiliary head and reported the unchanged baseline's score --
which the ledger then recorded as the direction having failed. Three watch-time
variants rolled up to a "well_tested" dead end on evidence that no watch-time
model had ever run.

WHY NOT JUST READ THE CSVs
--------------------------
`SPLITS` in data.py is the single definition of which dates are train, valid
and held-out. A solution that re-derives that boundary itself can quietly
include validation rows in training: it then scores brilliantly on validation,
wins ACCEPT, becomes the incumbent, and collapses on the held-out split.
Nothing downstream catches that -- `agent/verification.py` proves the reported
metrics match the predictions, never which rows produced them. So this reuses
data.py's own SPLITS rather than restating it.
"""
import collections
import csv
import os

from data import LABEL, SPLITS

# The seven fields data.load() has always returned, in their original
# positions. encode() and every generated train.py index these positionally
# (x[5] is duration_ms, x[6] is the label), so this order is a compatibility
# contract: extras are appended after it, never inserted into it.
BASE_FIELDS = ('date', 'user_id', 'video_id', 'author_id', 'tab', 'duration_ms', LABEL)

# Everything else in the log, available on request. Parsing all of them costs
# roughly +380MB and +2.7s over data.load(), and the baseline needs none of
# them -- so a solution asks for exactly what its hypothesis requires.
EXTRA_COLUMNS = {
    'hourmin': int,               # time of day as HHMM, e.g. 1900
    'time_ms': int,               # epoch milliseconds of the impression
    'is_click': int,
    'is_like': int,
    'is_follow': int,
    'is_comment': int,
    'is_forward': int,
    'is_hate': int,
    'play_time_ms': float,        # watch time; censored by duration_ms
    'profile_stay_time': float,
    'comment_stay_time': float,
    'is_profile_enter': int,
    'is_rand': int,
}

_LOGS = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')


def _parse(raw, cast):
    """Empty and malformed cells become 0 rather than raising.

    One unparseable cell in 1.14M rows must not take down a training run, and
    these are all count/duration columns where 0 is the honest reading of
    "nothing recorded".
    """
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return cast(0)


def load_full(data_dir, columns=()):
    """data.load(), plus the named EXTRA_COLUMNS, as attribute-accessible rows.

        from dataset import load_full
        splits = load_full(data_dir, columns=('play_time_ms', 'is_click'))
        splits['train'][0].play_time_ms

    Rows are namedtuples, so they remain tuples: x[5] is still duration_ms,
    x[6] is still the label, and `data.encode()` consumes the result unchanged.
    Extras follow the base fields in the order requested.

    Splits come from data.SPLITS, so this cannot disagree with data.load()
    about where the held-out boundary is.
    """
    columns = tuple(columns)
    unknown = [c for c in columns if c not in EXTRA_COLUMNS]
    if unknown:
        raise ValueError(
            f"unknown column(s) {unknown}; available: {sorted(EXTRA_COLUMNS)}"
        )

    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    Row = collections.namedtuple('Row', BASE_FIELDS + columns)
    casts = [(c, EXTRA_COLUMNS[c]) for c in columns]

    rows = []
    for f in _LOGS:
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                row = (int(r['date']), r['user_id'], r['video_id'],
                       vid2author.get(r['video_id'], 'UNK'), r['tab'],
                       float(r['duration_ms']), 1 if r[LABEL] != '0' else 0)
                if casts:
                    row += tuple(_parse(r.get(c), cast) for c, cast in casts)
                # Built here rather than converted afterwards: a post-hoc
                # [Row(*x) for x in rows] holds both representations of 1.4M
                # rows at once and cost ~265MB of peak for nothing.
                rows.append(Row._make(row))

    return {name: [x for x in rows if lo <= x[0] <= hi]
            for name, (lo, hi) in SPLITS.items()}
