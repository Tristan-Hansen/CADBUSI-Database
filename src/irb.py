"""IRB scope for every data pull in this repo. Single source of truth.

Protocol #21-003545 approves review of records dated 2018-01-01 through
2024-12-31 (inclusive). Exams dated 2025+ were pulled by mistake in an earlier
build, which led to a reportable deviation; an amendment to widen the window
is in flight but NOT approved. Until it is, nothing outside this window may be
queried, downloaded, or written into a database.

Change these dates only after the amendment is approved, and update README.md
and CLAUDE.md in the same commit.
"""
import pandas as pd

IRB_PROTOCOL = "21-003545"
IRB_START_DATE = "2018-01-01"
IRB_END_DATE = "2024-12-31"


def bq_date_filter(column):
    """BigQuery predicate keeping `column` (DATETIME/TIMESTAMP) inside the IRB window."""
    return f"DATE({column}) BETWEEN DATE('{IRB_START_DATE}') AND DATE('{IRB_END_DATE}')"


def assert_in_irb_window(dates, what):
    """Hard-fail if any date in `dates` is missing or outside the IRB window."""
    d = pd.to_datetime(pd.Series(dates), errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    start, end = pd.Timestamp(IRB_START_DATE), pd.Timestamp(IRB_END_DATE) + pd.Timedelta(days=1)
    bad = d.isna() | (d < start) | (d >= end)
    if bad.any():
        raise RuntimeError(
            f"IRB #{IRB_PROTOCOL}: {int(bad.sum())} {what} dated outside "
            f"{IRB_START_DATE}..{IRB_END_DATE} (or undated). Refusing to continue."
        )
