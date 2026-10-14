# CADBUSI-Database

Breast ultrasound data pipeline: BigQuery pull (`--query`) -> DICOM download (Cloud Run) ->
SQLite database build (`--database`) -> exports. See README.md for usage.

## IRB scope (hard rule)

- IRB protocol **#21-003545** covers records dated **2018-01-01 to 2024-12-31 inclusive**. Nothing
  outside that window may be queried, downloaded, or written into a database.
- 2025+ exams were pulled by mistake once, which led to a reportable deviation. An amendment to widen
  the window is pending, not approved. Do not act as though it were approved.
- `src/irb.py` is the single source of truth. Use `bq_date_filter()` in any new BigQuery query and
  `assert_in_irb_window()` on any new pull. Never hardcode the dates anywhere else.
- Do not widen data scope (dates, cohorts, new data elements or modalities) on your own initiative.
  Ask the user first, since this goes through the PI and the amendment.
- Existing databases built before the filter was added (e.g. `data/cadbusi_2026_7_16.db`) may contain
  out-of-window exams. Filter them by exam date before using them for new work, and say that you did.
