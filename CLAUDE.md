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

## Vertex jobs (`src/cloud/`)

Ported from BUS_framework. Not wired to any pipeline stage yet.

- `vertex_api.py`: REST client using ADC, with gcloud as a fallback. Run as
  `python -m src.cloud.vertex_api list|get|logs|cancel`.
- `jobs.py`: `spec()` builds a SPOT CustomJob running a bash boot script on stock Google images.
  `submit()` is a **dry run unless `go=True`**.
- `ledger.py`: spend ledger in `cloud_runs/gcp_spend.csv` with a hard budget gate. The cap defaults
  to $100 and can be overridden with `CADBUSI_GCP_CAP_USD`. Every submit passes the gate and is recorded.

Rules:
- Never submit a real job without the user's explicit OK for that job. Show the dry run first,
  including the worst-case cost.
- Spot only, always with a timeout. Run a canary (small slice) before any full run.
- Jobs write only under their own GCS prefix and must respect the IRB window (`src/irb.py`).
- Monitor with `vertex_api get/logs` and `python -m src.cloud.ledger`. Report failures with log excerpts.
