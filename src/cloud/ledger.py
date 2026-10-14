"""GCP spend ledger + hard budget gate for Vertex jobs. Ported from BUS_framework
(experiments/jepa_stage2/gcp_ledger.py), trimmed to a single ledger.

cloud_runs/gcp_spend.csv (gitignored, per machine), one row per job. `update()` refreshes
state/hours/$ from Vertex. `committed()` = $ of finished jobs + WORST CASE (timeout x rate) of
every unfinished job; `gate()` refuses a new job unless committed + its worst case < CAP_USD.

Rates are Vertex Custom Training SPOT incl. management fee, Mayo -20%, taken from
BUS_framework docs/gcp/PRICING.md. Add a machine here only with a sourced rate.

  python -m src.cloud.ledger            # update + print
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.cloud import vertex_api as V

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "cloud_runs" / "gcp_spend.csv"
CAP_USD = float(os.environ.get("CADBUSI_GCP_CAP_USD", "100"))
RATES = {"a2-highgpu-1g": (1.795, "BUS_framework PRICING.md Mayo SPOT a2-highgpu-1g (1x A100 40GB)"),
         "n2-standard-8": (0.45, "conservative upper bound; n2-standard-8 not in PRICING.md"),
         "n1-standard-8-t4": (0.85, "conservative upper bound (~Vertex on-demand list n1-standard-8 + 1x T4); "
                                    "not in PRICING.md, spot is lower")}
FIELDS = ["job_name", "job_id", "machine", "spot_rate_usd_hr", "rate_source", "timeout_h", "worst_case_usd",
          "submitted_utc", "start_utc", "end_utc", "state", "hours", "cost_usd", "note"]
FINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


class _Lock:
    """Cross-process lock around every ledger read-modify-write."""
    def __init__(self):
        self.path = LEDGER.with_suffix(".lock")

    def __enter__(self):
        import time
        LEDGER.parent.mkdir(exist_ok=True)
        for _ in range(600):
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:  # break a stale lock (> 5 min old)
                    if time.time() - self.path.stat().st_mtime > 300:
                        self.path.unlink()
                except FileNotFoundError:
                    pass
                time.sleep(0.5)
        raise RuntimeError(f"ledger lock timeout: {self.path}")

    def __exit__(self, *a):
        os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def read() -> list[dict]:
    if not LEDGER.exists():
        return []
    with open(LEDGER, newline="") as f:
        return list(csv.DictReader(f))


def write(rows):
    LEDGER.parent.mkdir(exist_ok=True)
    with open(LEDGER, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def billed_hours(job_id: str, gap_min: float = 30.0) -> float | None:
    """Billable hours from CONTAINER log spans (entries exist only while a VM runs the container).
    Attempts are split at gaps > gap_min (preemption / re-provisioning). None if logs can't be read."""
    try:
        t = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ts, tok = [], None
        while True:
            body = {"resourceNames": [f"projects/{V.PROJECT}"], "orderBy": "timestamp asc", "pageSize": 1000,
                    "filter": (f'resource.type="ml_job" AND resource.labels.job_id="{job_id}" AND '
                               f'logName:"workerpool0-0" AND timestamp>="{t}"')}
            if tok:
                body["pageToken"] = tok
            r = V.session().post("https://logging.googleapis.com/v2/entries:list", json=body, timeout=60).json()
            ts += [_ts(e["timestamp"][:26].rstrip("Z") + "+00:00") for e in r.get("entries", [])]
            tok = r.get("nextPageToken")
            if not tok:
                break
        if not ts:
            return 0.0
        ts.sort()
        hrs, start, prev = 0.0, ts[0], ts[0]
        for x in ts[1:]:
            if (x - prev).total_seconds() > gap_min * 60:
                hrs += (prev - start).total_seconds() / 3600
                start = x
            prev = x
        hrs += (prev - start).total_seconds() / 3600
        return hrs + 0.15 * (1 + sum(1 for a, b in zip(ts, ts[1:]) if (b - a).total_seconds() > gap_min * 60))
    except Exception as e:  # noqa: BLE001
        print(f"  billed_hours({job_id}) failed: {e}")
        return None


def update() -> list[dict]:
    with _Lock():
        rows = read()
        now = datetime.now(timezone.utc)
        for r in rows:
            if r["state"] in FINAL or not r["job_id"]:
                continue
            try:
                d = V.get_job(r["job_id"])
            except Exception as e:  # noqa: BLE001
                print(f"  describe {r['job_id']} failed: {e}")
                continue
            r["state"] = d.get("state", r["state"])
            st, en = _ts(d.get("startTime")), _ts(d.get("endTime"))
            r["start_utc"] = st.isoformat() if st else ""
            r["end_utc"] = en.isoformat() if en else ""
            hrs = ((en or now) - st).total_seconds() / 3600 if st else 0.0
            if r["state"] in FINAL:  # bill from container logs, never below the job's own start->end span
                bh = billed_hours(r["job_id"])
                if bh is not None:
                    hrs = max(bh, hrs)
            r["hours"] = f"{hrs:.3f}"
            r["cost_usd"] = f"{hrs * float(r['spot_rate_usd_hr']):.2f}"
        write(rows)
        return rows


def committed(rows=None) -> float:
    rows = read() if rows is None else rows
    return sum(float(r["cost_usd"] or 0) if r["state"] in FINAL else
               max(float(r["worst_case_usd"]), float(r["cost_usd"] or 0)) for r in rows)


def spent(rows=None) -> float:
    rows = read() if rows is None else rows
    return sum(float(r["cost_usd"] or 0) for r in rows)


def gate(machine: str, timeout_h: float, n_jobs: int = 1) -> float:
    """Raise SystemExit unless committed + worst case of the new job(s) stays under the cap."""
    rows = update()
    worst = RATES[machine][0] * timeout_h * n_jobs
    c = committed(rows)
    if c + worst >= CAP_USD:
        raise SystemExit(f"BUDGET GATE: committed ${c:.2f} + worst case ${worst:.2f} >= ${CAP_USD:.0f} -- not submitting")
    print(f"budget gate OK: committed ${c:.2f} + worst ${worst:.2f} = ${c + worst:.2f} < ${CAP_USD:.0f}")
    return worst


def record(job_name, job_id, machine, timeout_h, note=""):
    with _Lock():
        rows = read()
        rate, src = RATES[machine]
        rows.append({"job_name": job_name, "job_id": job_id, "machine": machine, "spot_rate_usd_hr": rate,
                     "rate_source": src, "timeout_h": timeout_h, "worst_case_usd": f"{rate * timeout_h:.2f}",
                     "submitted_utc": datetime.now(timezone.utc).isoformat(), "start_utc": "", "end_utc": "",
                     "state": "SUBMITTED", "hours": "0", "cost_usd": "0", "note": note})
        write(rows)


if __name__ == "__main__":
    rows = update()
    for r in rows:
        print(f"{r['job_name']:<34} {r['state']:<22} {r['hours']:>7}h ${r['cost_usd']:>6}  (worst ${r['worst_case_usd']})")
    print(f"SPENT ${spent(rows):.2f} | COMMITTED (incl. worst case of unfinished) ${committed(rows):.2f} | CAP ${CAP_USD:.0f}")
