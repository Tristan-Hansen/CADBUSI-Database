"""Minimal Vertex AI CustomJob + Cloud Logging client over REST with Python ADC.

Ported from BUS_framework (experiments/jepa_stage2/vertex_api.py). The gcloud CLI's user
credentials expire daily here; ADC (google.auth.default) is used first and the CLI is only a
fallback. Functions: create_job(display_name, job_spec, labels) -> job id, get_job(id) -> dict,
list_jobs(filter) -> [dict], cancel_job(id), job_logs(id, since_minutes) -> [text].

  python -m src.cloud.vertex_api list
  python -m src.cloud.vertex_api get <job_id>
  python -m src.cloud.vertex_api logs <job_id> [minutes]
  python -m src.cloud.vertex_api cancel <job_id>
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from config import CONFIG

PROJECT, REGION = CONFIG["env"]["project_id"], CONFIG["env"]["region"]
API = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/customJobs"
GCLOUD = shutil.which("gcloud") or "gcloud"
_session = None


def session():
    global _session
    if _session is None:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        _session = AuthorizedSession(creds)
    return _session


def _cli(args):
    r = subprocess.run([GCLOUD, *args, f"--project={PROJECT}", f"--region={REGION}", "--format=json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    return json.loads(r.stdout) if r.stdout.strip() else {}


def create_job(display_name: str, job_spec: dict, labels: dict) -> str:
    body = {"displayName": display_name, "labels": labels, "jobSpec": job_spec}
    try:
        r = session().post(API, json=body, timeout=60)
        if r.status_code >= 400:
            print(f"  REST create HTTP {r.status_code}: {r.text[:600]}", flush=True)
        r.raise_for_status()
        return r.json()["name"].rsplit("/", 1)[1]
    except Exception as e:  # noqa: BLE001
        print(f"  REST create failed ({e}); trying gcloud CLI", flush=True)
        import yaml
        tmp = Path(tempfile.gettempdir()) / f"{display_name}.yaml"
        tmp.write_text(yaml.safe_dump(job_spec))
        lab = ",".join(f"{k}={v}" for k, v in labels.items())
        d = _cli(["ai", "custom-jobs", "create", f"--display-name={display_name}", f"--config={tmp}",
                  f"--labels={lab}"])
        return d["name"].rsplit("/", 1)[1]


def get_job(job_id: str) -> dict:
    try:
        r = session().get(f"{API}/{job_id}", timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return _cli(["ai", "custom-jobs", "describe", job_id])


def list_jobs(filter: str = "", page_size: int = 20) -> list[dict]:
    """Most recent jobs first. `filter` uses Vertex list syntax, e.g. 'labels.pipeline="cadbusi-database"'."""
    params = {"pageSize": page_size}
    if filter:
        params["filter"] = filter
    r = session().get(API, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("customJobs", [])


def cancel_job(job_id: str):
    try:
        r = session().post(f"{API}/{job_id}:cancel", json={}, timeout=60)
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        subprocess.run([GCLOUD, "ai", "custom-jobs", "cancel", job_id, f"--project={PROJECT}",
                        f"--region={REGION}"], check=True)


def job_logs(job_id: str, since_minutes: int = 10, limit: int = 300) -> list[str]:
    from datetime import datetime, timedelta, timezone
    t = (datetime.now(timezone.utc) - timedelta(minutes=since_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {"resourceNames": [f"projects/{PROJECT}"], "orderBy": "timestamp asc", "pageSize": limit,
            "filter": f'resource.type="ml_job" AND resource.labels.job_id="{job_id}" AND timestamp>="{t}"'}
    r = session().post("https://logging.googleapis.com/v2/entries:list", json=body, timeout=60)
    r.raise_for_status()
    return [e.get("textPayload") or json.dumps(e.get("jsonPayload", "")) for e in r.json().get("entries", [])]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # job logs contain box-drawing chars
    if sys.argv[1] == "list":
        for d in list_jobs():
            print(d["name"].rsplit("/", 1)[1], d.get("state"), d.get("displayName"), d.get("createTime"))
    elif sys.argv[1] == "get":
        d = get_job(sys.argv[2])
        print(d.get("displayName"), d.get("state"), d.get("startTime"), d.get("endTime"), d.get("error", ""))
    elif sys.argv[1] == "logs":
        print("\n".join(job_logs(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)))
    elif sys.argv[1] == "cancel":
        cancel_job(sys.argv[2]); print("cancel requested")
