"""Run `main.py --query` as a Vertex job. DRY-RUN BY DEFAULT.

Everything the job writes goes under gs://shared-aif-bucket-87d1/Databases/runs/<run_id>/:
  _job/code.tar.gz      repo snapshot (tracked + untracked-but-not-ignored files, minus data/; no keys)
  query_data/           the query's CSVs + audit.json, copied out when the query finishes
  query_data/_job.log   full stdout/stderr of the run
The Python side runs under the write guard (CADBUSI_WRITE_PREFIX), and the boot script refuses
any destination outside Databases/runs/. Dependencies come from a prebuilt cp310 wheelhouse in
Databases/runs/_shared/ because job workers may not reach PyPI.

  python -m src.cloud.launch_query --wheelhouse <dir>   # stage the wheelhouse once
  python -m src.cloud.launch_query --limit 100          # dry run
  python -m src.cloud.launch_query --limit 100 --submit
"""
from __future__ import annotations

import argparse
import io
import subprocess
import tarfile
import time
from pathlib import Path

from config import CONFIG
from src.cloud import jobs

REPO = Path(__file__).resolve().parents[2]
BUCKET = CONFIG["BUCKET"]
RUNS = "Databases/runs"
WHEELHOUSE = f"{RUNS}/_shared/wheelhouse_query_py310.tar.gz"
MACHINE = "n2-standard-8"

BOOT = r"""set -eo pipefail
case "${RUN_PREFIX}" in Databases/runs/?*) ;; *) echo "refusing RUN_PREFIX=${RUN_PREFIX}"; exit 2;; esac
mkdir -p /workdir && cd /workdir
exec > >(tee -a /workdir/_job.log) 2>&1
echo "== $(date -u) run ${RUN_PREFIX} on $(hostname), $(nproc) cpus"; python3 --version
echo "== network reachability (informational)"
for u in https://pypi.org/simple/ https://huggingface.co https://github.com https://bigquery.googleapis.com; do
  printf '%s -> ' "$u"; curl -s -m 8 -o /dev/null -w '%{http_code}\n' "$u" || echo unreachable
done
gsutil -q cp "gs://${BUCKET}/${RUN_PREFIX}/_job/code.tar.gz" . && mkdir -p repo && tar xzf code.tar.gz -C repo
gsutil -q cp "gs://${BUCKET}/${WHEELHOUSE}" wh.tar.gz && tar xzf wh.tar.gz
pip install -q --no-index --find-links wheelhouse -r query_requirements.in 2>&1 | tail -40
cd repo
export CADBUSI_WRITE_PREFIX="${RUN_PREFIX}/"
status=0
python3 main.py --query --limit "${LIMIT}" || status=$?
echo "== query exit status ${status}"
ls -la data/ || true
cp /workdir/_job.log data/_job.log 2>/dev/null || true
gsutil -q -m cp data/* "gs://${BUCKET}/${RUN_PREFIX}/query_data/" || true
[ "${status}" = 0 ] && echo DONE | gsutil -q cp - "gs://${BUCKET}/${RUN_PREFIX}/query_data/_DONE"
exit ${status}
"""


def code_tarball() -> bytes:
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=REPO,
                           capture_output=True, check=True).stdout.decode().split("\0")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in filter(None, files):
            p = REPO / f
            if p.is_file() and not f.startswith("data/") and p.stat().st_size < 20_000_000:
                tar.add(p, arcname=f)
    return buf.getvalue()


def bucket():
    from google.cloud import storage
    return storage.Client().bucket(BUCKET)


def stage_wheelhouse(src: Path, dest: str = WHEELHOUSE, extra=("query_requirements.in",)):
    """Tar `src` (holding wheelhouse/ plus the `extra` requirement files) and upload it once."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src / "wheelhouse", arcname="wheelhouse")
        for name in extra:
            tar.add(src / name, arcname=name)
    bucket().blob(dest).upload_from_string(buf.getvalue(), timeout=1800)
    print(f"staged gs://{BUCKET}/{dest} ({len(buf.getvalue()) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="patients passed to main.py --query --limit")
    ap.add_argument("--run-id", default=time.strftime("query_test_%Y%m%d_%H%M"))
    ap.add_argument("--timeout-hours", type=float, default=2.0)
    ap.add_argument("--wheelhouse", type=Path, help="stage this wheelhouse dir and exit")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    if a.wheelhouse:
        return stage_wheelhouse(a.wheelhouse)

    prefix = f"{RUNS}/{a.run_id}"
    env = {"BUCKET": BUCKET, "RUN_PREFIX": prefix, "WHEELHOUSE": WHEELHOUSE, "LIMIT": a.limit}
    sp = jobs.spec(env, BOOT, MACHINE, a.timeout_hours, disk_gb=100,
                   service_account=CONFIG["env"]["service_account_identity"], image=jobs.PY310_CPU_IMAGE)
    print(f"run folder: gs://{BUCKET}/{prefix}/  (query_data/ written there, nothing else)")
    if a.submit:
        code = code_tarball()
        bucket().blob(f"{prefix}/_job/code.tar.gz").upload_from_string(code, timeout=600)
        print(f"staged code snapshot ({len(code) / 1e6:.1f} MB)")
    jobs.submit(f"cadbusi-{a.run_id}".replace("_", "-"), sp, MACHINE, a.timeout_hours, go=a.submit,
                labels={"stage": "query"}, note=f"--query --limit {a.limit} -> {prefix}/query_data")


if __name__ == "__main__":
    main()
