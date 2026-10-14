"""Run `main.py --birad_parsing` (CPU) and `main.py --database` (GPU) as Vertex jobs on a run folder
that already holds query output. DRY-RUN BY DEFAULT.

Run folder layout, all under gs://shared-aif-bucket-87d1/Databases/runs/<run_id>/:
  query_data/        from launch_query; the birads stage adds birads.csv here
  gemini/            Gemini batch input/output for the birads stage
  database_test/     cadbusi.db, images/, videos/, inpainted/ from the database stage, + _logs/
  _job/              code snapshot (+ dicom_list.txt for the database stage)
Both stages run under the write guard (only the run folder is writable). The database stage reads
DICOMs from CONFIG["storage"]["download_path"], read-only, restricted by default to the exams that
appear in query_data/endpoint_data.csv, and the encryption key from gs://.../Keys/ (it refuses to
run without it, so a new key can never be generated silently).

  python -m src.cloud.launch_database --run-id query_test_20261013 --stage birads [--submit]
  python -m src.cloud.launch_database --run-id query_test_20261013 --stage database [--submit]
  python -m src.cloud.launch_database --gpu-wheelhouse <dir>    # stage the GPU wheelhouse once
"""
from __future__ import annotations

import argparse
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import CONFIG
from src.cloud import jobs
from src.cloud.launch_query import BUCKET, RUNS, WHEELHOUSE, bucket, code_tarball, stage_wheelhouse

GPU_WHEELHOUSE = f"{RUNS}/_shared/wheelhouse_database_py310_torch23.tar.gz"
CPU_MACHINE, GPU_MACHINE = "n2-standard-8", "n1-standard-8-t4"  # README: a T4 is enough

COMMON = r"""set -eo pipefail
case "${RUN_PREFIX}" in Databases/runs/?*) ;; *) echo "refusing RUN_PREFIX=${RUN_PREFIX}"; exit 2;; esac
mkdir -p /workdir && cd /workdir
exec > >(tee -a /workdir/_job.log) 2>&1
echo "== $(date -u) ${STAGE} for ${RUN_PREFIX} on $(hostname), $(nproc) cpus"; python3 --version
gsutil -q cp "gs://${BUCKET}/${RUN_PREFIX}/_job/code.tar.gz" . && mkdir -p repo && tar xzf code.tar.gz -C repo
gsutil -q cp "gs://${BUCKET}/${WHEELHOUSE}" wh.tar.gz && tar xzf wh.tar.gz
"""

BIRADS_BOOT = COMMON + r"""
pip install -q --no-index --find-links wheelhouse -r query_requirements.in 2>&1 | tail -40
cd repo && mkdir -p data
gsutil -q cp "gs://${BUCKET}/${RUN_PREFIX}/query_data/endpoint_data.csv" data/
export CADBUSI_WRITE_PREFIX="${RUN_PREFIX}/"
status=0
python3 main.py --birad_parsing || status=$?
echo "== birads exit status ${status}"
cp /workdir/_job.log data/_birads_job.log
[ -f data/birads.csv ] && gsutil -q cp data/birads.csv "gs://${BUCKET}/${RUN_PREFIX}/query_data/"
gsutil -q cp data/_birads_job.log "gs://${BUCKET}/${RUN_PREFIX}/query_data/" || true
exit ${status}
"""

DATABASE_BOOT = COMMON + r"""
nvidia-smi -L
python3 -c "import torch, torchvision; print('image torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available())"
pip install -q --no-index --find-links wheelhouse -r gpu_requirements.in -c constraints.txt 2>&1 | tail -40
cd repo && mkdir -p data
for f in endpoint_data.csv lesion_pathology.csv birads.csv; do
  gsutil -q cp "gs://${BUCKET}/${RUN_PREFIX}/query_data/${f}" data/
done
# The key must be the shared one: encrypt_ids() silently creates a NEW key when the file is missing.
gsutil -q cp "gs://${BUCKET}/Keys/encryption_key.pkl" encryption_key.pkl
[ -s encryption_key.pkl ] || { echo "encryption key missing, refusing to run"; exit 3; }
gsutil -q cp "gs://${BUCKET}/${RUN_PREFIX}/_job/dicom_list.txt" /workdir/dicom_list.txt
# EasyOCR builds its reader at import time (image_processing imports tools.reader), before
# download_models() runs, and would try GitHub, which jobs cannot reach. Stage its weights first.
mkdir -p ~/.EasyOCR/model
gsutil -q cp "gs://${BUCKET}/${WEIGHTS_PREFIX}/easyocr_model/*.pth" ~/.EasyOCR/model/
export CADBUSI_WRITE_PREFIX="${RUN_PREFIX}/"
export CADBUSI_DATABASE_DIR="${RUN_PREFIX}/database_test/"
export CADBUSI_DEBUG_DATA_RANGE=""
export CADBUSI_DICOM_LIST=/workdir/dicom_list.txt
status=0
python3 main.py --database || status=$?
echo "== database exit status ${status}"
cp /workdir/_job.log data/_job.log
gsutil -q -m cp data/*.json data/*.csv data/_job.log "gs://${BUCKET}/${RUN_PREFIX}/database_test/_logs/" || true
[ "${status}" = 0 ] && echo DONE | gsutil -q cp - "gs://${BUCKET}/${RUN_PREFIX}/database_test/_DONE"
exit ${status}
"""


def dicom_list(prefix: str) -> list[str]:
    """DICOM paths in the download folder for the exams present in this run's endpoint_data.csv."""
    import pandas as pd
    b = bucket()
    df = pd.read_csv(io.BytesIO(b.blob(f"{prefix}/query_data/endpoint_data.csv").download_as_bytes()),
                     dtype={"PATIENT_ID": str, "ACCESSION_NUMBER": str})
    pairs = sorted(set(zip(df.PATIENT_ID.str.strip(), df.ACCESSION_NUMBER.str.strip())))
    dl = CONFIG["storage"]["download_path"].strip("/")

    def files(p):
        return [x.name for x in b.list_blobs(prefix=f"{dl}/{p[0]}_{p[1]}/") if x.name.endswith(".dcm")]
    with ThreadPoolExecutor(16) as ex:
        out = [f for fs in ex.map(files, pairs) for f in fs]
    print(f"{len(pairs)} exams in endpoint_data.csv -> {len(out)} DICOMs in {dl}/")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=False)
    ap.add_argument("--stage", choices=["birads", "database"])
    ap.add_argument("--timeout-hours", type=float)
    ap.add_argument("--gpu-wheelhouse", type=Path, help="stage this GPU wheelhouse dir and exit")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    if a.gpu_wheelhouse:
        return stage_wheelhouse(a.gpu_wheelhouse, GPU_WHEELHOUSE, ("gpu_requirements.in", "constraints.txt"))
    if not (a.run_id and a.stage):
        ap.error("--run-id and --stage are required")

    prefix = f"{RUNS}/{a.run_id}"
    sa = CONFIG["env"]["service_account_identity"]
    if a.stage == "birads":
        timeout = a.timeout_hours or 3.0
        machine = CPU_MACHINE
        env = {"BUCKET": BUCKET, "RUN_PREFIX": prefix, "WHEELHOUSE": WHEELHOUSE, "STAGE": "birads"}
        sp = jobs.spec(env, BIRADS_BOOT, machine, timeout, disk_gb=100, service_account=sa,
                       image=jobs.PY310_CPU_IMAGE)
    else:
        timeout = a.timeout_hours or 3.0
        machine = GPU_MACHINE
        env = {"BUCKET": BUCKET, "RUN_PREFIX": prefix, "WHEELHOUSE": GPU_WHEELHOUSE, "STAGE": "database",
               "WEIGHTS_PREFIX": CONFIG["weights"]["gcs_prefix"].strip("/")}
        sp = jobs.spec(env, DATABASE_BOOT, machine, timeout, disk_gb=200, service_account=sa)
    print(f"run folder: gs://{BUCKET}/{prefix}/")
    if a.submit:
        b = bucket()
        code = code_tarball()
        b.blob(f"{prefix}/_job/code.tar.gz").upload_from_string(code, timeout=600)
        print(f"staged code snapshot ({len(code) / 1e6:.1f} MB)")
        if a.stage == "database":
            missing = [f for f in ("endpoint_data.csv", "lesion_pathology.csv", "birads.csv")
                       if not b.blob(f"{prefix}/query_data/{f}").exists()]
            if missing:
                raise SystemExit(f"query_data/ is missing {missing}; run the query/birads stages first")
            b.blob(f"{prefix}/_job/dicom_list.txt").upload_from_string("\n".join(dicom_list(prefix)) + "\n")
    jobs.submit(f"cadbusi-{a.stage}-{a.run_id}".replace("_", "-"), sp, machine, timeout, go=a.submit,
                labels={"stage": a.stage}, note=f"--{a.stage} on {prefix}")


if __name__ == "__main__":
    main()
