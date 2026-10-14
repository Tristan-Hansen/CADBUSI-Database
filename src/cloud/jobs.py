"""Build and submit Vertex CustomJobs for this repo. DRY-RUN BY DEFAULT, SPOT ONLY.

Pattern from BUS_framework (experiments/frozen_probe_grid/crop_rebuild/submit_crop_rebuild.py):
stock Google images (no custom Docker build), code + deps staged to GCS and copied in by a bash
boot script, every submit passes the budget gate and is recorded in the ledger.

No pipeline stage is wired up yet; a stage's launcher builds a boot script and calls
submit(name, spec(...), machine=..., timeout_h=..., go=args.submit).
"""
from __future__ import annotations

import shlex

from src.cloud import ledger
from src.cloud import vertex_api as V

BUCKET = "shared-aif-bucket-87d1"
CPU_IMAGE = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cpu:latest"
GPU_IMAGE = ("us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310@sha256:"
             "2f104d4ea2e5bd2f1921178b7ab5562cb18d04429552e64113be927b03c4a8ee")  # pinned, as in BUS_framework
# Python 3.10 on CPU-only machines (Vertex rejects the GPU image without an accelerator); proven by
# BUS_framework's schema-probe jobs on n1-standard-4. Use with cp310 wheelhouses.
PY310_CPU_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-3.py310:latest"
# Machine profiles with a GPU: name -> (Vertex machineType, accelerator, count). A2 types come with
# their A100 built in; N1 types take an attachable GPU, so the profile name carries the GPU.
GPU_ACCELERATOR = {"a2-highgpu-1g": ("a2-highgpu-1g", "NVIDIA_TESLA_A100", 1),
                   "n1-standard-8-t4": ("n1-standard-8", "NVIDIA_TESLA_T4", 1)}
LABELS = {"pipeline": "cadbusi-database"}


def spec(env: dict, boot: str, machine: str, timeout_h: float, disk_gb: int = 200,
         service_account: str | None = None, image: str | None = None) -> dict:
    """CustomJob spec: one SPOT worker running `boot` under bash with `env` exported first.

    `service_account`: the default Vertex identity gets 403 on the PHI BigQuery datasets; jobs that
    query them must run as the Workbench SA (CONFIG["env"]["service_account_identity"]).
    `image`: override the default image (PY310_CPU_IMAGE for prebuilt cp310 wheelhouses on CPU machines)."""
    # Inside Vertex, client libraries default to the job's tenant project ("...-tp"), which has no
    # BigQuery rights; pin our project so bigquery.Client() / storage.Client() bill and run there.
    env = {"GOOGLE_CLOUD_PROJECT": V.PROJECT, **env}
    exports = "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in env.items() if v is not None)
    gpu = machine in GPU_ACCELERATOR
    ms = {"machineType": machine}
    if gpu:
        ms["machineType"], ms["acceleratorType"], ms["acceleratorCount"] = GPU_ACCELERATOR[machine]
    out = {"workerPoolSpecs": [{"machineSpec": ms, "replicaCount": 1,
                                 "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": disk_gb},
                                 "containerSpec": {"imageUri": image or (GPU_IMAGE if gpu else CPU_IMAGE),
                                                   "command": ["/bin/bash", "-c"],
                                                   "args": [exports + "\n\n" + boot]}}],
            "scheduling": {"strategy": "SPOT", "restartJobOnWorkerRestart": True,
                           "timeout": f"{int(timeout_h * 3600)}s"}}
    if service_account:
        out["serviceAccount"] = service_account
    return out


def submit(name: str, job_spec: dict, machine: str, timeout_h: float, go: bool = False,
           labels: dict | None = None, note: str = "") -> str | None:
    """Print the plan and return None unless `go`; otherwise gate on budget, submit, and record."""
    worst = ledger.RATES[machine][0] * timeout_h
    if not go:
        print(f"[DRY RUN] {name}: {machine} spot, timeout {timeout_h}h, worst ${worst:.2f}")
        return None
    assert job_spec["scheduling"]["strategy"] == "SPOT", "spot only"
    ledger.gate(machine, timeout_h)
    jid = V.create_job(name, job_spec, {**LABELS, **(labels or {})})
    ledger.record(name, jid, machine, timeout_h, note)
    print(f"{name}: submitted {jid} (worst ${worst:.2f})")
    return jid
