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
GPU_ACCELERATOR = {"a2-highgpu-1g": ("NVIDIA_TESLA_A100", 1)}
LABELS = {"pipeline": "cadbusi-database"}


def spec(env: dict, boot: str, machine: str, timeout_h: float, disk_gb: int = 200) -> dict:
    """CustomJob spec: one SPOT worker running `boot` under bash with `env` exported first."""
    exports = "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in env.items() if v is not None)
    gpu = machine in GPU_ACCELERATOR
    ms = {"machineType": machine}
    if gpu:
        ms["acceleratorType"], ms["acceleratorCount"] = GPU_ACCELERATOR[machine]
    return {"workerPoolSpecs": [{"machineSpec": ms, "replicaCount": 1,
                                 "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": disk_gb},
                                 "containerSpec": {"imageUri": GPU_IMAGE if gpu else CPU_IMAGE,
                                                   "command": ["/bin/bash", "-c"],
                                                   "args": [exports + "\n\n" + boot]}}],
            "scheduling": {"strategy": "SPOT", "restartJobOnWorkerRestart": True,
                           "timeout": f"{int(timeout_h * 3600)}s"}}


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
