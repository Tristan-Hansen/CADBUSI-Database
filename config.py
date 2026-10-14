"""Shared pipeline configuration. Committed: keep it identical on every machine.

Secrets do NOT go here. The Labelbox API key is read from the gitignored
`labelbox-credentials.env` (KEY=value) next to this file, or from the
LABELBOX_API_KEY environment variable, which wins if both are set.
"""
import os
from pathlib import Path

from src.credentials import parse

_secrets = Path(__file__).with_name("labelbox-credentials.env")
if _secrets.is_file():
    for _k, _v in parse(_secrets.read_text(encoding="utf-8")).items():
        os.environ.setdefault(_k, _v)

CONFIG = {
    # Environment configuration
    "env": {
        "project_id": "aif-usr-p-rad-brstai-87d1",
        "region": "us-central1",
        "topic_name": "dicom-processing-topic",
        "subscription_name": "dicom-processing-subscription",
        "my_service_account": "M302453",
        "service_account_identity": "gsa6-va-prj-aif-p-M302453-87d1@aif-usr-p-rad-brstai-87d1.iam.gserviceaccount.com"
    },

    # Cloud Run configuration
    "cloud_run": {
        "service": "pubsub-push-cloudrun",
        "version": "1.0",
        "ar": "shared-aif-artifact-registry-docker-87d1",
        "ar_name": "pubsub-push-cloudrun",
        "target_tag": "us-central1-docker.pkg.dev/aif-usr-p-rad-brstai-87d1/shared-aif-artifact-registry-docker-87d1/pubsub-push-cloudrun:1.0",
        "vpc_shared": "aif-env-sharedvpc-148f",
        "vpc_name": "aif-vpc-p-ops-auto-01"
    },

    # Storage configuration
    "storage": {
        "gcs_log": "gs://shared-aif-bucket-87d1/cloudbuild_log",
        "gcs_stage": "gs://shared-aif-bucket-87d1/cloudbuild_stage",
        "bucket_name": "shared-aif-bucket-87d1",
        "download_path": "Downloads/2025-11-26-5",
    },

    "BUCKET": "shared-aif-bucket-87d1",
    "WINDIR": "D:/DATA/CADBUSI/",
    "DATABASE_DIR": "Databases/database_2026_1_13_main/",

    "LABELBOX_API_KEY": os.environ.get("LABELBOX_API_KEY", ""),
    "PROJECT_ID": "cmfhc9qf9023k07ykgk187rw0",  # Labelbox project for retreive_lesion_cropping.py
    "LABELBOX_LABELS": "labelbox_data/",
    "TARGET_CASES": "D:/DATA/CASBUSI/failed_cases.csv",
    "VIDEO_SAMPLING": 0, # every nth frame, 0 turns off videos
    "DEBUG_DATA_RANGE": [0, 10000],
}
