# CADBUSI-Database

SQLite database manager that is designed to process breast ultrasound data from Mayo Clinic and store it in a structured format, making it easy to anonymize, manipulate, label, analyze, and prepare for ML research.

## Setup

- Create a GCP AI Factory instance with a T4 GPU
- Clone repository: `git clone https://github.com/Poofy1/CADBUSI-Database.git`
- Go into dir: `cd CADBUSI-Database`
- Install requirements: `pip install -r requirements.txt`
- Create/Configure: `./config.py`
- Obtain certificate `./src/dicom_downloader/CertEmulationCA.crt`
- Obtain user privileges: `gcloud auth application-default login --no-launch-browser`

## Configuration
All user parameters will be controlled from a `./config.py` file, you will need to configure the following parameters:
```
CONFIG = {
    # Environment configuration
    "env": {
        "project_id": "your-project-id",
        "region": "us-central1",
        "topic_name": "dicom-processing-topic",
        "subscription_name": "dicom-processing-subscription",
        "my_service_account": "your-service-id",
        "service_account_identity": "your-service-account@your-project-id.iam.gserviceaccount.com"
    },
    
    # Cloud Run configuration
    "cloud_run": {
        "service": "pubsub-push-cloudrun",
        "version": "1.0",
        "ar": "your-artifact-registry",
        "ar_name": "pubsub-push-cloudrun",
        "target_tag": "us-central1-docker.pkg.dev/your-project-id/your-artifact-registry/pubsub-push-cloudrun:1.0",
        "vpc_shared": "your-shared-vpc-id",
        "vpc_name": "your-vpc-name"
    },
    
    # Storage configuration
    "storage": {
        "gcs_log": "gs://your-bucket-name/cloudbuild_log",
        "gcs_stage": "gs://your-bucket-name/cloudbuild_stage",
        "bucket_name": "your-bucket-name",
        "download_path": "Downloads",
    },
    
    "BUCKET": "your-bucket-name",
    "WINDIR": "D:/DATA/YOUR_PROJECT/",
    "DATABASE_DIR": "Databases/database_YYYY_MM_DD/", # Final location of the database.

    "LABELBOX_API_KEY": "your-labelbox-api-key",
    "PROJECT_ID": "your-labelbox-project-id",
    "LABELBOX_LABELS": "labelbox_data/",
    "TARGET_CASES": "/failed_cases.csv", # Directory of worst performing cases from training. Prepares these cases for instance labeling on Label Box.
    "VIDEO_SAMPLING": 0, # every nth frame, 0 turns off videos
    "DEBUG_DATA_RANGE": [], # Process a reduced set of dicom files (Ex: [0, 1000]).
}
```
## Usage
The pipeline is operated through a single command-line interface in main.py, which provides several functions. For general purpose, you should perform these commands in this order: 

```
python main.py --query [optional: --limit=N]
python main.py --deploy
python main.py --cleanup
python main.py --database [optional: --existing-db path/to/old_cadbusi.db]
python main.py --merge-db <src.db> <dest.db>   # manual merge after an incremental --database run
```



### Querying Data

To query breast imaging data:
`python main.py --query [optional: --limit=N]`

This will:
1. Run a query to retrieve breast imaging records
2. Filter and clean the radiology and pathology data
3. Create a final dataset for processing
4. Save results to `query_data/endpoint_data.csv`

Example with a limit:`python main.py --query --limit=100`

#### Query Diagram `--query [optional: limit=N]`
![CADBUSI Query](/demo/CADBUSI_Query.png)

### Downloading DICOM Files

The tool offers Cloud Run deployment for efficient DICOM downloads. Dicoms will appear in specified GCP bucket storage:
```
# Deploy the FastAPI service to Cloud Run and start dicom data download (REQUIRED)
python main.py --deploy

# Resend the download requests to the pre-deployed service (OPTIONAL)
python main.py --rerun 

# Clean up Cloud Run resources when finished (REQUIRED)
python main.py --cleanup
```

IMPORTANT: After `python main.py --deploy` finishes execution, that does not mean the data transfer is complete. The download requests have been sent to Cloud Run. Check the bucket storage to see when population is finished. Only then should you run `python main.py --cleanup`

### Processing DICOM Files

To process the downloaded DICOM files into a complete database:

`python main.py --database [optional: --skip-inpaint]`

This will:
1. Generate encryption keys for safely anonymizing patient IDs
2. Deidentify DICOM files from `CONFIG['storage']['download_path']`
3. Process image files in the specified output directory in the destination bucket
4. `[optional: --skip-inpaint]` will skip the caliper removal process

Example:

`python main.py --database`

### Incremental Processing

To parse only DICOMs that aren't already in a previous database, pass `--existing-db`:

`python main.py --database --existing-db path/to/old_cadbusi.db`

This will:
1. Open the provided DB read-only and collect every `(patient_id, accession_number)` pair in `StudyCases`.
2. Skip any DICOM whose pair is already present, so only new data is parsed.
3. Write a fresh `cadbusi.db` to `CONFIG["DATABASE_DIR"]/cadbusi.db` containing only the new cases.

The old DB is never modified. Use the same encryption key as the previous run — the DB stores encrypted IDs.

### Merging Databases

After verifying an incremental run, merge the new subset into the old DB:

`python main.py --merge-db path/to/new_subset.db path/to/old_cadbusi.db [--anon-data path/to/anon_data.csv]`

This copies every row from each table in `new_subset.db` into `old_cadbusi.db` via `INSERT OR IGNORE` (existing rows are kept).

If `--anon-data` is provided, after the table-by-table merge the destination's `StudyCases` rows get refreshed from the CSV: clinical fields (`left_diagnosis`, `right_diagnosis`, `bi_rads`, `findings`, etc.) on accession_numbers that already exist in the destination are updated to the CSV's values. Rows in the CSV whose accession_number is not already in the destination are skipped — no placeholder rows are inserted.

Only the `.db` is merged — you are responsible for copying the corresponding `images/` and `videos/` files from the new run's `DATABASE_DIR` into the destination's `images/` and `videos/` folders so the DB rows still point at files that exist on disk.

## Labels Database

Annotation labels (`ImageLabels`, `LesionLabels`, `CaliperLabels`) live in a separate `cadbusi_labels.db`, managed via `src/DB_processing/labels_database.py`.

**Sync with GCS bucket:**

Downloads from / uploads to `gs://<bucket_name>/databases/cadbusi_labels.db`. The local working copy is saved to `CONFIG["DATABASE_DIR"]/cadbusi_labels.db`.

```python
from src.DB_processing.labels_database import open_labels_db_from_bucket, update_labels_db_in_bucket
from tools.storage_adapter import StorageClient
from config import CONFIG

StorageClient.get_instance(bucket_name=CONFIG["storage"]["bucket_name"]) # Example: "DATABASE_DIR": "Databases/database_2026_1_13_main/",

# Download latest version
db = open_labels_db_from_bucket()

# Edit database
db.insert_image_labels_batch([{"dicom_hash": "abc123", "cyst": 1, "version": "v1"}])
db.close()

# Upload new database
update_labels_db_in_bucket()
```

Override the local path or bucket path via `local_path=` / `bucket_path=` on any call.

## Data Pipeline
- [CADBUSI-Database](https://github.com/Poofy1/CADBUSI-Database)
- [CADBUSI-Training](https://github.com/Poofy1/CADBUSI-Training)
