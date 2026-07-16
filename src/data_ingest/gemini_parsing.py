import os
import json
import re
import pandas as pd
import time
from pathlib import Path
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from google.cloud import storage
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import CONFIG
from labeling.labelbox_api.findings_prepare import anonymize_dates_times_and_names

# Initialize Vertex AI client
# NOTE: Gemini 3.x models are only served from the global endpoint (regional
# batch returns 404 MODEL_NOT_SUPPORTED_FOR_BATCH), but this project's
# gcp.resourceLocations org policy blocks the global endpoint entirely
# (400 FAILED_PRECONDITION). Until batch support lands in-region, only
# 2.5-series models work here.
os.environ['GOOGLE_CLOUD_PROJECT'] = CONFIG['env']['project_id']
os.environ['GOOGLE_CLOUD_LOCATION'] = CONFIG['env']['region']
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'

client = genai.Client(http_options=HttpOptions(api_version="v1"))
storage_client = storage.Client()

SYSTEM_PROMPT = """You are a medical data extraction assistant specializing in breast imaging reports.

Your task: Extract ALL lesions mentioned in radiology text and output them as a JSON array.

Each report begins with a header line stating the study laterality, e.g.: Study Laterality: "LEFT" (possible values: "LEFT", "RIGHT", "BILATERAL").

Each lesion should have these fields:
- laterality: which breast the lesion is in: "LEFT" or "RIGHT", or "BILATERAL"
- direction: clock position (e.g., "2:00", "12:00") or "na" if not specified
- distance: distance from nipple (e.g., "4cm", "2cm") or "na" if not specified
- size: maximum dimension (e.g., "5mm", "1.2cm") or "na" if not specified
- type: lesion description (e.g., "oval circumscribed mass") or "na" if unclear. Include lesion type: "mass, lymph node, cyst, etc.". Other characteristics of interest include: "circumscribed, macrolobulated, microlobulated, indistinct, angular, spiculated, oval, round, irregular, parallel, not parallel, anechoic, hypoechoic, isoechoic, hyperechoic, complex, heterogeneous, no posterior features, enhancement, shadowing, combined pattern, abrupt interface, echogenic halo, architectural distortion"

CRITICAL RULES:
1. When multiple locations share the same size/type (distributed attributes), create separate entries for EACH location with the shared attributes repeated.
2. Use "na" for any missing values - never guess.
3. Output ONLY valid JSON - no explanation, no markdown code blocks.
4. Order lesions as they appear in the text.
5. Normalize units: keep as written (don't convert mm to cm or vice versa).
6. For size ranges like "up to 3mm", use the maximum value.
7. Laterality: if the study laterality is "LEFT" or "RIGHT", assign that side to every lesion (the study laterality is authoritative, even if the surrounding text mentions the other breast). If the study laterality is "BILATERAL", use context clues (e.g. "left breast", "right breast at 12:00") to assign each lesion's side; use "na" only when the side truly cannot be determined.

Output format:
[{"laterality": "...", "direction": "...", "distance": "...", "size": "...", "type": "..."}, ...]

If no lesions are found, output: []
"""

FEW_SHOT_EXAMPLES = [
    (
        """Study Laterality: "LEFT"
        Additional evaluation was performed for an asymmetry within the left
        upper outer breast, which persists as a well-circumscribed ovoid 5 mm
        mass on additional diagnostic imaging. Targeted ultrasound was performed
        within the left upper outer quadrant which demonstrates a well-circumscribed
        fibrocystic complex measuring 5 mm x 2 mm x 2 mm in the left breast 2:00 4 cm from the
        nipple. Incidental note made of a 3 mm anechoic cyst within the left 6-7 o'clock
        periareolar position. No suspicious masses or other abnormalities
        are identified.""",
        [
            {"laterality": "LEFT", "direction": "2:00", "distance": "4cm", "size": "5mm", "type": "circumscribed ovoid mass"},
            {"laterality": "LEFT", "direction": "6:30", "distance": "0cm", "size": "3mm", "type": "circumscribed fibrocystic complex"}
        ]
    ),
    (
        """Study Laterality: "LEFT"
        Ultrasound of the left breast upper outer quadrant at 1:00, 2 cm from
        the nipple, 2:00, 3 cm from the nipple, and 3:00 6 cm from the nipple
        show multiple small benign cysts measuring up to 3 mm x 3 mm x 2 mm
        which account for the mammographic findings.""",
        [
            {"laterality": "LEFT", "direction": "1:00", "distance": "2cm", "size": "3mm", "type": "benign cyst"},
            {"laterality": "LEFT", "direction": "2:00", "distance": "3cm", "size": "3mm", "type": "benign cyst"},
            {"laterality": "LEFT", "direction": "3:00", "distance": "6cm", "size": "3mm", "type": "benign cyst"}
        ]
    ),
    (
        """Study Laterality: "LEFT"
        There is an oval hypoechoic mass measuring 3.8 x 1.4 x 3.6 cm with well defined, 
        thin margins in the right breast upper outer quadrant at 10 o'clock.  The mass is parallel to the chest wall.  
        This is at the site of palpable concern marked on skin. Ultrasound-guided biopsy recommended.    
        Findings and recommendations were discussed with the patient and Dr. Hines.    Ultrasound guided core biopsy is recommended.       
        BI-RADS Category 4: Suspicious Abnormality       Addendum:    
        Pathology results from US-guided right breast biopsy in the 10 o'clock position performed on 10/7/11 reveal fibroadenoma.  
        (Refer to pathology report for detailed description.) This is a benign, concordant and specific diagnosis. Given size of lesion, 
        surgical consult recommended for excision.""",
        [
            {"laterality": "LEFT", "direction": "10:00", "distance": "na", "size": "3.8cm", "type": "parallel oval hypoechoic mass"}
        ]
    ),
    (
        """Study Laterality: "BILATERAL"
        Bilateral diagnostic mammogram with spot compression views of the outer and inner left breast were performed.
        There are waxing and waning similar-appearing oval and lobulated masses in the upper outer left breast and a
        superficial oval mass in the lower inner left breast. Stable radiopaque marker in the upper inner left breast
        at the site of prior benign biopsy. No suspicious findings are seen. Intact bilateral retropectoral saline implants.
        Targeted ultrasound of the prior area of sonographic abnormality in the right breast at 12:00, 7 cm from nipple
        demonstrates a stable circumscribed, hypoechoic mass measuring 0.7 x 0.5 x 0.6 cm (previously measured 0.6 x 0.5 x 0.6 cm).
        Sonographic survey of the outer left breast demonstrates ductal ectasia and numerous similar-appearing oval,
        circumscribed, hypoechoic masses, which are most compatible with simple and complicated cysts and correlate
        with the mammographic findings.""",
        [
            {"laterality": "RIGHT", "direction": "12:00", "distance": "7cm", "size": "0.7cm", "type": "circumscribed oval hypoechoic mass"}
        ]
    ),
]


def build_vertex_contents(text: str) -> list[dict]:
    """Build Vertex AI contents format with few-shot examples."""
    contents = []
    
    # Add system prompt as first user message
    contents.append({
        "role": "user",
        "parts": [{"text": SYSTEM_PROMPT}]
    })
    contents.append({
        "role": "model",
        "parts": [{"text": "Understood. I will extract lesions according to these rules."}]
    })
    
    # Add few-shot examples
    for example_input, example_output in FEW_SHOT_EXAMPLES:
        contents.append({
            "role": "user",
            "parts": [{"text": example_input.strip()}]
        })
        contents.append({
            "role": "model",
            "parts": [{"text": json.dumps(example_output)}]
        })
    
    # Add actual query
    contents.append({
        "role": "user",
        "parts": [{"text": text.strip()}]
    })
    
    return contents


def format_as_label(lesions: list[dict]) -> str:
    """Convert lesion dicts back to label format: [laterality, dir, dist, size, type]"""
    labels = []
    for l in lesions:
        label = f"[{l.get('laterality', 'na')}, {l['direction']}, {l['distance']}, {l['size']}, {l['type']}]"
        labels.append(label)
    return ", ".join(labels)


def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    """Upload file to GCS and return gs:// URI."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def download_from_gcs(gcs_uri: str, local_path: str):
    """Download file from GCS."""
    # Parse gs://bucket/path
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def get_batch_error_details(job):
    """Get detailed error information from a failed batch job."""
    print(f"\n{'='*60}")
    print("ERROR DETAILS")
    print(f"{'='*60}")
    print(f"Job name: {job.name}")
    print(f"State: {job.state}")
    
    # Try to get error from job object
    if hasattr(job, 'error'):
        print(f"Error: {job.error}")
    
    # The job object might have more details
    print(f"\nFull job info:")
    print(job)
    print(f"{'='*60}")
    
def submit_batch(
    input_jsonl_path: str,
    gcs_bucket: str,
    gcs_input_prefix: str,
    gcs_output_prefix: str,
    model: str = "gemini-2.5-flash-lite"
):
    """
    Submit a batch job to Vertex AI.
    
    Args:
        input_jsonl_path: Local path to JSONL file
        gcs_bucket: GCS bucket name (without gs://)
        gcs_input_prefix: Path prefix in bucket for input
        gcs_output_prefix: Path prefix in bucket for output
        model: Gemini model to use
    
    Returns:
        Batch job object
    """
    print(f"\nSubmitting batch from: {input_jsonl_path}")
    
    # Upload input file to GCS
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_blob_name = f"{gcs_input_prefix}/batch_input_{timestamp}.jsonl"
    input_uri = upload_to_gcs(input_jsonl_path, gcs_bucket, input_blob_name)
    print(f"Uploaded input to: {input_uri}")
    
    # Create output URI
    output_uri = f"gs://{gcs_bucket}/{gcs_output_prefix}/batch_output_{timestamp}/"
    
    # Submit batch job
    job = client.batches.create(
        model=model,
        src=input_uri,
        config=CreateBatchJobConfig(dest=output_uri)
    )
    
    print(f"\nBatch job created: {job.name}")
    print(f"Job state: {job.state}")
    print(f"Output will be at: {output_uri}")
    
    return job, output_uri


def check_batch_status(job_name: str):
    """Check the status of a batch job."""
    # List all batches and find the one with matching name
    for job in client.batches.list():
        if job.name == job_name:
            print(f"\nJob name: {job.name}")
            print(f"State: {job.state}")
            print(f"Create time: {job.create_time}")
            return job
    
    raise ValueError(f"Job {job_name} not found")


def wait_for_batch(job_name: str, check_interval: int = 60):
    """Wait for batch job to complete."""
    print(f"\nWaiting for batch job to complete...")
    print(f"Checking every {check_interval} seconds...")
    
    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_EXPIRED
    }
    
    start_time = time.time()
    
    while True:
        # Find the job by listing and filtering
        job = None
        for batch_job in client.batches.list():
            if batch_job.name == job_name:
                job = batch_job
                break
        
        if job is None:
            raise ValueError(f"Job {job_name} not found")
        
        elapsed = time.time() - start_time
        print(f"\rState: {job.state.name} | Elapsed: {elapsed/60:.1f}m", end="", flush=True)
        
        if job.state in completed_states:
            print(f"\n\nBatch completed with state: {job.state.name}")
            print(f"Total time: {elapsed/60:.1f} minutes")
            return job
        
        time.sleep(check_interval)


def download_batch_results(output_uri: str, local_output_dir: str):
    """
    Download results from GCS. Results should be in a single file.
    """
    print(f"\nDownloading results from: {output_uri}")
    
    # Parse bucket and prefix
    parts = output_uri.replace("gs://", "").rstrip("/").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Download all result files
    result_files = []
    for blob in blobs:
        if blob.name.endswith('.jsonl'):
            local_path = Path(local_output_dir) / Path(blob.name).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            result_files.append(str(local_path))
            print(f"Downloaded: {blob.name}")
    
    if not result_files:
        raise ValueError("No result files found!")
    
    # Sort files alphabetically (important if multiple files)
    result_files.sort()
    
    print(f"\n{len(result_files)} result file(s) found")
    if len(result_files) > 1:
        print("WARNING: Multiple result files - processing in alphabetical order:")
        for i, f in enumerate(result_files):
            print(f"  {i+1}. {Path(f).name}")
    
    return result_files

def create_batch_jsonl(
    csv_path: str, 
    output_jsonl: str, 
    text_column: str = "ultrasound_findings",
    model: str = "gemini-2.5-flash"
):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    # Build few-shot examples ONCE (not system instruction - put in contents)
    few_shot_contents = []
    for example_input, example_output in FEW_SHOT_EXAMPLES:
        few_shot_contents.extend([
            {"role": "user", "parts": [{"text": example_input.strip()}]},
            {"role": "model", "parts": [{"text": json.dumps(example_output)}]}
        ])
    
    num_requests = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            text = row[text_column]
            if pd.isna(text) or not str(text).strip():
                continue
            
            anonymized_text = anonymize_dates_times_and_names(str(text))

            # Prepend the study laterality header (matches few-shot example format)
            laterality = row.get('Study_Laterality')
            laterality = str(laterality).upper().strip() if pd.notna(laterality) else "na"
            input_text = f'Study Laterality: "{laterality}"\n{anonymized_text}'

            # Few-shot + actual query
            contents = few_shot_contents + [
                {"role": "user", "parts": [{"text": input_text}]}
            ]
            
            batch_request = {
                "custom_id": f"row_{int(idx)}",
                "request": {
                    "system_instruction": {
                        "parts": [{"text": SYSTEM_PROMPT}]
                    },
                    "contents": contents
                }
            }
            
            f.write(json.dumps(batch_request) + '\n')
            num_requests += 1
    
    print(f"\nCreated batch file: {output_jsonl}")
    print(f"Total requests: {num_requests}")
    
    return num_requests


def parse_batch_results_with_comparison(
    result_files: list[str],
    original_csv: str,
    output_csv: str,
    index_file: str = None  # No longer needed
):
    """Parse results using custom_id from each result."""
    print(f"\nParsing results from {len(result_files)} file(s)")
    print(f"Loading original data from: {original_csv}")
    
    # Load original CSV
    df_original = pd.read_csv(original_csv)
    
    results = []
    
    # Parse all result files
    for result_file in result_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                
                # Extract row index from custom_id
                custom_id = result.get('custom_id', '')
                if not custom_id.startswith('row_'):
                    print(f"Warning: Unexpected custom_id format: {custom_id}")
                    continue
                
                row_idx = int(custom_id.split('_')[1])
                
                # Parse response
                try:
                    response = result.get('response', {})
                    candidates = response.get('candidates', [])
                    
                    if not candidates:
                        raise ValueError("No candidates in response")
                    
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    
                    if not parts:
                        raise ValueError("No parts in content")
                    
                    raw_output = parts[0].get('text', '').strip()
                    
                    if raw_output.startswith("```"):
                        raw_output = re.sub(r"```(?:json)?\n?", "", raw_output).strip()
                    
                    lesions = json.loads(raw_output)
                    prediction = format_as_label(lesions)
                    prediction_json = json.dumps(lesions)
                    num_lesions = len(lesions)
                    error = None
                    
                except Exception as e:
                    prediction = ""
                    prediction_json = "[]"
                    num_lesions = 0
                    error = str(e)
                
                # Get original row
                try:
                    original_row = df_original.loc[row_idx]
                except KeyError:
                    print(f"Warning: row_idx {row_idx} not found in DataFrame")
                    continue
                
                results.append({
                    'row_index': row_idx,
                    'custom_id': custom_id,
                    'input_text': original_row.get('ultrasound_findings', ''),
                    'structured_output': original_row.get('structured_output', ''),
                    'prediction': prediction,
                    'prediction_json': prediction_json,
                    'num_lesions': num_lesions,
                    'error': error
                })
    
    # Create DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('row_index')
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\nSaved {len(df)} results to: {output_csv}")
    print(f"Successful extractions: {df['error'].isna().sum()}")
    print(f"Errors: {df['error'].notna().sum()}")
    
    # Print sample comparison
    if len(df) > 0:
        print("\n=== SAMPLE COMPARISON ===")
        sample = df[df['error'].isna()].head(3)
        for idx, row in sample.iterrows():
            print(f"\n--- Row {row['row_index']} ---")
            print(f"Input: {row['input_text'][:100]}...")
            print(f"Original: {row['structured_output']}")
            print(f"Predicted: {row['prediction']}")
    
    return df

def run_batch_pipeline(
    csv_path: str,
    gcs_bucket: str,
    text_column: str = "ultrasound_findings",
    model: str = "gemini-2.5-flash-lite",
    output_dir: str = "batch_results",
    gcs_input_prefix: str = "batch_input",
    gcs_output_prefix: str = "batch_output",
    wait: bool = True,
    batch_size: int = 200000,
    limit: int = None,
    final_output_csv: str = None
):
    """
    Complete Vertex AI batch pipeline.

    Args:
        csv_path: Path to input CSV
        gcs_bucket: GCS bucket name (without gs://)
        text_column: Column containing text to process
        model: Gemini model to use
        output_dir: Local directory to save outputs
        gcs_input_prefix: GCS prefix for input files
        gcs_output_prefix: GCS prefix for output files
        wait: Whether to wait for batch completion
        batch_size: Number of rows to process per batch (default: 200000)
        limit: Optional limit on number of rows to process (default: None processes all rows)
        final_output_csv: Path to save final consolidated results CSV (default: None)

    Returns:
        Dictionary with job info and output paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load CSV to determine batching
    print(f"Loading CSV to determine batching: {csv_path}")
    df_full = pd.read_csv(csv_path)

    # Apply limit if specified
    if limit is not None:
        df_full = df_full.head(limit)
        print(f"Limiting to first {limit} rows")

    # Filter out rows where ultrasound_findings is empty
    initial_count = len(df_full)
    df_full = df_full[df_full[text_column].notna() & (df_full[text_column].astype(str).str.strip() != '')]
    df_full = df_full.reset_index(drop=True)
    filtered_count = initial_count - len(df_full)

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows with empty {text_column}")

    total_rows = len(df_full)

    if total_rows == 0:
        print("No valid rows to process!")
        return {
            'total_batches': 0,
            'batch_results': []
        }

    num_batches = (total_rows + batch_size - 1) // batch_size

    print("="*60)
    print("VERTEX AI GEMINI BATCH PIPELINE")
    print("="*60)
    print(f"Model: {model}")
    print(f"GCS Bucket: gs://{gcs_bucket}")
    print(f"Total rows: {total_rows}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print("="*60)

    all_results = []

    # Process each batch
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_rows)

        print(f"\n{'='*60}")
        print(f"PROCESSING BATCH {batch_num + 1}/{num_batches}")
        print(f"Rows {start_idx} to {end_idx-1}")
        print(f"{'='*60}")

        # Create temp CSV for this batch
        batch_csv = output_path / f"temp_batch_{batch_num}.csv"
        df_batch = df_full.iloc[start_idx:end_idx]
        df_batch.to_csv(batch_csv, index=False)

        # Generate filenames for this batch
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_jsonl = output_path / f"batch_input_{timestamp}_batch{batch_num}.jsonl"
        results_csv = output_path / f"batch_results_{timestamp}_batch{batch_num}.csv"

        # Step 1: Create batch file
        num_requests = create_batch_jsonl(str(batch_csv), str(batch_jsonl), text_column, model)

        # Step 2: Submit batch
        job, output_uri = submit_batch(
            str(batch_jsonl),
            gcs_bucket,
            gcs_input_prefix,
            gcs_output_prefix,
            model
        )

        # Save job name to file
        job_name_file = output_path / f"job_name_{timestamp}_batch{batch_num}.txt"
        with open(job_name_file, 'w') as f:
            f.write(job.name)
        print(f"\nJob name saved to: {job_name_file}")

        if not wait:
            print(f"\nBatch {batch_num + 1} submitted! Not waiting for completion.")
            all_results.append({
                'batch_num': batch_num,
                'job_name': job.name,
                'job_name_file': str(job_name_file),
                'output_uri': output_uri,
                'rows': f"{start_idx}-{end_idx-1}"
            })
            continue

        # Step 3: Wait for completion
        job = wait_for_batch(job.name)

        if job.state != JobState.JOB_STATE_SUCCEEDED:
            print(f"\nBatch {batch_num + 1} did not succeed: {job.state}")
            get_batch_error_details(job)
            all_results.append({
                'batch_num': batch_num,
                'job_name': job.name,
                'status': job.state,
                'output_uri': output_uri,
                'rows': f"{start_idx}-{end_idx-1}"
            })
            continue

        # Step 4: Download results
        result_files = download_batch_results(output_uri, str(output_path))

        # Step 5: Parse results with comparison
        df_batch_results = parse_batch_results_with_comparison(
            result_files,
            str(batch_csv),
            str(results_csv),
        )

        all_results.append({
            'batch_num': batch_num,
            'job_name': job.name,
            'batch_input': str(batch_jsonl),
            'results_csv': str(results_csv),
            'output_uri': output_uri,
            'rows': f"{start_idx}-{end_idx-1}",
            'num_results': len(df_batch_results)
        })

        # Clean up temp batch CSV
        batch_csv.unlink()

    print("\n" + "="*60)
    print("ALL BATCHES COMPLETE!")
    print("="*60)
    for result in all_results:
        print(f"Batch {result['batch_num'] + 1}: {result.get('num_results', 'pending')} results | Rows {result['rows']}")
    print("="*60)

    # If final_output_csv is specified, combine all batch results into final format
    if final_output_csv and wait:
        print(f"\nCreating final output CSV: {final_output_csv}")

        # Collect all batch result DataFrames
        all_batch_dfs = []
        for result in all_results:
            if 'results_csv' in result:
                df_batch = pd.read_csv(result['results_csv'])
                all_batch_dfs.append(df_batch)

        if all_batch_dfs:
            # Combine all batch results
            df_combined = pd.concat(all_batch_dfs, ignore_index=True)
            df_combined = df_combined.sort_values('row_index')

            # Use the filtered df_full (already has limit applied and empty rows removed)
            # Create final output with only required columns
            df_final = pd.DataFrame({
                'PATIENT_ID': df_full['PATIENT_ID'].values,
                'ACCESSION_NUMBER': df_full['ACCESSION_NUMBER'].values,
                'lesion_descriptions': df_combined['prediction'].values
            })

            # Save final CSV
            df_final.to_csv(final_output_csv, index=False)
            print(f"Saved final results to: {final_output_csv}")
            print(f"Total rows: {len(df_final)}")

            return {
                'total_batches': num_batches,
                'batch_results': all_results,
                'final_output_csv': final_output_csv,
                'final_dataframe': df_final
            }

    return {
        'total_batches': num_batches,
        'batch_results': all_results
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    result = run_batch_pipeline(
        csv_path='/data/endpoint_data.csv',
        gcs_bucket=CONFIG['BUCKET'],
        text_column='ultrasound_findings',
        model='gemini-2.5-flash',
        output_dir='batch_results_gemini',
        gcs_input_prefix='cadbusi/batch_input',
        gcs_output_prefix='cadbusi/batch_output',
        wait=True,
        batch_size=200000,
        limit=None,  # Set to a number like 1000 to process only first N rows
        final_output_csv='/data/birads.csv'
    )