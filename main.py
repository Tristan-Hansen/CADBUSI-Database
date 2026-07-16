import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from src.encrypt_keys import encrypt_ids

from tools.storage_adapter import * 
from config import CONFIG
import argparse
import os
import sys

env = os.path.dirname(os.path.abspath(__file__))



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DICOM processing pipeline')
    
    # Query arguments
    parser.add_argument('--query', action='store_true', help='Run breast imaging query')
    parser.add_argument('--limit', type=int, help='Optional limit for the query/export')
    
    # Download arguments
    parser.add_argument('--deploy', action='store_true', help='Deploy FastAPI to Cloud Run')
    parser.add_argument('--rerun', action='store_true', help='Send message to pre-deployed FastAPI on Cloud Run')
    parser.add_argument('--cleanup', action='store_true', help='Clean up resources')

    # BIRAD parsing arguments
    parser.add_argument('--birad_parsing', action='store_true', help='Run Gemini BIRAD lesion parsing on endpoint data')

    # Anonymize arguments
    parser.add_argument('--database', action='store_true', help='Process database')
    parser.add_argument('--skip-inpaint', action='store_true', help='Skip the inpainting step')
    parser.add_argument('--existing-db', type=str, default=None,
        help='Path to a previous cadbusi.db; skip (patient_id, accession_number) pairs already in it')

    # Merge: takes SRC and DEST paths; runs INSERT OR IGNORE per table
    parser.add_argument('--merge-db', nargs=2, metavar=('SRC', 'DEST'),
        help='Merge SRC cadbusi.db into DEST cadbusi.db via INSERT OR IGNORE')
    parser.add_argument('--anon-data', type=str, default=None,
        help='Path to anon_data.csv; when set with --merge-db, refreshes clinical '
             'fields on StudyCases rows that already exist in DEST. Rows in the '
             'CSV with no matching accession in DEST are skipped (no placeholder '
             'rows inserted).')

    return parser.parse_args()

def main():
    # Determine storage client
    storage = StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    
    # Main entry point for the script
    args = parse_arguments()
    OUTPUT_PATH = os.path.join(env, "data")
    DICOM_QUERY_PATH = os.path.join(OUTPUT_PATH, 'endpoint_data.csv')
    DATABASE_LOCAL_PATH = os.path.join(OUTPUT_PATH, 'cadbusi.db')
    DATABASE_GCP_PATH = f'{CONFIG["DATABASE_DIR"]}/cadbusi.db'
    
    
    # Handle query command
    if args.query:
        from src.data_ingest.query import run_breast_imaging_query
        from src.data_ingest.clean_pathology import filter_path_data
        from src.data_ingest.clean_radiology import filter_rad_data
        from src.data_ingest.filter_data import create_final_dataset
        limit = args.limit
        
        # Run the query with the specified limit
        rad_df, path_df, surgery_df = run_breast_imaging_query(limit=limit)

        # Parse that data
        rad_df = filter_rad_data(rad_df, OUTPUT_PATH, surgery_df=surgery_df)
        path_df = filter_path_data(path_df, OUTPUT_PATH)

        # Filter data
        create_final_dataset(rad_df, path_df, OUTPUT_PATH, surgery_df=surgery_df)
    
    elif args.merge_db:
        from src.DB_processing.db_merge import merge_databases
        src_path, dest_path = args.merge_db
        merge_databases(src_path, dest_path, anon_data_path=args.anon_data)

    elif args.deploy or args.cleanup or args.rerun:
        from src.dicom_downloader.dicom_download import dicom_download_remote_start
        dicom_download_remote_start(DICOM_QUERY_PATH, args.deploy, args.cleanup)

    elif args.birad_parsing:
        from src.data_ingest.gemini_parsing import run_batch_pipeline

        print("="*60)
        print("BIRAD LESION PARSING WITH GEMINI")
        print("="*60)

        # Set paths
        input_csv = DICOM_QUERY_PATH  # /data/endpoint_data.csv
        output_csv = os.path.join(OUTPUT_PATH, 'birads.csv')

        if args.limit:
            print(f"Processing with limit: {args.limit} rows")
        else:
            print("Processing all rows")

        # Run batch pipeline
        result = run_batch_pipeline(
            csv_path=input_csv,
            gcs_bucket=CONFIG['BUCKET'],
            text_column='ultrasound_findings',
            model='gemini-2.5-flash',
            output_dir='batch_results_gemini',
            gcs_input_prefix='cadbusi/batch_input',
            gcs_output_prefix='cadbusi/batch_output',
            wait=True,
            batch_size=200000,
            limit=args.limit,
            final_output_csv=output_csv
        )

        print("\n" + "="*60)
        print("BIRAD PARSING COMPLETE")
        print("="*60)
        print(f"Output saved to: {output_csv}")
        print("="*60)

    elif args.database:
        from src.DB_processing.image_processing import analyze_images
        from src.DB_processing.data_selection import Select_Data
        from src.DB_processing.dcm_parser import Parse_Dicom_Files
        from src.DB_processing.video_processing import ProcessVideoData
        from src.DB_processing.lesion_matching import Match_Lesions, Populate_Lesion_Types
        from src.ML_processing.lesion_detection import Locate_Lesions
        from src.ML_processing.inpaint_N2N import Inpaint_Dataset_N2N
        from src.ML_processing.caliper_coordinates import Locate_Calipers
        from src.DB_processing.validation_split import PerformSplit
        from src.DB_processing.split_bilateral import split_bilateral_cases_in_db
        from src.DB_processing.split_regions import split_regions_in_db
        from src.ML_processing.ultrasound_cropping import generate_crop_regions
        from src.ML_processing.caliper_pipeline.caliper_pipeline_run import run_caliper_pipeline
        from src.ML_processing.download_models import download_models
        
        lesion_pathology = f'{env}/data/lesion_pathology.csv'
        lesion_anon_file = f'{env}/data/lesion_anon_data.csv'
        birads_descriptions = f'{env}/data/birads.csv'
        birads_anon_file = f'{env}/data/birads_anon_data.csv'
        anon_file = f'{env}/data/anon_data.csv'
        key_output = f'{env}/encryption_key.pkl'
        BUCKET_PATH = f'{CONFIG["storage"]["download_path"]}/'
        
        if os.path.exists(DATABASE_LOCAL_PATH):
            print("WARNING: Previous database still exists, this can cause issues! Recommend rebuilding from scratch")
        
        print(f"Starting database processing for {BUCKET_PATH}...")
        download_models() # Download all models

        # Step 1: Encrypt IDs
        print("Step 1/5: Encrypting IDs...")
        key = encrypt_ids(DICOM_QUERY_PATH, anon_file, key_output)
        key = encrypt_ids(lesion_pathology, lesion_anon_file, key_output)
        key = encrypt_ids(birads_descriptions, birads_anon_file, key_output)
        
        # Step 2: Parse DICOM files
        Parse_Dicom_Files(CONFIG, anon_file, lesion_anon_file, birads_anon_file, BUCKET_PATH,
                          encryption_key=key, existing_db_path=args.existing_db)
        
        # Step 3: Run OCR
        print("Step 3/5: Processing image data...")
        analyze_images(CONFIG["DATABASE_DIR"])
        generate_crop_regions(CONFIG)
        run_caliper_pipeline()

        print("Splitting dual-region crops...")
        split_regions_in_db()

        # Step 4: Clean data
        print("Step 4/5: Cleaning image data...")
        Select_Data(CONFIG["DATABASE_DIR"])
        
        # Split bilateral cases into LEFT and RIGHT
        print("Splitting bilateral cases...")
        split_bilateral_cases_in_db()
        
        # Make inpainting optional
        if not args.skip_inpaint:
            print("Running inpainting (can be skipped with --skip-inpaint)...")
            Inpaint_Dataset_N2N( f'{CONFIG["DATABASE_DIR"]}/images/')
            
        Locate_Lesions(f'{CONFIG["DATABASE_DIR"]}/images/')
        #Locate_Calipers(f'{CONFIG["DATABASE_DIR"]}/images/')
        Match_Lesions()
        Populate_Lesion_Types()

        # Step 5: Process video
        print("Step 5/5: Processing video data...")
        ProcessVideoData(CONFIG["DATABASE_DIR"])
        
        # Step 6 validation split
        PerformSplit()
        
        #Upload Database
        if storage.is_gcp:
            blob = storage._bucket.blob(DATABASE_GCP_PATH.replace('//', '/').rstrip('/'))
            local_full_path = os.path.join(storage.windir, DATABASE_LOCAL_PATH) if storage.windir else DATABASE_LOCAL_PATH
            blob.upload_from_filename(local_full_path)
            print('Database uploaded')
    
    else:
        print("No action specified. Use --help for available options.")

if __name__ == "__main__":
    main()