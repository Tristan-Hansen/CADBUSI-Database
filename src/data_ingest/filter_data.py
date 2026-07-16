import os
import json
import pandas as pd
from tqdm import tqdm
import re, sys

# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory
env = os.path.dirname(env)  # Go back one directory
sys.path.insert(0, env)
from src.DB_processing.tools import append_audit
from src.data_ingest.classification import determine_final_interpretation, audit_interpretations

def extract_cancer_type(text):
    if pd.isna(text):
        return "UNKNOWN"
    
    # Convert to uppercase for consistent matching
    text = str(text).upper()
    
    # Define specific cancer type patterns (ordered by specificity)
    cancer_patterns = [
        # Specific carcinoma types
        (r"INVASIVE\s+(?:GRADE\s+\d+.*?)?DUCTAL\s+CARCINOMA", "INVASIVE DUCTAL CARCINOMA"),
        (r"DUCTAL\s+CARCINOMA\s+IN\s+SITU", "DUCTAL CARCINOMA IN SITU"),
        (r"LOBULAR\s+CARCINOMA\s+IN\s+SITU", "LOBULAR CARCINOMA IN SITU"),
        (r"INVASIVE\s+MAMMARY\s+CARCINOMA", "INVASIVE MAMMARY CARCINOMA"),
        (r"\bDCIS\b", "DUCTAL CARCINOMA IN SITU"),
        (r"\bLCIS\b", "LOBULAR CARCINOMA IN SITU"),
        (r"ADENOID\s+CYSTIC\s+CARCINOMA", "ADENOID CYSTIC CARCINOMA"),
        
        # Other specific cancer types
        (r"ADENOCARCINOMA", "ADENOCARCINOMA"),
        (r"INFLAMMATORY\s+CARCINOMA", "INFLAMMATORY CARCINOMA"),
        (r"MUCINOUS\s+CARCINOMA", "MUCINOUS CARCINOMA"),
        (r"TUBULAR\s+CARCINOMA", "TUBULAR CARCINOMA"),
        (r"MEDULLARY\s+CARCINOMA", "MEDULLARY CARCINOMA"),
        (r"PAPILLARY\s+CARCINOMA", "PAPILLARY CARCINOMA"),
        
        # Metastatic findings
        (r"METASTATIC\s+CARCINOMA", "METASTATIC CARCINOMA"),
        (r"METASTATIC\s+ADENOCARCINOMA", "METASTATIC ADENOCARCINOMA"),
        (r"MICROMETASTASIS", "MICROMETASTASIS"),
        (r"ISOLATED\s+TUMOR\s+CELLS?", "ISOLATED TUMOR CELLS"),
        
        # High-risk lesions 
        (r"MULTIFOCAL\s+ATYPICAL\s+DUCTAL\s+HYPERPLASIA", "MULTIFOCAL ATYPICAL DUCTAL HYPERPLASIA"),
        (r"ATYPICAL\s+DUCTAL\s+HYPERPLASIA", "ATYPICAL DUCTAL HYPERPLASIA"),
        (r"ATYPICAL\s+LOBULAR\s+HYPERPLASIA", "ATYPICAL LOBULAR HYPERPLASIA"),
        (r"\bADH\b", "ATYPICAL DUCTAL HYPERPLASIA"),
        (r"\bALH\b", "ATYPICAL LOBULAR HYPERPLASIA"),
        
        # General carcinoma (catch-all)
        (r"(?:INVASIVE|INFILTRATIVE)\s+LOBULAR\s+CARCINOMA", "INVASIVE LOBULAR CARCINOMA"),
        (r"(?:INVASIVE|INFILTRATIVE)\s+CARCINOMA", "INVASIVE CARCINOMA"),
        (r"\bCARCINOMA\b", "CARCINOMA"),
        
        # Other malignancies
        (r"SARCOMA", "OTHER"),
        (r"LYMPHOMA", "OTHER"),
        (r"CARCINOSARCOMA", "OTHER"),
        (r"MELANOMA", "OTHER"),
        (r"SQUAMOUS\s+CELL\s+CARCINOMA", "OTHER"),
    ]
    
    # Check each pattern
    for pattern, cancer_type in cancer_patterns:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            # Get context around the match (100 characters before)
            start_pos = max(0, match.start() - 100)
            context_before = text[start_pos:match.start()]
            
            # Check if this is a negated finding
            negation_patterns = [
                r"NEGATIVE\s+FOR",
                r"NO\s+EVIDENCE\s+OF",
                r"FREE\s+OF",
                r"ABSENCE\s+OF",
                r"NO\s+",
                r"WITHOUT",
                r"RULED\s+OUT",
                r"EXCLUDE[SD]?",
            ]
            
            is_negated = False
            for neg_pattern in negation_patterns:
                if re.search(neg_pattern, context_before):
                    is_negated = True
                    break
            
            # If not negated, we found a positive cancer finding
            if not is_negated:
                return cancer_type
    
    # Check for benign/negative findings (combined)
    benign_patterns = [
        # Explicit negative findings
        r"NEGATIVE\s+FOR\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|DCIS|TUMOR|METASTATIC|METASTASIS)",
        r"NO\s+EVIDENCE\s+OF\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|DCIS|TUMOR|NEOPLASM|ATYPIA)",
        r"ABSENCE\s+OF\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|DCIS|TUMOR|NEOPLASM)",
        
        # Benign findings
        r"\bBENIGN\b",
        r"FIBROCYSTIC",
        r"FIBROADENOMA", 
        r"NORMAL\s+BREAST\s+TISSUE",
        r"SCLEROSING\s+ADENOSIS",
        r"USUAL\s+DUCTAL\s+HYPERPLASIA",
        r"\bPASH\b",
        r"NEGATIVE\s+MARGIN",
        r"NO\s+SIGNIFICANT\s+HISTOPATHOLOGY",
    ]
    
    for pattern in benign_patterns:
        if re.search(pattern, text):
            return "BENIGN"
    
    return "UNKNOWN"


def fill_pathology_accession_numbers(final_df):
    """
    Fill in accession numbers for rows with lesion_diag entries.
    For each patient with is_biopsy = T, find lesion_diag rows from 1 day before 
    the biopsy up to 6 months after and assign the accession number from the most recent 
    previous MODALITY = US row before that specific biopsy.
    Only assigns if lateralities match:
    - Study_Laterality BILATERAL matches any Pathology_Laterality
    - Study_Laterality LEFT/RIGHT must match exactly with Pathology_Laterality
    
    Additionally, copy SYNOPTIC_REPORT and final_diag data from pathology rows to other rows 
    with the same ACCESSION_NUMBER.
    """
    # Pre-filter all relevant records once (major speedup)
    biopsy_mask = final_df['is_biopsy'] == 'T'
    biopsy_records = final_df[biopsy_mask & pd.notna(final_df['DATE'])][
        ['PATIENT_ID', 'DATE']
    ].reset_index()
    
    lesion_diag_mask = (
        pd.notna(final_df['lesion_diag']) & 
        (pd.isna(final_df['ACCESSION_NUMBER']) | (final_df['ACCESSION_NUMBER'] == '')) &
        pd.notna(final_df['Pathology_Laterality']) &
        pd.notna(final_df['DATE'])
    )
    lesion_diag_records = final_df[lesion_diag_mask][
        ['PATIENT_ID', 'DATE', 'Pathology_Laterality']
    ].reset_index()
    
    us_records_mask = (
        #(final_df['MODALITY'] == 'US') &
        pd.notna(final_df['ACCESSION_NUMBER']) &
        (final_df['ACCESSION_NUMBER'] != '') &
        pd.notna(final_df['Study_Laterality']) &
        pd.notna(final_df['DATE'])
    )
    us_records = final_df[us_records_mask][
        ['PATIENT_ID', 'DATE', 'ACCESSION_NUMBER', 'Study_Laterality']
    ].reset_index()
    
    if biopsy_records.empty or lesion_diag_records.empty or us_records.empty:
        return final_df
    
    # Vectorized matching using merge operations
    accession_updates = {}
    
    # Cross join biopsies with lesion_diag records by patient
    biopsy_lesion_matches = pd.merge(
        biopsy_records, 
        lesion_diag_records, 
        on='PATIENT_ID', 
        suffixes=('_biopsy', '_lesion')
    )
    
    # Filter by date window vectorized
    one_day_before = biopsy_lesion_matches['DATE_biopsy'] - pd.Timedelta(days=1)
    six_months_later = biopsy_lesion_matches['DATE_biopsy'] + pd.Timedelta(days=180)
    
    date_filtered = biopsy_lesion_matches[
        (biopsy_lesion_matches['DATE_lesion'] >= one_day_before) &
        (biopsy_lesion_matches['DATE_lesion'] <= six_months_later)
    ]
    
    if not date_filtered.empty:
        # Fully vectorized approach: merge date_filtered with us_records
        # This creates all possible biopsy-lesion-US combinations
        print("Vectorized matching of pathology accession numbers...")
        biopsy_lesion_us = pd.merge(
            date_filtered,
            us_records,
            on='PATIENT_ID',
            suffixes=('', '_us')
        )

        # Filter: US date must be before biopsy date
        biopsy_lesion_us = biopsy_lesion_us[
            biopsy_lesion_us['DATE'] < biopsy_lesion_us['DATE_biopsy']
        ]

        if not biopsy_lesion_us.empty:
            # Check laterality matching vectorized
            laterality_match = (
                (biopsy_lesion_us['Study_Laterality'] == 'BILATERAL') |
                (biopsy_lesion_us['Study_Laterality'] == biopsy_lesion_us['Pathology_Laterality'])
            )
            biopsy_lesion_us = biopsy_lesion_us[laterality_match]

            if not biopsy_lesion_us.empty:
                # For each lesion, get the most recent US record (max DATE)
                # Sort by DATE to ensure we get the most recent
                biopsy_lesion_us = biopsy_lesion_us.sort_values('DATE')

                # Group by lesion index and take the last (most recent) US record
                most_recent_us = biopsy_lesion_us.groupby('index_lesion').last()

                # Create the updates dictionary
                accession_updates = most_recent_us['ACCESSION_NUMBER'].to_dict()
    
    # Batch update accession numbers (much faster than individual .at[] calls)
    if accession_updates:
        indices = list(accession_updates.keys())
        values = list(accession_updates.values())
        final_df.loc[indices, 'ACCESSION_NUMBER'] = values
    
    # Optimized SYNOPTIC_REPORT copying using groupby
    synoptic_mask = (
        pd.notna(final_df['SYNOPTIC_REPORT']) & 
        (final_df['SYNOPTIC_REPORT'] != '') &
        pd.notna(final_df['ACCESSION_NUMBER']) &
        (final_df['ACCESSION_NUMBER'] != '')
    )
    
    if synoptic_mask.any():
        # Get first SYNOPTIC_REPORT and final_diag for each accession number
        source_data = final_df[synoptic_mask].groupby('ACCESSION_NUMBER')[
            ['SYNOPTIC_REPORT', 'final_diag']
        ].first()
        
        # Find rows that need updates
        needs_update_mask = (
            pd.notna(final_df['ACCESSION_NUMBER']) &
            (final_df['ACCESSION_NUMBER'] != '') &
            (pd.isna(final_df['SYNOPTIC_REPORT']) | (final_df['SYNOPTIC_REPORT'] == ''))
        )
        
        if needs_update_mask.any():
            update_rows = final_df[needs_update_mask]
            
            # Vectorized merge to get the data
            merged_data = update_rows[['ACCESSION_NUMBER']].merge(
                source_data, 
                left_on='ACCESSION_NUMBER', 
                right_index=True, 
                how='left'
            )
            
            # Batch update using .loc
            valid_updates = pd.notna(merged_data['SYNOPTIC_REPORT'])
            if valid_updates.any():
                update_indices = update_rows[valid_updates].index
                final_df.loc[update_indices, 'SYNOPTIC_REPORT'] = merged_data.loc[valid_updates, 'SYNOPTIC_REPORT'].values
                final_df.loc[update_indices, 'final_diag'] = merged_data.loc[valid_updates, 'final_diag'].values
    
    # Logging
    filled_count = len(accession_updates)
    pathology_filled_count = (needs_update_mask & pd.notna(merged_data['SYNOPTIC_REPORT'])).sum() if 'merged_data' in locals() else 0
    
    append_audit("query_clean.pathology_accession_filled", filled_count)
    append_audit("query_clean.pathology_data_copied", pathology_filled_count)

    return final_df


def prepare_dataframes(rad_df, path_df):
    """Prepare and standardize dataframes for combining."""

    # Convert Patient_ID to string in both dataframes - use inplace for better performance
    rad_df['PATIENT_ID'] = rad_df['PATIENT_ID'].astype(str)
    path_df['PATIENT_ID'] = path_df['PATIENT_ID'].astype(str)

    # Convert date columns and rename in one step
    rad_df['DATE'] = pd.to_datetime(rad_df['RADIOLOGY_DTM'], errors='coerce')
    path_df['DATE'] = pd.to_datetime(path_df['SPECIMEN_RECEIVED_DTM'], errors='coerce')

    # Unify the encounter column. The warehouse names differ between fact tables
    # (FACT_RADIOLOGY.ENCOUNTER_NUMBER vs FACT_PATHOLOGY.ENCOUNTER_ID) but the
    # values share a domain. Renaming here gives downstream code a single column.
    if 'ENCOUNTER_NUMBER' in rad_df.columns:
        rad_df = rad_df.rename(columns={'ENCOUNTER_NUMBER': 'ENCOUNTER_ID'})

    # Drop columns more efficiently (in-place)
    columns_to_drop = ['RADIOLOGY_NARRATIVE', 'PROCEDURE_CODE_TEXT', 'SERVICE_RESULT_STATUS', 'RADIOLOGY_REPORT', 'RAD_SERVICE_RESULT_STATUS']
    rad_df = rad_df.drop(columns=columns_to_drop, errors='ignore')
    rad_df.drop('RADIOLOGY_DTM', axis=1, inplace=True)
    path_df.drop('SPECIMEN_RECEIVED_DTM', axis=1, inplace=True)

    return rad_df, path_df


def combine_dataframes(rad_df, path_df):
    """Combine radiology and pathology dataframes, keeping pathology on separate rows."""
    # Select only needed columns from path_df to reduce memory usage. ENCOUNTER_ID
    # is included so check_encounter_linked_path can join rad <-> path.
    needed_columns = ['PATIENT_ID', 'DATE', 'SPECIMEN_RESULT_DTM', 'Pathology_Laterality', 'final_diag', 'lesion_diag', 'SYNOPTIC_REPORT', 'path_interpretation', 'ENCOUNTER_ID']
    path_needed = path_df[needed_columns] if all(col in path_df.columns for col in needed_columns) else path_df
    
    # Create a copy of path_needed with the same columns as rad_df, plus any additional columns we need
    path_records_df = pd.DataFrame(columns=list(set(rad_df.columns) | set(path_needed.columns)))
    
    # Fill in values from path_needed
    for col in path_needed.columns:
        path_records_df[col] = path_needed[col]
    
    # Concatenate more efficiently with optimized settings
    final_df = pd.concat([rad_df, path_records_df], ignore_index=True, copy=False)
    
    return final_df

def add_prior_breast_cancer(final_df):
    """
    Add columns 'left_prior_breast_cancer' and 'right_prior_breast_cancer' to track
    if there was prior malignancy in each breast separately.
    Tracks malignancy based on left_diagnosis/right_diagnosis columns.

    Args:
        final_df: DataFrame containing columns:
                 PATIENT_ID, DATE, left_diagnosis, right_diagnosis

    Returns:
        DataFrame with added 'left_prior_breast_cancer' and 'right_prior_breast_cancer' columns (0/1)
    """

    print("Processing prior breast cancer by laterality...")

    # Convert DATE to datetime if it isn't already
    final_df['DATE'] = pd.to_datetime(final_df['DATE'], errors='coerce')

    # Initialize the new columns
    final_df['left_prior_breast_cancer'] = 0
    final_df['right_prior_breast_cancer'] = 0

    # Sort by patient and date for efficient processing
    df_sorted = final_df.sort_values(['PATIENT_ID', 'DATE']).copy()

    # Group by patient
    grouped = df_sorted.groupby('PATIENT_ID')

    # Process each patient
    for patient_id, group in grouped:
        # Track malignancy status separately for each breast
        left_malignancy_found = False
        right_malignancy_found = False

        # Iterate through records in chronological order
        for idx, row in group.iterrows():
            # Set the prior values for this row (before updating flags)
            final_df.loc[idx, 'left_prior_breast_cancer'] = 1 if left_malignancy_found else 0
            final_df.loc[idx, 'right_prior_breast_cancer'] = 1 if right_malignancy_found else 0

            # Now check if current row has malignancy and update flags
            # Check left_diagnosis for malignancy
            left_diag = str(row.get('left_diagnosis', '')).upper()
            if 'MALIGNANT' in left_diag:
                left_malignancy_found = True

            # Check right_diagnosis for malignancy
            right_diag = str(row.get('right_diagnosis', '')).upper()
            if 'MALIGNANT' in right_diag:
                right_malignancy_found = True

    # Count statistics
    total_with_left_prior_cancer = (final_df['left_prior_breast_cancer'] == 1).sum()
    total_with_right_prior_cancer = (final_df['right_prior_breast_cancer'] == 1).sum()

    print(f"Records with left prior breast cancer: {total_with_left_prior_cancer}")
    print(f"Records with right prior breast cancer: {total_with_right_prior_cancer}")
    append_audit("query_clean.records_with_left_prior_cancer", int(total_with_left_prior_cancer))
    append_audit("query_clean.records_with_right_prior_cancer", int(total_with_right_prior_cancer))

    return final_df

def create_pathology_subset_csv(final_df):
    """
    Create a separate CSV containing only rows that have all pathology-related fields
    """

    # Define the required columns
    required_columns = ['PATIENT_ID', 'ACCESSION_NUMBER', 'DATE', 'lesion_diag', 'SYNOPTIC_REPORT', 'Pathology_Laterality', 'path_interpretation']

    # Select only the required columns
    pathology_subset = final_df[required_columns].copy()

    # Remove rows with any null/NA values in any of the columns
    pathology_subset = pathology_subset.dropna()

    # Also remove rows with empty strings
    for col in required_columns:
        pathology_subset = pathology_subset[pathology_subset[col] != '']

    return pathology_subset
    
def handle_duplicate_accessions(df):
    """
    Handle duplicate accessions based on modality with specific business rules:
    1. If duplicates include modality = US row, remove all other modality rows for that accession
    2. If duplicates include 2+ rows with modality = US, remove entire accession
    3. If duplicates include 0 modality = US, collapse into one row with comma-separated modalities
    """
    print("\nHandling duplicate accessions based on modality...")

    # Find all duplicate accessions
    duplicate_mask = df.duplicated(subset=['ACCESSION_NUMBER'], keep=False)
    duplicate_accessions = df[duplicate_mask]['ACCESSION_NUMBER'].unique()

    if len(duplicate_accessions) == 0:
        print("No duplicate accessions found")
        return df

    print(f"Found {len(duplicate_accessions)} accessions with duplicates - processing with vectorized operations...")

    # Separate duplicates from non-duplicates for faster processing
    duplicates_df = df[duplicate_mask].copy()
    non_duplicates_df = df[~duplicate_mask].copy()

    # Add is_us flag for faster processing
    duplicates_df['is_us'] = duplicates_df['MODALITY'] == 'US'

    # Group by accession and count US rows (vectorized)
    grouped = duplicates_df.groupby('ACCESSION_NUMBER')
    us_counts = grouped['is_us'].sum()

    # Classify each accession into one of the three cases
    case1_accessions = us_counts[us_counts == 1].index  # Exactly 1 US
    case2_accessions = us_counts[us_counts >= 2].index  # 2+ US
    case3_accessions = us_counts[us_counts == 0].index  # 0 US

    rows_to_keep = []

    # Case 1: Keep only US rows (vectorized)
    if len(case1_accessions) > 0:
        case1_mask = duplicates_df['ACCESSION_NUMBER'].isin(case1_accessions)
        case1_us_mask = case1_mask & duplicates_df['is_us']
        case1_keep = duplicates_df[case1_us_mask].copy()
        case1_keep.drop(columns=['is_us'], inplace=True)
        rows_to_keep.append(case1_keep)
        us_only_kept_count = (case1_mask & ~duplicates_df['is_us']).sum()
    else:
        us_only_kept_count = 0

    # Case 2: Remove all rows (count them but don't keep any)
    if len(case2_accessions) > 0:
        case2_mask = duplicates_df['ACCESSION_NUMBER'].isin(case2_accessions)
        multiple_us_removed_count = case2_mask.sum()
    else:
        multiple_us_removed_count = 0

    # Case 3: Collapse into one row with combined modalities (vectorized)
    if len(case3_accessions) > 0:
        case3_df = duplicates_df[duplicates_df['ACCESSION_NUMBER'].isin(case3_accessions)]

        # Get first row of each accession
        first_rows = case3_df.groupby('ACCESSION_NUMBER').first()

        # Get combined modalities for each accession
        combined_modalities = case3_df.groupby('ACCESSION_NUMBER')['MODALITY'].apply(
            lambda x: ', '.join(sorted(x.dropna().unique()))
        )

        # Update the modality column in first_rows
        first_rows['MODALITY'] = combined_modalities
        first_rows.drop(columns=['is_us'], inplace=True)

        rows_to_keep.append(first_rows.reset_index(drop=True))

        # Count removed rows (total case3 rows minus one kept per accession)
        non_us_collapsed_count = len(case3_df) - len(case3_accessions)
    else:
        non_us_collapsed_count = 0

    # Combine all rows to keep
    if rows_to_keep:
        kept_duplicates = pd.concat(rows_to_keep, ignore_index=True)
        df_filtered = pd.concat([non_duplicates_df, kept_duplicates], ignore_index=True)
    else:
        df_filtered = non_duplicates_df

    # Calculate total rows removed
    total_removed = us_only_kept_count + multiple_us_removed_count + non_us_collapsed_count

    # Print metrics
    print(f"\nDuplicate handling metrics:")
    print(f"  Case 1 (US only kept): Removed {us_only_kept_count} non-US rows from accessions with 1 US row")
    print(f"  Case 2 (Multiple US): Removed {multiple_us_removed_count} total rows from accessions with 2+ US rows")
    print(f"  Case 3 (No US collapsed): Collapsed {non_us_collapsed_count} duplicate rows into combined modality entries")
    print(f"  Total rows removed: {total_removed}")

    # Audit logging
    append_audit("query_clean.duplicate_us_only_kept_removed", us_only_kept_count)
    append_audit("query_clean.duplicate_multiple_us_removed", multiple_us_removed_count)
    append_audit("query_clean.duplicate_non_us_collapsed", non_us_collapsed_count)
    append_audit("query_clean.rad_duplicates_removed", total_removed)

    return df_filtered

def audit_pathology_dates(df):
    """
    Calculate days from biopsy to pathology SPECIMEN_RESULT_DTM for each patient and record in the audit.
    Only considers cases where the DATE vs DATE difference is within 2 weeks.

    Also audits day distance from biopsy to the most recent row before that biopsy.
    """
    # Ensure DATE column is in datetime format
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['SPECIMEN_RESULT_DTM'] = pd.to_datetime(df['SPECIMEN_RESULT_DTM'], errors='coerce')
    
    days_differences = []
    exam_to_biopsy_differences = []
    
    # Get total number of patients for progress bar
    unique_patients = df['PATIENT_ID'].nunique()
    
    # Use groupby with tqdm progress bar
    for patient_id, patient_group in tqdm(df.groupby('PATIENT_ID'), 
                                         total=unique_patients, 
                                         desc="Auditing pathology dates"):
        # Sort this patient's records (matches original)
        patient_records = patient_group.sort_values('DATE')
        
        # Find biopsy and pathology records (matches original logic exactly)
        biopsy_records = patient_records[patient_records['is_biopsy'] == 'T']
        pathology_records = patient_records[pd.notna(patient_records['path_interpretation'])]
        
        # Audit 1: Biopsy to pathology differences (existing logic)
        if not biopsy_records.empty and not pathology_records.empty:
            for biopsy_idx, biopsy in biopsy_records.iterrows():
                biopsy_date = biopsy['DATE']
                
                if pd.isna(biopsy_date):
                    continue
                
                # Vectorize the inner loop for pathology records
                valid_pathology = pathology_records[
                    pd.notna(pathology_records['DATE']) & 
                    pd.notna(pathology_records['SPECIMEN_RESULT_DTM'])
                ]
                
                if not valid_pathology.empty:
                    date_diffs = (valid_pathology['DATE'] - biopsy_date).dt.days
                    # Keep original logic: skip if within 2 weeks
                    outside_window = ~((date_diffs >= 0) & (date_diffs <= 14))
                    
                    if outside_window.any():
                        result_diffs = (valid_pathology.loc[outside_window, 'SPECIMEN_RESULT_DTM'] - biopsy_date).dt.days
                        days_differences.extend(result_diffs.tolist())
        
        # Audit 2: Most recent exam to biopsy differences (new logic)
        if not biopsy_records.empty:
            for biopsy_idx, biopsy in biopsy_records.iterrows():
                biopsy_date = biopsy['DATE']
                
                if pd.isna(biopsy_date):
                    continue
                
                previous_records = patient_records[
                    (patient_records['DATE'] < biopsy_date) & 
                    (pd.notna(patient_records['DATE']))
                ]
                
                if not previous_records.empty:
                    most_recent_exam = previous_records.iloc[-1]
                    most_recent_date = most_recent_exam['DATE']
                    exam_diff = (biopsy_date - most_recent_date).days
                    exam_to_biopsy_differences.append(exam_diff)
    
    append_audit("query_clean.pathology_date_from_biopsy", days_differences)
    append_audit("query_clean.exam_date_from_biospy", exam_to_biopsy_differences)
    
def add_surgery_records(final_df, surgery_df):
    """
    Append surgery records to the combined dataset as separate rows (same
    pattern as pathology rows): PATIENT_ID and DATE reuse the existing
    columns (DATE = SURGCASE_SURGICAL_OPERATION_END_DTM), plus the surgical
    case ID and procedure description as new columns.
    """
    if surgery_df is None or surgery_df.empty:
        print("No surgery data to add")
        return final_df

    surgery_records = surgery_df[[
        'PAT_PATIENT_CLINIC_NUMBER',
        'SURGCASE_SURGICAL_OPERATION_END_DTM',
        'SURGPROC_SURGICAL_CASE_ID',
        'SURGPROC_SURGICAL_PROCEDURE_DESCRIPTION',
        'SURGPROC_SURGICAL_PROCEDURE_BILATERAL_CODE',
    ]].copy()

    surgery_records = surgery_records.rename(columns={
        'PAT_PATIENT_CLINIC_NUMBER': 'PATIENT_ID',
        'SURGPROC_SURGICAL_PROCEDURE_DESCRIPTION': 'procedure_description',
        'SURGPROC_SURGICAL_PROCEDURE_BILATERAL_CODE': 'procedure_laterality',
    })
    surgery_records['PATIENT_ID'] = surgery_records['PATIENT_ID'].astype(str)
    surgery_records['DATE'] = pd.to_datetime(
        surgery_records['SURGCASE_SURGICAL_OPERATION_END_DTM'], errors='coerce'
    )
    surgery_records.drop(columns=['SURGCASE_SURGICAL_OPERATION_END_DTM'], inplace=True)
    surgery_records = surgery_records.drop_duplicates()

    combined = pd.concat([final_df, surgery_records], ignore_index=True)
    print(f"Added {len(surgery_records)} surgery rows to the combined dataset")
    append_audit("query_clean.surgery_rows_added", len(surgery_records))

    return combined


def add_prior_interventions(debug_df):
    """
    Add a 'prior_intervention' column to the combined dataset (after surgery
    rows have been appended). For each row, the column holds a JSON string
    listing every surgical procedure performed on the same breast BEFORE that
    row's DATE, e.g.:

        [{"type": "LUMPECTOMY BREAST", "age": "128 days"},
         {"type": "LUMPECTOMY BREAST", "age": "512 days"}]

    Laterality matching (row side vs procedure_laterality):
    - Bilateral surgeries match any row laterality
    - Left/Right surgeries match rows with the same side, or BILATERAL rows
    - Surgeries with unknown codes (e.g. 'Anterior', blank) are skipped since
      they can't be attributed to a breast

    The row's laterality comes from Study_Laterality, falling back to
    Pathology_Laterality, then to the surgery bilateral code for surgery rows.
    Rows with no laterality or no prior matching surgeries are left empty.
    """
    if 'procedure_description' not in debug_df.columns:
        print("No surgery columns present -- skipping prior interventions")
        debug_df['prior_intervention'] = None
        return debug_df

    print("\nProcessing prior interventions...")

    debug_df['DATE'] = pd.to_datetime(debug_df['DATE'], errors='coerce')
    debug_df['prior_intervention'] = None

    surgery_mask = (
        pd.notna(debug_df['procedure_description']) &
        pd.notna(debug_df['DATE'])
    )
    if not surgery_mask.any():
        print("No surgery rows found -- skipping prior interventions")
        return debug_df

    surgeries = debug_df.loc[surgery_mask, [
        'PATIENT_ID', 'DATE',
        'procedure_description',
        'procedure_laterality',
    ]].sort_values('DATE')

    surgeries_by_patient = {pid: group for pid, group in surgeries.groupby('PATIENT_ID')}

    def get_row_laterality(row):
        for col in ('Study_Laterality', 'Pathology_Laterality',
                    'procedure_laterality'):
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                return str(val).upper().strip()
        return None

    # Only rows dated after a surgery for the same patient can have priors
    candidate_mask = (
        pd.notna(debug_df['DATE']) &
        debug_df['PATIENT_ID'].isin(surgeries_by_patient.keys())
    )

    rows_with_priors = 0
    for idx, row in tqdm(debug_df[candidate_mask].iterrows(),
                         total=int(candidate_mask.sum()),
                         desc="Prior interventions"):
        row_lat = get_row_laterality(row)
        if row_lat not in ('LEFT', 'RIGHT', 'BILATERAL'):
            continue

        patient_surgeries = surgeries_by_patient[row['PATIENT_ID']]
        prior = patient_surgeries[patient_surgeries['DATE'] < row['DATE']]
        if prior.empty:
            continue

        entries = []
        for _, surg in prior.iterrows():
            code = surg['procedure_laterality']
            code = str(code).upper().strip() if pd.notna(code) else ''

            side_match = (
                code == 'BILATERAL' or
                code == row_lat or
                (row_lat == 'BILATERAL' and code in ('LEFT', 'RIGHT'))
            )
            if not side_match:
                continue

            age_days = (row['DATE'] - surg['DATE']).days
            entries.append({
                'type': surg['procedure_description'],
                'age': f"{age_days} days",
            })

        if entries:
            # Most recent procedure first
            entries.sort(key=lambda e: int(e['age'].split()[0]))
            debug_df.at[idx, 'prior_intervention'] = json.dumps(entries)
            rows_with_priors += 1

    print(f"Rows with prior interventions: {rows_with_priors}")
    append_audit("query_clean.rows_with_prior_intervention", rows_with_priors)

    return debug_df


def create_final_dataset(rad_df, path_df, output_path, surgery_df=None):
    """Main function to create the final dataset with pathology records on separate rows."""
    print("\nLinking data:")

    # Prepare dataframes
    rad_df, path_df = prepare_dataframes(rad_df, path_df)
    path_df_length = len(path_df)

    # Combine dataframes
    final_df = combine_dataframes(rad_df, path_df)

    final_df = fill_pathology_accession_numbers(final_df)

    # Determine final interpretation
    final_df = determine_final_interpretation(final_df)

    # Add prior breast cancer column (tracks laterality)
    final_df = add_prior_breast_cancer(final_df)

    # CREATE THE LESION PATHOLOGY SUBSET CSV
    pathology_subset = create_pathology_subset_csv(final_df)

    # Surgery rows go into the debug CSV only -- downstream filtering
    # (endpoint_data.csv) continues from final_df without them, and they
    # would be dropped by the US/diagnosis filter anyway.
    debug_df = add_surgery_records(final_df, surgery_df)

    # Add prior_intervention JSON column (prior surgeries on the same breast)
    debug_df = add_prior_interventions(debug_df)

    # Save to CSV
    debug_df.to_csv(f'{output_path}/combined_dataset_debug.csv', index=False)

    audit_pathology_dates(final_df)

    # Keep US rows OR rows with diagnosis (left_diagnosis or right_diagnosis)
    initial_count = len(final_df)
    is_us = final_df['MODALITY'].str.contains('US', na=False, case=False)
    has_diagnosis = final_df['left_diagnosis'].notna() | final_df['right_diagnosis'].notna()

    final_df_filtered = final_df[is_us | has_diagnosis].copy()

    # Clear ENDPOINT_ADDRESS for non-US rows
    non_us_mask = ~final_df_filtered['MODALITY'].str.contains('US', na=False, case=False)
    final_df_filtered.loc[non_us_mask, 'ENDPOINT_ADDRESS'] = None

    filtered_count = initial_count - len(final_df_filtered)
    append_audit("query_clean.rad_non_US_no_diagnosis_removed", filtered_count - path_df_length)

    us_count = is_us.sum()
    non_us_with_diagnosis = len(final_df_filtered[non_us_mask])
    print(f"Kept {us_count} US rows and {non_us_with_diagnosis} non-US rows with diagnosis")
    append_audit("query_clean.us_rows", us_count)
    append_audit("query_clean.non_us_with_diagnosis", non_us_with_diagnosis)

    # Handle duplicate accessions based on modality
    final_df_filtered = handle_duplicate_accessions(final_df_filtered)

    # Remove US rows with empty ENDPOINT_ADDRESS (non-US rows already have cleared endpoints, so skip those)
    us_rows = final_df_filtered['MODALITY'].str.contains('US', na=False, case=False)
    empty_endpoint_in_us = us_rows & final_df_filtered['ENDPOINT_ADDRESS'].isna()
    empty_endpoint_count = sum(empty_endpoint_in_us)
    final_df_filtered = final_df_filtered[~empty_endpoint_in_us]
    append_audit("query_clean.rad_us_missing_address_removed", empty_endpoint_count)
    
    # Count total interpretations
    audit_interpretations(final_df_filtered)

    # Remove rows where both left_diagnosis and right_diagnosis are empty AND is_biopsy is True
    both_empty = final_df_filtered['left_diagnosis'].isna() & final_df_filtered['right_diagnosis'].isna()
    is_biopsy = final_df_filtered['is_biopsy'] == 'T'
    remove_condition = both_empty & is_biopsy

    empty_interpretation_count = sum(remove_condition)
    final_df_filtered = final_df_filtered[~remove_condition]
    append_audit("query_clean.rad_missing_final_interp_biopsy", empty_interpretation_count)
    

    # Extract STUDY_ID from ENDPOINT_ADDRESS
    final_df_filtered['STUDY_ID'] = final_df_filtered['ENDPOINT_ADDRESS'].apply(
        lambda url: url.split('/')[-1] if pd.notna(url) else None
    )

    # Clean lesion pathology
    pathology_subset = pathology_subset[pathology_subset['ACCESSION_NUMBER'].isin(final_df_filtered['ACCESSION_NUMBER'])]
    pathology_subset['cancer_type'] = pathology_subset['lesion_diag'].apply(extract_cancer_type)
    
    # Save the filtered dataset. ENCOUNTER_ID is carried through earlier steps
    # (it powers check_encounter_linked_path and is visible in
    # combined_dataset_debug.csv) but should not leak into endpoint_data.csv --
    # drop it right at the boundary.
    final_df_filtered = final_df_filtered.drop(columns=['ENCOUNTER_ID'], errors='ignore')
    final_df_filtered.to_csv(f'{env}/data/endpoint_data.csv', index=False)
    pathology_subset.to_csv(f'{output_path}/lesion_pathology.csv', index=False)

    # Print statistics
    print(f"Data ready with {len(final_df_filtered)} accessions")
    append_audit("query_clean.final_case_count", len(final_df_filtered))
    
    return final_df_filtered



if __name__ == "__main__":
    # Load the parsed radiology and pathology data
    try:
        output_path = os.path.join(env, "data")
        rad_file_path = f'{output_path}/parsed_radiology.csv'
        path_file_path = f'{output_path}/parsed_pathology.csv'
        
        rad_df = pd.read_csv(rad_file_path)
        print(f"Loaded radiology data with {len(rad_df)} records")

        path_df = pd.read_csv(path_file_path)
        print(f"Loaded pathology data with {len(path_df)} records")

        surgery_file_path = f'{output_path}/raw_surgery.csv'
        if os.path.exists(surgery_file_path):
            surgery_df = pd.read_csv(surgery_file_path)
            print(f"Loaded surgery data with {len(surgery_df)} records")
        else:
            surgery_df = None
            print("No raw_surgery.csv found -- skipping surgery rows")

        # Call the create_final_dataset function
        create_final_dataset(rad_df, path_df, output_path, surgery_df=surgery_df)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you've run the parsing scripts to create the parsed CSV files first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")