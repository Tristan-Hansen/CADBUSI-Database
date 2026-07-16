"""
Surgery data handling for the combined dataset.

Prepares raw surgery records (rename, collapse multi-diagnosis rows, classify
the procedure reason), appends them to the combined dataset as their own rows,
and computes the per-row 'prior_intervention' JSON column describing prior
surgical procedures on the same breast.
"""

import json
import re
import pandas as pd
from tqdm import tqdm
from src.DB_processing.tools import append_audit


# Reason classification for prior interventions, checked in precedence order
# (first match wins across a procedure's diagnoses).
REASON_PATTERNS = [
    ('cancer', re.compile(
        r'MALIGNAN|CARCINOMA|\bCANCER\b|SARCOMA|PAGET', re.IGNORECASE)),
    ('reconstruction', re.compile(
        r'RECONSTRUCT|ABSENCE OF BREAST|MASTECTOMY|IMPLANT|PROSTHES|TISSUE EXPANDER',
        re.IGNORECASE)),
    ('injury/infection', re.compile(
        r'INFECTION|CELLULITIS|ABSCESS|MASTITIS|WOUND|DEHISCENCE|SEROMA|HEMATOMA'
        r'|HEMORRHAGE|CONTUSION|INJURY|TRAUMA|NECROSIS|LACERATION|RADIATION',
        re.IGNORECASE)),
    ('cosmetic', re.compile(
        r'COSMETIC|MACROMASTIA|GYNECOMASTIA|HYPERTROPHY BREAST|REDUNDANT SKIN'
        r'|EXCESSIVE AND REDUNDANT|ASYMMETRY|GENDER INCONGRUENCE|PLASTIC SURGERY'
        r'|BREAST REDUCTION|PTOSIS',
        re.IGNORECASE)),
]

# Personal/family history and genetic-risk diagnoses mention cancer terms but
# do not indicate active disease -- without this, reconstructions tagged with
# "Cancer Breast Personal History" would all classify as cancer.
CANCER_RISK_HISTORY_RE = re.compile(
    r'HISTORY|SUSCEPTIB|BRCA|PALB2|GENE MUTATION|CARRIER|ELEVATED RISK',
    re.IGNORECASE)


def classify_procedure_reason(diagnosis_text):
    """
    Classify a procedure's '; '-joined diagnosis text into one of:
    cancer, reconstruction, injury/infection, cosmetic, other.
    Each diagnosis is classified individually and the highest-precedence
    category across them wins.
    """
    if pd.isna(diagnosis_text) or not str(diagnosis_text).strip():
        return 'other'

    best_idx = len(REASON_PATTERNS)
    for diag in str(diagnosis_text).split(';'):
        diag = diag.strip()
        if not diag:
            continue
        for i, (category, pattern) in enumerate(REASON_PATTERNS):
            if not pattern.search(diag):
                continue
            if category == 'cancer' and CANCER_RISK_HISTORY_RE.search(diag):
                continue
            best_idx = min(best_idx, i)
            break

    return REASON_PATTERNS[best_idx][0] if best_idx < len(REASON_PATTERNS) else 'other'


def prepare_surgery_records(surgery_df):
    """
    Clean raw surgery rows into one row per procedure:
    - rename to pipeline column names (PATIENT_ID, DATE, procedure_*)
    - collapse multiple diagnosis rows per procedure into a single row with
      '; '-joined unique diagnosis texts
    - classify each procedure's reason (cancer / reconstruction /
      injury/infection / cosmetic / other) into 'procedure_reason'
    """
    surgery_records = surgery_df[[
        'PAT_PATIENT_CLINIC_NUMBER',
        'SURGCASE_SURGICAL_OPERATION_END_DTM',
        'SURGPROC_SURGICAL_CASE_ID',
        'SURGPROC_SURGICAL_PROCEDURE_DESCRIPTION',
        'SURGPROC_SURGICAL_PROCEDURE_BILATERAL_CODE',
        'DIAGCODE_DIAGNOSIS_DESCRIPTION',
    ]].copy()

    surgery_records = surgery_records.rename(columns={
        'PAT_PATIENT_CLINIC_NUMBER': 'PATIENT_ID',
        'SURGPROC_SURGICAL_PROCEDURE_DESCRIPTION': 'procedure_description',
        'SURGPROC_SURGICAL_PROCEDURE_BILATERAL_CODE': 'procedure_laterality',
        'DIAGCODE_DIAGNOSIS_DESCRIPTION': 'procedure_diagnosis',
    })
    surgery_records['PATIENT_ID'] = surgery_records['PATIENT_ID'].astype(str)
    surgery_records['DATE'] = pd.to_datetime(
        surgery_records['SURGCASE_SURGICAL_OPERATION_END_DTM'], errors='coerce'
    )
    surgery_records.drop(columns=['SURGCASE_SURGICAL_OPERATION_END_DTM'], inplace=True)
    surgery_records = surgery_records.drop_duplicates()

    # Collapse multiple diagnosis rows per procedure into a single row
    group_cols = ['PATIENT_ID', 'DATE', 'SURGPROC_SURGICAL_CASE_ID',
                  'procedure_description', 'procedure_laterality']
    surgery_records = (
        surgery_records
        .groupby(group_cols, dropna=False)['procedure_diagnosis']
        .apply(lambda s: '; '.join(pd.unique(s.dropna().astype(str))))
        .reset_index()
    )
    surgery_records.loc[surgery_records['procedure_diagnosis'] == '', 'procedure_diagnosis'] = None

    surgery_records['procedure_reason'] = (
        surgery_records['procedure_diagnosis'].apply(classify_procedure_reason)
    )

    return surgery_records


def add_surgery_records(final_df, surgery_df):
    """
    Append surgery records to the combined dataset as separate rows (same
    pattern as pathology rows): PATIENT_ID and DATE reuse the existing
    columns (DATE = SURGCASE_SURGICAL_OPERATION_END_DTM), plus the surgical
    case ID, procedure description, laterality, diagnosis and reason as new
    columns.
    """
    if surgery_df is None or surgery_df.empty:
        print("No surgery data to add")
        return final_df

    surgery_records = prepare_surgery_records(surgery_df)

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

        [{"type": "LUMPECTOMY BREAST", "laterality": "LEFT", "reason": "cancer", "age": "128 days"},
         {"type": "LUMPECTOMY BREAST", "laterality": "BILATERAL", "reason": "other", "age": "512 days"}]

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
        'procedure_reason',
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
                'laterality': code,
                'reason': surg['procedure_reason'],
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
