from google.cloud import bigquery
import time
import os
import pandas as pd
from tqdm import tqdm
from src.DB_processing.tools import append_audit
# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory
env = os.path.dirname(env)  # Go back one directory

def get_radiology_data(limit=None):
    """
    Get radiology data for breast imaging studies, ensuring all relevant accessions
    for each patient are included, but still filtering for breast imaging only
    
    Args:
        limit (int, optional): Limit the number of patients returned
    
    Returns:
        pandas.DataFrame: Query results as a dataframe
    """
    
    print("Initializing BigQuery client for radiology data...")
    client = bigquery.Client()
    
    # Build the limit clause for the CTE
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    
    query = f"""
    -- Deduplicate Patient table first, preferring non-null US_CORE_BIRTHSEX
    WITH deduplicated_patients AS (
      SELECT 
        CLINIC_NUMBER,
        US_CORE_BIRTHSEX,
        ROW_NUMBER() OVER (
          PARTITION BY CLINIC_NUMBER 
          ORDER BY CASE WHEN US_CORE_BIRTHSEX IS NOT NULL THEN 0 ELSE 1 END
        ) AS rn
      FROM `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.Patient`
    ),
    -- First identify ALL patients with US modality at series level (excluding males)
    us_imaging_patients AS (
      SELECT DISTINCT PAT_PATIENT.CLINIC_NUMBER AS PATIENT_ID
      FROM `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.ImagingStudy` imaging
      INNER JOIN `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.ImagingStudySeries` imaging_series
        ON (imaging.id = imaging_series.imaging_study_id)
      INNER JOIN deduplicated_patients PAT_PATIENT 
        ON (imaging.clinic_number = PAT_PATIENT.CLINIC_NUMBER AND PAT_PATIENT.rn = 1)
      WHERE imaging_series.SERIES_MODALITY_CODE = 'US'
        AND PAT_PATIENT.US_CORE_BIRTHSEX != 'M'
      {limit_clause}
    )
    -- Then get radiology data for these patients with BREAST test descriptions
    SELECT DISTINCT
      PAT_PATIENT.CLINIC_NUMBER AS PATIENT_ID,
      RAD_FACT_RADIOLOGY.ACCESSION_NBR AS ACCESSION_NUMBER,
      imaging_studies.DESCRIPTION,
      imaging_studies.PROCEDURE_CODE_TEXT,
      ENDPOINT.ADDRESS AS ENDPOINT_ADDRESS,
      PAT_PATIENT.US_CORE_BIRTHSEX,
      IMAGINGSTUDYSERIES.SERIES_MODALITY_CODE AS MODALITY,
      RAD_FACT_RADIOLOGY.RADIOLOGY_NARRATIVE,
      RAD_FACT_RADIOLOGY.RADIOLOGY_REPORT,
      RAD_FACT_RADIOLOGY.SERVICE_RESULT_STATUS,
      RAD_FACT_RADIOLOGY.RADIOLOGY_DTM,
      RAD_FACT_RADIOLOGY.RADIOLOGY_REVIEW_DTM,
      RAD_FACT_RADIOLOGY.ENCOUNTER_NUMBER,
      RAD_FACT_RADIOLOGY.SITE_NAME AS SITE,
      RADTEST_DIM_RADIOLOGY_TEST_NAME.RADIOLOGY_TEST_DESCRIPTION AS TEST_DESCRIPTION,
      -- Added demographic fields
      PAT_DIM_PATIENT.PATIENT_ETHNICITY_NAME AS ETHNICITY,
      PAT_DIM_PATIENT.PATIENT_DEATH_DATE AS DEATH_DATE,
      PAT_DIM_PATIENT.PATIENT_PRIMARY_ZIPCODE AS ZIPCODE,
      PAT_DIM_PATIENT.PATIENT_RACE_NAME AS RACE,
      DATE_DIFF(EXTRACT(DATE FROM RAD_FACT_RADIOLOGY.RADIOLOGY_DTM), PAT_DIM_PATIENT.PATIENT_BIRTH_DATE, YEAR) - 
        IF(EXTRACT(MONTH FROM PAT_DIM_PATIENT.PATIENT_BIRTH_DATE)*100 + EXTRACT(DAY FROM PAT_DIM_PATIENT.PATIENT_BIRTH_DATE) > 
          EXTRACT(MONTH FROM RAD_FACT_RADIOLOGY.RADIOLOGY_DTM)*100 + EXTRACT(DAY FROM RAD_FACT_RADIOLOGY.RADIOLOGY_DTM),1,0) AS AGE_AT_EVENT,
      PAT_DIM_PATIENT.PATIENT_BIRTH_DATE AS BIRTH_DATE,
      -- Added breast-specific pathology fields
      breast_results.A1_PATHOLOGY_TXT,
      breast_results.A1_PATHOLOGY_CATEGORY_DESC,
      breast_results.A2_PATHOLGY_TXT,
      breast_results.A2_PATHOLOGY_CATEGORY_DESC,
      -- Added location fields
      rad_exam.LOCATION_ID,
      location.name AS LOCATION_NAME,
      location.address_city AS LOCATION_CITY,
      location.address_state AS LOCATION_STATE,
      location.address_zip AS LOCATION_ZIP,
      location.address_line AS LOCATION_ADDRESS,
      location.physical_type_text AS LOCATION_TYPE,
      dim_location.LOCATION_DESCRIPTION,
      RES_PVDR_DIM_HEALTHCARE_PROVIDER.PROVIDER_EMPLOYEE_ID AS RESULT_EMPLOYEE_ID
    FROM us_imaging_patients
    INNER JOIN 
      deduplicated_patients PAT_PATIENT 
      ON us_imaging_patients.PATIENT_ID = PAT_PATIENT.CLINIC_NUMBER AND PAT_PATIENT.rn = 1
    INNER JOIN 
      `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_PATIENT` PAT_DIM_PATIENT
      ON PAT_PATIENT.CLINIC_NUMBER = PAT_DIM_PATIENT.PATIENT_CLINIC_NUMBER
    INNER JOIN 
      `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_RADIOLOGY` RAD_FACT_RADIOLOGY 
      ON PAT_DIM_PATIENT.PATIENT_DK = RAD_FACT_RADIOLOGY.PATIENT_DK
    LEFT JOIN 
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.ImagingStudy` imaging_studies
      ON (RAD_FACT_RADIOLOGY.ACCESSION_NBR = imaging_studies.ACCESSION_IDENTIFIER_VALUE)
    LEFT JOIN
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.ImagingStudySeries` IMAGINGSTUDYSERIES
      ON (imaging_studies.id = IMAGINGSTUDYSERIES.imaging_study_id)
    LEFT JOIN 
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.Endpoint` ENDPOINT 
      ON (imaging_studies.gcp_endpoint_id = ENDPOINT.id)
    INNER JOIN
      `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_RADIOLOGY_TEST_NAME` RADTEST_DIM_RADIOLOGY_TEST_NAME
      ON (RAD_FACT_RADIOLOGY.RADIOLOGY_TEST_NAME_DK = RADTEST_DIM_RADIOLOGY_TEST_NAME.RADIOLOGY_TEST_NAME_DK)
    LEFT JOIN
      `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_HEALTHCARE_PROVIDER` RES_PVDR_DIM_HEALTHCARE_PROVIDER
      ON (RAD_FACT_RADIOLOGY.RESULT_PROVIDER_DK = RES_PVDR_DIM_HEALTHCARE_PROVIDER.PROVIDER_DK)
    LEFT JOIN
      `ml-mps-adl-intudp-phi-p-d5cb.phi_rad_udpwh_us_p.DIM_RADIOLOGY_EXAM` rad_exam
      ON (RAD_FACT_RADIOLOGY.ACCESSION_NBR = rad_exam.ACCESSION_NBR_ID)
    LEFT JOIN 
      `ml-mps-adl-intudp-phi-p-d5cb.phi_rad_udpwh_us_p.DIM_RADIOLOGY_EXAM_RESULTS_BREAST` breast_results
      ON (rad_exam.RADIOLOGY_EXAM_DK = breast_results.RADIOLOGY_EXAM_DK)
    LEFT JOIN
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.Location` location
      ON (rad_exam.LOCATION_ID = location.identifier_value)
    LEFT JOIN
      `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_LOCATION` dim_location
      ON (RAD_FACT_RADIOLOGY.LOCATION_DK = dim_location.LOCATION_DK)
    WHERE RADTEST_DIM_RADIOLOGY_TEST_NAME.RADIOLOGY_TEST_DESCRIPTION LIKE '%BREAST%'
      AND imaging_studies.DESCRIPTION LIKE '%BREAST%'
      AND (IMAGINGSTUDYSERIES.SERIES_MODALITY_CODE IS NULL OR IMAGINGSTUDYSERIES.SERIES_MODALITY_CODE NOT IN ('SR', 'PR', 'KO'))
    """

    query_start_time = time.time()
    print("Executing radiology query...")
    df = client.query(query).to_dataframe()
    query_end_time = time.time()
    query_duration = query_end_time - query_start_time

    print(f"Radiology query complete. Retrieved {len(df)} rows for {len(df['PATIENT_ID'].unique())} patients in {query_duration:.2f} seconds.")

    return df

def get_pathology_data(patient_ids, batch_size=1000):
    """
    Get pathology data for specific patient IDs, processing in batches
    
    Args:
        patient_ids (list): List of patient IDs to query
        batch_size (int): Number of patients to process in each batch
    
    Returns:
        pandas.DataFrame: Query results as a dataframe
    """
    start_time = time.time()
    print("Initializing BigQuery client for pathology data...")
    client = bigquery.Client()
    
    # Process in batches
    all_results = []
    total_patients = len(patient_ids)
    total_batches = (total_patients + batch_size - 1) // batch_size
    
    # Create a tqdm progress bar
    for i in tqdm(range(0, total_patients, batch_size), total=total_batches):
        batch = patient_ids[i:i+batch_size]

        # Format IDs appropriately
        if batch and all(str(id).isdigit() for id in batch):
            ids_str = ', '.join([str(id) for id in batch])
        else:
            ids_str = ', '.join([f"'{id}'" for id in batch])
        
        query = f"""
        SELECT 
          PAT_DIM_PATIENT.PATIENT_CLINIC_NUMBER AS PATIENT_ID,
          PATH_FACT_PATHOLOGY.SPECIMEN_NOTE,
          PATH_FACT_PATHOLOGY.SPECIMEN_UPDATE_DTM,
          PATH_FACT_PATHOLOGY.SPECIMEN_RESULT_DTM,
          PATH_FACT_PATHOLOGY.SPECIMEN_RECEIVED_DTM,
          PATH_FACT_PATHOLOGY.SPECIMEN_SERVICE_DESCRIPTION,
          PATH_FACT_PATHOLOGY.ENCOUNTER_ID,
          DIAGCODE_DIM_DIAGNOSIS_CODE.DIAGNOSIS_NAME,
          PATH_FACT_PATHOLOGY.PATHOLOGY_COUNT,
          PATH_FACT_PATHOLOGY.SPECIMEN_COMMENT,
          PATH_FACT_PATHOLOGY.SPECIMEN_ACCESSION_NUMBER,
          SPECDET.PART_DESCRIPTION,
          SPECPARTYP.SPECIMEN_PART_TYPE_CODE,
          SPECPARTYP.SPECIMEN_PART_TYPE_NAME
        FROM `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_PATHOLOGY` PATH_FACT_PATHOLOGY
        INNER JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_PATIENT` PAT_DIM_PATIENT
          ON (PATH_FACT_PATHOLOGY.PATIENT_DK = PAT_DIM_PATIENT.PATIENT_DK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_PATHOLOGY_DIAGNOSIS_CODE_BRIDGE` PATHDIAG
          ON (PATH_FACT_PATHOLOGY.PATHOLOGY_FPK = PATHDIAG.PATHOLOGY_FPK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_DIAGNOSIS_CODE` DIAGCODE_DIM_DIAGNOSIS_CODE
          ON (PATHDIAG.DIAGNOSIS_CODE_DK = DIAGCODE_DIM_DIAGNOSIS_CODE.DIAGNOSIS_CODE_DK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_PATHOLOGY_SPECIMEN_DETAIL` SPECDET
          ON (PATH_FACT_PATHOLOGY.PATHOLOGY_FPK = SPECDET.PATHOLOGY_FPK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_SPECIMEN_PART_TYPE` SPECPARTYP
          ON (SPECDET.SPECIMEN_PART_TYPE_DK = SPECPARTYP.SPECIMEN_PART_TYPE_DK)
        WHERE PAT_DIM_PATIENT.PATIENT_CLINIC_NUMBER IN ({ids_str})
        AND LOWER(SPECPARTYP.SPECIMEN_PART_TYPE_NAME) LIKE '%breast%'
        """
        
        batch_df = client.query(query).to_dataframe()
        all_results.append(batch_df)
    
    # Combine all batch results
    if all_results:
        df = pd.concat(all_results, ignore_index=True)
    else:
        df = pd.DataFrame()
    
    total_duration = time.time() - start_time
    print(f"Pathology query complete. Retrieved {len(df)} total rows in {total_duration:.2f} seconds.")
    
    return df

def get_surgery_data(patient_ids, batch_size=1000):
    """
    Get breast surgical case data for specific patient IDs, processing in batches.

    Clinical document / surgical note text is intentionally not queried -- the
    FACT_CLINICAL_DOCUMENTS table is ~5.5 TB and made the run far too slow for
    the little value the notes added.

    Args:
        patient_ids (list): List of patient IDs to query
        batch_size (int): Number of patients to process in each batch

    Returns:
        pandas.DataFrame: Query results as a dataframe
    """
    start_time = time.time()
    print("Initializing BigQuery client for surgery data...")
    client = bigquery.Client()

    all_results = []
    total_patients = len(patient_ids)
    total_batches = (total_patients + batch_size - 1) // batch_size

    for i in tqdm(range(0, total_patients, batch_size), total=total_batches):
        batch = patient_ids[i:i+batch_size]

        # Format IDs appropriately
        if batch and all(str(id).isdigit() for id in batch):
            ids_str = ', '.join([str(id) for id in batch])
        else:
            ids_str = ', '.join([f"'{id}'" for id in batch])

        query = f"""
        SELECT DISTINCT
          PAT_DIM_PATIENT.PATIENT_CLINIC_NUMBER AS PAT_PATIENT_CLINIC_NUMBER,
          SURGDIAG_FACT_SURGICAL_CASE_DIAGNOSIS.SURGICAL_CASE_ID AS SURGDIAG_SURGICAL_CASE_ID,
          SURGDIAG_FACT_SURGICAL_CASE_DIAGNOSIS.SURGICAL_DIAGNOSIS_DTM AS SURGDIAG_SURGICAL_DIAGNOSIS_DTM,
          SURGPROC_DIM_SURGICAL_PROCEDURE.SURGICAL_PROCEDURE_DESCRIPTION AS SURGPROC_SURGICAL_PROCEDURE_DESCRIPTION,
          SURGPROC_DIM_SURGICAL_PROCEDURE.SURGICAL_PROCEDURE_CODE AS SURGPROC_SURGICAL_PROCEDURE_CODE,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_ASA_SCORE AS SURGCASE_SURGICAL_ASA_SCORE,
          DATE_DIFF(EXTRACT(DATE FROM SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_DTM), PAT_DIM_PATIENT.PATIENT_BIRTH_DATE, YEAR) -
            IF(EXTRACT(MONTH FROM PAT_DIM_PATIENT.PATIENT_BIRTH_DATE)*100 + EXTRACT(DAY FROM PAT_DIM_PATIENT.PATIENT_BIRTH_DATE) >
              EXTRACT(MONTH FROM SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_DTM)*100 + EXTRACT(DAY FROM SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_DTM),1,0) AS SURGCASE_PATIENT_AGE_AT_EVENT,
          ANALOC_DIM_ANATOMIC_LOCATION.ANATOMIC_LOCATION_DESCRIPTION AS ANALOC_ANATOMIC_LOCATION_DESCRIPTION,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_ANESTHESIA_CODE AS SURGCASE_SURGICAL_ANESTHESIA_CODE,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_ANESTHESIA_DESCRIPTION AS SURGCASE_SURGICAL_ANESTHESIA_DESCRIPTION,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_ADD_ON_INDICATOR AS SURGCASE_SURGICAL_CASE_ADD_ON_INDICATOR,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_CLASS AS SURGCASE_SURGICAL_CASE_CLASS,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_COUNT AS SURGCASE_SURGICAL_CASE_COUNT,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_ISOLATION_INDICATOR AS SURGCASE_SURGICAL_CASE_ISOLATION_INDICATOR,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CLOSURE_DTM AS SURGCASE_SURGICAL_CLOSURE_DTM,
          DIAGCODE_DIM_DIAGNOSIS_CODE.DIAGNOSIS_CODE AS DIAGCODE_DIAGNOSIS_CODE,
          SURGDIAG_FACT_SURGICAL_CASE_DIAGNOSIS.SURGICAL_CASE_DIAGNOSIS_COUNT AS SURGDIAG_SURGICAL_CASE_DIAGNOSIS_COUNT,
          DIAGCODE_DIM_DIAGNOSIS_CODE.DIAGNOSIS_DESCRIPTION AS DIAGCODE_DIAGNOSIS_DESCRIPTION,
          DIAGCODE_DIM_DIAGNOSIS_CODE.DIAGNOSIS_METHOD_NAME AS DIAGCODE_DIAGNOSIS_METHOD_NAME,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_INCISION_BEGIN_DTM AS SURGCASE_SURGICAL_INCISION_BEGIN_DTM,
          LOC_DIM_LOCATION.LOCATION_DESCRIPTION AS LOC_LOCATION_DESCRIPTION,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_OP_ROOM_ENTER_DTM AS SURGCASE_SURGICAL_OP_ROOM_ENTER_DTM,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_OP_ROOM_EXIT_DTM AS SURGCASE_SURGICAL_OP_ROOM_EXIT_DTM,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_OPERATION_END_DTM AS SURGCASE_SURGICAL_OPERATION_END_DTM,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_OPERATION_START_DTM AS SURGCASE_SURGICAL_OPERATION_START_DTM,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_PACU_LEVEL_CODE AS SURGCASE_SURGICAL_PACU_LEVEL_CODE,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_POST_OP_DIAGNOSIS_NOTE AS SURGCASE_SURGICAL_POST_OP_DIAGNOSIS_NOTE,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_POST_OP_PROCEDURE_NOTE AS SURGCASE_SURGICAL_POST_OP_PROCEDURE_NOTE,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_POSTPROCEDURE_DTM AS SURGCASE_SURGICAL_POSTPROCEDURE_DTM,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_PRE_OP_DIAGNOSIS_NOTE AS SURGCASE_SURGICAL_PRE_OP_DIAGNOSIS_NOTE,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_PRE_OP_PROCEDURE_NOTE AS SURGCASE_SURGICAL_PRE_OP_PROCEDURE_NOTE,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_PREPROCEDURE_DTM AS SURGCASE_SURGICAL_PREPROCEDURE_DTM,
          CLINSERV1_DIM_CLINICAL_SERVICE.CLINICAL_SERVICE_CODE AS CLINSERV1_CLINICAL_SERVICE_CODE,
          CLINSERV1_DIM_CLINICAL_SERVICE.CLINICAL_SERVICE_DESCRIPTION AS CLINSERV1_CLINICAL_SERVICE_DESCRIPTION,
          CASEPRIORITY_DIM_SURGICAL_CASE_PRIORITY.SURGICAL_CASE_PRIORITY_CODE AS CASEPRIORITY_SURGICAL_CASE_PRIORITY_CODE,
          CASEPRIORITY_DIM_SURGICAL_CASE_PRIORITY.SURGICAL_CASE_PRIORITY_DESCRIPTION AS CASEPRIORITY_SURGICAL_CASE_PRIORITY_DESCRIPTION,
          SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.SURGICAL_PROCEDURE_NHSN_CLOSURE_TECHNIQUE AS SURGPROC_SURGICAL_PROCEDURE_NHSN_CLOSURE_TECHNIQUE,
          SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.SURGICAL_PROCEDURE_ORDERED_DESCRIPTION AS SURGPROC_SURGICAL_PROCEDURE_ORDERED_DESCRIPTION,
          SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.PRIMARY_PROCEDURE_INDICATOR AS SURGPROC_PRIMARY_PROCEDURE_INDICATOR,
          SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.SURGICAL_PROCEDURE_BILATERAL_CODE AS SURGPROC_SURGICAL_PROCEDURE_BILATERAL_CODE,
          SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.SURGICAL_CASE_ID AS SURGPROC_SURGICAL_CASE_ID,
          SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.SURGICAL_CASE_PROCEDURE_COUNT AS SURGPROC_SURGICAL_CASE_PROCEDURE_COUNT,
          SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.SURGICAL_PROCEDURE_SEQUENCE AS SURGPROC_SURGICAL_PROCEDURE_SEQUENCE,
          SURGCASE_FACT_SURGICAL_CASE.SITE_CODE AS SURGCASE_LOCATION_SITE_NAME,
          SURGCASE_FACT_SURGICAL_CASE.VISIT_NUMBER AS SURGCASE_VISIT_NUMBER,
          SURGCASE_FACT_SURGICAL_CASE.SURGICAL_VISIT_TYPE AS SURGCASE_SURGICAL_VISIT_TYPE,
          WOUNDTYPE_DIM_WOUND_TYPE.WOUND_TYPE_CODE AS WOUNDTYPE_WOUND_TYPE_CODE,
          WOUNDTYPE_DIM_WOUND_TYPE.WOUND_TYPE_DESCRIPTION AS WOUNDTYPE_WOUND_TYPE_DESCRIPTION
        FROM `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_SURGICAL_CASE` SURGCASE_FACT_SURGICAL_CASE
        INNER JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_LOCATION` LOC_DIM_LOCATION
          ON (SURGCASE_FACT_SURGICAL_CASE.LOCATION_DK = LOC_DIM_LOCATION.LOCATION_DK)
        INNER JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_PATIENT` PAT_DIM_PATIENT
          ON (SURGCASE_FACT_SURGICAL_CASE.PATIENT_DK = PAT_DIM_PATIENT.PATIENT_DK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_SURGICAL_CASE_PROCEDURE` SURGPROC_FACT_SURGICAL_CASE_PROCEDURE
          ON (SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_FPK = SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.SURGICAL_CASE_FPK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_SURGICAL_CASE_DIAGNOSIS` SURGDIAG_FACT_SURGICAL_CASE_DIAGNOSIS
          ON (SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_FPK = SURGDIAG_FACT_SURGICAL_CASE_DIAGNOSIS.SURGICAL_CASE_FPK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_DIAGNOSIS_CODE` DIAGCODE_DIM_DIAGNOSIS_CODE
          ON (SURGDIAG_FACT_SURGICAL_CASE_DIAGNOSIS.DIAGNOSIS_CODE_DK = DIAGCODE_DIM_DIAGNOSIS_CODE.DIAGNOSIS_CODE_DK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_SURGICAL_PROCEDURE` SURGPROC_DIM_SURGICAL_PROCEDURE
          ON (SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.SURGICAL_PROCEDURE_DK = SURGPROC_DIM_SURGICAL_PROCEDURE.SURGICAL_PROCEDURE_DK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_ANATOMIC_LOCATION` ANALOC_DIM_ANATOMIC_LOCATION
          ON (SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.ANATOMIC_LOCATION_DK = ANALOC_DIM_ANATOMIC_LOCATION.ANATOMIC_LOCATION_DK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_WOUND_TYPE` WOUNDTYPE_DIM_WOUND_TYPE
          ON (SURGPROC_FACT_SURGICAL_CASE_PROCEDURE.WOUND_TYPE_DK = WOUNDTYPE_DIM_WOUND_TYPE.WOUND_TYPE_DK)
        INNER JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_SURGICAL_CASE_PRIORITY` CASEPRIORITY_DIM_SURGICAL_CASE_PRIORITY
          ON (SURGCASE_FACT_SURGICAL_CASE.SURGICAL_CASE_PRIORITY_DK = CASEPRIORITY_DIM_SURGICAL_CASE_PRIORITY.SURGICAL_CASE_PRIORITY_DK)
        INNER JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_CLINICAL_SERVICE` CLINSERV1_DIM_CLINICAL_SERVICE
          ON (SURGCASE_FACT_SURGICAL_CASE.CLINICAL_SERVICE_DK = CLINSERV1_DIM_CLINICAL_SERVICE.CLINICAL_SERVICE_DK)
        WHERE
          PAT_DIM_PATIENT.PATIENT_CLINIC_NUMBER IN ({ids_str})
          AND SURGPROC_DIM_SURGICAL_PROCEDURE.SURGICAL_PROCEDURE_DESCRIPTION LIKE '%BREAST%'
        """

        batch_df = client.query(query).to_dataframe()
        all_results.append(batch_df)

    if all_results:
        df = pd.concat(all_results, ignore_index=True)
    else:
        df = pd.DataFrame()

    total_duration = time.time() - start_time
    print(f"Surgery query complete. Retrieved {len(df)} total rows in {total_duration:.2f} seconds.")

    return df


def run_surgery_query(patient_ids):
    """
    Run the batched breast surgery query and return the deduplicated result.

    (Clinical document / surgical note text was previously pulled here too, but
    FACT_CLINICAL_DOCUMENTS is ~5.5 TB and made the run far too slow for the
    little value the notes added, so that query was dropped.)

    Args:
        patient_ids (list): List of patient IDs to query

    Returns:
        pandas.DataFrame: Surgery data
    """
    surgery_df = get_surgery_data(patient_ids)
    surgery_df = surgery_df.drop_duplicates()

    return surgery_df


def run_breast_imaging_query(limit=None):
    """
    Run queries with complete radiology, pathology, and lab data per patient
    
    Args:
        limit (int, optional): Limit the number of patients to process
    """
    print(f"Setting query limit to {limit}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(env, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Set up our audit destination
    append_audit("query.patient_limit", limit, new_file=True)
    
    total_start_time = time.time()
    print("Starting breast imaging query process...")
    
    # Step 1: Get all radiology data for breast imaging
    print("\n=== RADIOLOGY DATA QUERY ===")
    rad_df = get_radiology_data(limit=limit)

    # Audit radiology results
    rad_path = os.path.join(env, "data", "raw_radiology.csv")
    rad_df.to_csv(rad_path, index=False)
    append_audit("query.raw_rad_record_count", len(rad_df))
    
    # Extract unique patient IDs from the radiology data
    patient_ids = rad_df['PATIENT_ID'].unique().tolist()
    append_audit("query.raw_rad_unique_patients", len(patient_ids))
    print(f"Extracted {len(patient_ids)} unique patient IDs for pathology query")
    
    # Step 2: Get pathology data for these patients
    print("\n=== PATHOLOGY DATA QUERY ===")
    path_df = get_pathology_data(patient_ids)
    
    # Audit pathology results
    path_path = os.path.join(env, "data", "raw_pathology.csv")
    path_df.to_csv(path_path, index=False)
    append_audit("query.raw_path_record_count", len(path_df))
    
    
    
    # Calculate patient coverage metrics
    patients_with_path = path_df['PATIENT_ID'].nunique()
    path_coverage_percentage = (patients_with_path / len(patient_ids)) * 100 if patient_ids else 0

    print(f"{patients_with_path} of {len(patient_ids)} radiology patients ({path_coverage_percentage:.1f}%) have pathology data")

    append_audit("query.rad_patients_with_path", patients_with_path)

    # Step 3: Get breast surgery data for these patients
    print("\n=== SURGERY DATA QUERY ===")
    surgery_df = run_surgery_query(patient_ids)

    # Audit surgery results
    surgery_path = os.path.join(env, "data", "raw_surgery.csv")
    surgery_df.to_csv(surgery_path, index=False)
    append_audit("query.raw_surgery_record_count", len(surgery_df))

    if not surgery_df.empty:
        patients_with_surgery = surgery_df['PAT_PATIENT_CLINIC_NUMBER'].nunique()
        surgery_coverage_percentage = (patients_with_surgery / len(patient_ids)) * 100 if patient_ids else 0
        print(f"{patients_with_surgery} of {len(patient_ids)} radiology patients ({surgery_coverage_percentage:.1f}%) have breast surgery data")
        append_audit("query.rad_patients_with_surgery", patients_with_surgery)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nAll queries complete! Total execution time: {total_duration:.2f} seconds")

    return rad_df, path_df, surgery_df