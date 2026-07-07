
import os
import pandas as pd
import re
from src.DB_processing.tools import append_audit
from src.data_ingest.findings_parser import add_ultrasound_classifications
from src.ML_processing.findings_BERT import add_ultrasound_classifications_bert
# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory
env = os.path.dirname(env)  # Go back one directory


def determine_laterality(row):
    # Function to check a single text field
    def check_text_for_laterality(text, right_text, left_text):
        if pd.isna(text):
            return None
        
        text = text.upper()
        
        
        # Check for BILATERAL indicators
        if "BILAT" in text or "BOTH" in text:
            return "BILATERAL"
        
        # Check for clear RIGHT indicators
        elif any(x in text for x in right_text) and "BILAT" not in text:
            return "RIGHT"
        
        # Check for clear LEFT indicators
        elif any(x in text for x in left_text) and "BILAT" not in text:
            return "LEFT"
        
        # If no laterality is found, return None
        else:
            return None
    
    # First try DESCRIPTION column
    if 'DESCRIPTION' in row and not pd.isna(row['DESCRIPTION']):
        laterality = check_text_for_laterality(row['DESCRIPTION'], ["RIGHT", "R BI", " RT", "RT "], ["LEFT", "L BI", " LT", "LT "])
        if laterality is not None:
            return laterality
        
    # Then try DESCRIPTION column
    if 'TEST_DESCRIPTION' in row and not pd.isna(row['TEST_DESCRIPTION']):
        laterality = check_text_for_laterality(row['TEST_DESCRIPTION'], ["RIGHT", "R BI",], ["LEFT", "L BI",])
        if laterality is not None:
            return laterality
    
    # If not found or DESCRIPTION is empty, try RADIOLOGY_REPORT
    if 'RADIOLOGY_REPORT' in row and not pd.isna(row['RADIOLOGY_REPORT']):
        laterality = check_text_for_laterality(row['RADIOLOGY_REPORT'], ["RIGHT", "R BI"], ["LEFT", "L BI"])
        if laterality is not None:
            return laterality
    
    # If still not found, return None
    return None


def extract_birads_and_description(row):
    # First try RADIOLOGY_REPORT if available
    if 'RADIOLOGY_REPORT' in row and not pd.isna(row['RADIOLOGY_REPORT']):
        text = row['RADIOLOGY_REPORT']
        result = extract_birads_from_text(text)
        if result[0] is not None:  # If BI-RADS was found in RADIOLOGY_REPORT
            return result
    
    # If no result from RADIOLOGY_REPORT, try RADIOLOGY_NARRATIVE
    if 'RADIOLOGY_NARRATIVE' in row and not pd.isna(row['RADIOLOGY_NARRATIVE']):
        text = row['RADIOLOGY_NARRATIVE']
        result = extract_birads_from_text(text)
        if result[0] is not None:  # If BI-RADS was found in RADIOLOGY_NARRATIVE
            return result
    
    return None, None

def extract_birads_from_text(text):
    if pd.isna(text):
        return None, None
    
    # List of keywords that should end a description
    end_keywords = [
        'benign', 'malignant', 'malignancy', 'suspicious', 
        'negative', 'positive', 'cancer', 'indeterminate', 'incomplete'
    ]
    
    # Create pattern to find any of the keywords
    end_pattern = r'(?i)(' + '|'.join(end_keywords) + r')[^\w]'
    
    # Valid BI-RADS category numbers with optional subcategories
    birads_numbers = r'(?:0|1|2|3|4[abcABC]?|5|6)'
    
    # Add new patterns for standalone "Category X" without BI-RADS
    category_patterns = [
        # Standalone Category patterns
        rf'\bCategory\s*(?:number|#|no\.)?[:\s]*\(?({birads_numbers})\)?(?:[:]\s*|\s+|,\s*|\s*-\s*|\s*/\s*)([^\.]+)',
        rf'\bCategory\s*(?:number|#|no\.)?[:\s]*\(?({birads_numbers})\)?(?:\s+\(([^)]+)\))',
        
        # More relaxed pattern for just the category number with parenthetical description
        rf'\bCategory\s+({birads_numbers})(?:\s*\(([^)]+)\))',
    ]
    
    # Original patterns
    original_patterns = [
        # "ACR code X Description" format
        r'ACR\s+code:?\s+(\d+[a-z]?)\s+([^\.]+)',

        # "BI-RADS ASSESSMENT: CODE: X-DESCRIPTION" format
        r'BI-?RADS:?\s+ASSESSMENT:\s*CODE:\s*(\d+[a-z]?)-([^\.\s]+)',
        r'BI-?RADS:?\s*Code\s*(\d+[a-z]?),\s*([^\.\s]+)',
        
        # Ultrasound-specific patterns with description
        r'(?:Ultrasound|US)\s+BI-?RADS:\s*\(?(\d+[a-z]?)\)?\s+([^\s\.]+)',
        
        # General BI-RADS patterns with descriptions - Combined several patterns
        r'(?:BI-?RADS\s*(?:ASSESSMENT|Category|code|Final\s+Assessment)?|ASSESSMENT:\s*BI-?RADS|OVERALL\s*STUDY\s*BI-?RADS)(?:\s*CATEGORY)?[:]?\s*\(?(\d+[a-z]?)\)?(?:[:]\s*|\s+|,\s*|\s*-\s*)([^\.]+)(?:\.)?',
        
        # Special case for impression
        r'(?:IMPRESSION:|ASSESSMENT:)?\s*(?:BI-?RADS)\s*\(?(\d+[a-z]?)\)?\s*(?:-|,)\s*([^\.]+)',
        
        # Special case for description before number
        r'BI-?RADS:\s*([^(]+)\s*\((\d+[a-z]?)\)',
        
        # Final Assessment without BI-RADS explicitly mentioned
        r'Final\s+Assessment:\s*\(?(\d+[a-z]?)\)?\s*(?:-|,)?\s*([^\.]+)',
        
        # Pattern for BI-RADS ATLAS category
        r'BI-?RADS(?:®\s*ATLAS)?\s*category\s*\(?overall\)?:\s*\(?(\d+[a-z]?)\)?\s+([^\.]+)',
        
        # Pattern for BI-RADS® Category with registered trademark
        r'BI-?RADS®\s*Category:\s*(\d+[a-z]?)\s*-\s*([^\.]+)',
    ]
    
    # Combine both sets of patterns, checking the new ones first
    all_patterns = category_patterns + original_patterns

    for pattern in all_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and len(match.groups()) >= 2:
            # Special case for the "description before number" pattern
            if pattern == r'BI-?RADS:\s*([^(]+)\s*\((\d+[a-z]?)\)':
                birads_category = match.group(2)
                full_description = match.group(1).strip()
            else:
                birads_category = match.group(1)
                full_description = match.group(2).strip() if match.group(2) else None
            
            # Convert any letters in the BI-RADS category to uppercase
            if birads_category:
                birads_category = ''.join([c.upper() if c.isalpha() else c for c in birads_category])
            
            # If we have a description, process it
            if full_description:
                # Truncate description at any of the specified keywords
                keyword_match = re.search(end_pattern, full_description + ' ')
                if keyword_match:
                    # Get the position of the keyword plus its length
                    key_end_pos = keyword_match.end() - 1  # -1 to exclude the non-word character
                    description = full_description[:key_end_pos].strip()
                else:
                    description = full_description
            else:
                description = None
                
            return birads_category, description
    
    # Simpler patterns without description capturing
    simple_patterns = [
        r'(?:Ultrasound|US)\s+BI-?RADS:\s*\(?(\d+[a-z]?)\)?',
        r'(?:BI-?RADS|BIRADS)(?:\s*(?:Category|CATEGORY|code))?(?::|:?\s+CATEGORY)?\s*\(?(\d+[A-Za-z]?)\)?',
        r'OVERALL\s*STUDY\s*BI-?RADS:\s*\(?(\d+[A-Za-z]?)\)?',
        r'BI-?RADS\s+Category\s+No\.\s*(\d+[a-z]?)',
        # New simple category pattern
        rf'\bCategory\s*(?:number|#|no\.)?[:\s]*\(?({birads_numbers})\)?',
        rf'\bCategory\s+({birads_numbers})\b',
    ]
    
    for pattern in simple_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            birads_category = match.group(1)
            # Convert any letters in the BI-RADS category to uppercase
            if birads_category:
                birads_category = ''.join([c.upper() if c.isalpha() else c for c in birads_category])
            return birads_category, None
    
    # Special case for assessment with pathology but no explicit BI-RADS
    pathology_match = re.search(r'ASSESSMENT:\s*\d+:\s*(Pathology\s+\w+)', text, re.IGNORECASE)
    if pathology_match:
        return None, pathology_match.group(1).strip()
    
    return None, None

def extract_density(text):
    # Extract text after "DENSITY:" until next section header (WORD:)
    if pd.isna(text):
        return None
    
    # Check if "DENSITY:" exists in the text
    if "DENSITY:" not in text:
        return None
    
    # Split by "DENSITY:" and get the content after it
    after_density = text.split("DENSITY:")[1].strip()
    
    # Use regex to find the next uppercase word followed by a colon
    match = re.search(r'([A-Z]{2,}:)', after_density)
    
    if match:
        # Get position of the next section header
        end_pos = match.start()
        # Extract text from after "DENSITY:" until the next section header
        density_text = after_density[:end_pos].strip()
        return density_text
    else:
        # If no next section header is found, return all text after "DENSITY:"
        return after_density
    
    
def extract_findings_and_fallback(row):
    # First try RADIOLOGY_REPORT if available
    if 'RADIOLOGY_REPORT' in row and not pd.isna(row['RADIOLOGY_REPORT']):
        text = row['RADIOLOGY_REPORT']
        result = extract_findings(text)
        if result is not None:  # If FINDINGS was found in RADIOLOGY_REPORT
            return result
    
    # If no result from RADIOLOGY_REPORT, try RADIOLOGY_NARRATIVE
    if 'RADIOLOGY_NARRATIVE' in row and not pd.isna(row['RADIOLOGY_NARRATIVE']):
        text = row['RADIOLOGY_NARRATIVE']
        result = extract_findings(text)
        if result is not None:  # If FINDINGS was found in RADIOLOGY_NARRATIVE
            return result
    
    return None

def extract_findings(text):
    if pd.isna(text):
        return None
    
    # Case-sensitive patterns to search for
    findings_patterns = ["FINDINGS", "Ultrasound Findings", "Findings:"]
    findings_match = None
    
    # Find which pattern exists in the text (case-sensitive)
    for pattern in findings_patterns:
        if pattern in text:
            findings_match = text.split(pattern, 1)  # Split only on first occurrence
            break
    
    # If no findings pattern found, return None
    if findings_match is None or len(findings_match) < 2:
        return None
    
    # Get content after the findings keyword
    after_findings = findings_match[1]
    
    # Strip leading punctuation: ':', ' -', ': -', or whitespace combinations
    after_findings = after_findings.lstrip()  # Remove leading whitespace first
    
    # Remove leading ':' or '-' characters
    while after_findings and after_findings[0] in [':', '-']:
        after_findings = after_findings[1:].lstrip()
    
    # Look for "IMPRESSION" to mark the end
    if "IMPRESSION" in after_findings:
        end_pos = after_findings.find("IMPRESSION")
        findings_text = after_findings[:end_pos].strip()
        return findings_text
    else:
        # No IMPRESSION found, return all text after FINDINGS
        return after_findings.strip()

def extract_rad_pathology_txt(text):
    if pd.isna(text):
        return None
    
    # Check if "PATHOLOGY:" exists in the text
    if "PATHOLOGY:" not in text:
        return None
    
    # Split by "PATHOLOGY:" and get the content after it
    after_pathology = text.split("PATHOLOGY:")[1].strip()
    
    # Use regex to find the next uppercase word followed by a colon
    match = re.search(r'([A-Z]{2,}:)', after_pathology)
    
    if match:
        # Get position of the next section header
        end_pos = match.start()
        # Extract text from after "PATHOLOGY:" until the next section header
        pathology_text = after_pathology[:end_pos].strip()
        return pathology_text
    else:
        # If no next section header is found, return all text after "PATHOLOGY:"
        return after_pathology

def check_for_biopsy(row):
    """
    Check for biopsy and ultrasound biopsy in DESCRIPTION or TEST_DESCRIPTION columns
    
    Args:
        row: The dataframe row with columns to check
        
    Returns:
        Tuple of (biopsy_found, ultrasound_biopsy_found) where each is 'T' if found, 'F' if not
    """
    biopsy_found = 'F'
    ultrasound_biopsy_found = 'F'
    
    # Check DESCRIPTION column
    if 'DESCRIPTION' in row and not pd.isna(row['DESCRIPTION']):
        description_upper = row['DESCRIPTION'].upper()

        # Check for biopsy
        if ('BIOPSY' in description_upper or 'BX' in description_upper or
            'ASP' in description_upper or 'SPECIMEN' in description_upper):
            biopsy_found = 'T'

            # Check if it's an ultrasound biopsy in this column
            if ('US' in description_upper or 'ULTRASOUND' in description_upper or
                re.search(r'\bUL\b', description_upper)):
                ultrasound_biopsy_found = 'T'

    # Check TEST_DESCRIPTION column
    if 'TEST_DESCRIPTION' in row and not pd.isna(row['TEST_DESCRIPTION']):
        test_description_upper = row['TEST_DESCRIPTION'].upper()

        # Check for biopsy
        if ('BIOPSY' in test_description_upper or 'BX' in test_description_upper or
            'ASP' in test_description_upper or 'SPECIMEN' in test_description_upper):
            biopsy_found = 'T'

            # Check if it's an ultrasound biopsy in this column
            if ('US' in test_description_upper or 'ULTRASOUND' in test_description_upper or
                re.search(r'\bUL\b', test_description_upper)):
                ultrasound_biopsy_found = 'T'
    
    return biopsy_found, ultrasound_biopsy_found


def extract_rad_impression(text):
    if pd.isna(text):
        return None
    
    # Check if "IMPRESSION" exists in the text
    if "IMPRESSION" not in text:
        return None
    
    # Check if "IMPRESSION:" or "IMPRESSION" exists in the text
    if "IMPRESSION:" in text:
        # Split by "IMPRESSION:" and get the content after it
        after_impression = text.split("IMPRESSION:", 1)[1].strip()
    elif "IMPRESSION" in text:
        # Split by "IMPRESSION" and get the content after it
        after_impression = text.split("IMPRESSION", 1)[1].strip()
        # Remove any colons within the first 50 characters
        if len(after_impression) > 50:
            first_50 = after_impression[:50].replace(':', '')
            after_impression = first_50 + after_impression[50:]
        else:
            after_impression = after_impression.replace(':', '')
    else:
        return None
    
    # Use regex to find the next uppercase word followed by a colon
    match = re.search(r'([A-Z]{2,}:)', after_impression)
    
    if match:
        # Get position of the next section header
        end_pos = match.start()
        # Extract text from after "IMPRESSION:" until the next section header
        impression_text = after_impression[:end_pos].strip()
        return impression_text
    else:
        # If no next section header is found, return all text after "IMPRESSION:"
        return after_impression
    
def remove_outside_records(radiology_df):
    """
    Remove rows where 'OUTSIDE' appears in the TEST_DESCRIPTION column
    """
    initial_row_count = len(radiology_df)
    
    # Make a copy to avoid warnings about setting values on a slice
    filtered_df = radiology_df.copy()
    
    # Check if the column exists
    if 'TEST_DESCRIPTION' in filtered_df.columns:
        # Create a mask for rows where 'OUTSIDE' is not in TEST_DESCRIPTION
        # Handle NaN values with a boolean mask
        mask = ~filtered_df['TEST_DESCRIPTION'].fillna('').str.upper().str.contains('OUTSIDE')
        
        # Apply the mask to filter out rows with 'OUTSIDE'
        filtered_df = filtered_df[mask]
    
    # Calculate how many rows were removed
    removed_count = initial_row_count - len(filtered_df)
    print(f"Removed {removed_count} outside records")
    
    return filtered_df


def remove_axilla_records(radiology_df):
    """
    Remove rows where 'AXILLA' appears in the TEST_DESCRIPTION column
    """
    initial_row_count = len(radiology_df)
    filtered_df = radiology_df.copy()
    
    if 'TEST_DESCRIPTION' in filtered_df.columns:
        mask = ~filtered_df['TEST_DESCRIPTION'].fillna('').str.upper().str.contains('AXILLA')
        filtered_df = filtered_df[mask]
    
    removed_count = initial_row_count - len(filtered_df)
    print(f"Removed {removed_count} axilla records")
    append_audit("query_clean_rad.axilla_removed", removed_count)
    
    return filtered_df

def add_previous_worst_mg_column(radiology_df):
    """
    Add columns 'previous_worst_MG' and 'previous_worst_MG_accession' that contain 
    the worst BI-RADS value and corresponding accession number from previous
    worst mammography exams. Tracks LEFT and RIGHT separately, with BILATERAL updating/using both.
    
    Args:
        radiology_df: DataFrame containing radiology data with columns:
                     PATIENT_ID, MODALITY, BI-RADS, Study_Laterality, RADIOLOGY_DTM, ACCESSION_NUMBER
                     
    Returns:
        DataFrame with added 'previous_worst_MG' and 'previous_worst_MG_accession' columns
    """
    
    # Define valid BI-RADS values in order from best to worst
    valid_birads_order = ['0', '1', '2', '3', '4', '4A', '4B', '4C', '5', '6']
    
    def is_valid_birads(birads_value):
        """Check if BI-RADS value is one of the accepted values."""
        if pd.isna(birads_value) or birads_value is None:
            return False
        birads_str = str(birads_value).upper().strip()
        return birads_str in valid_birads_order
    
    def get_worse_birads_with_accession(current_worst_tuple, new_birads, new_accession):
        """
        Return the worse of two BI-RADS values along with corresponding accession.
        
        Args:
            current_worst_tuple: (birads_value, accession_number) or None
            new_birads: new BI-RADS value to compare
            new_accession: accession number for the new BI-RADS
            
        Returns:
            tuple: (worse_birads, corresponding_accession)
        """
        if current_worst_tuple is None:
            return (str(new_birads).upper().strip(), new_accession)
        
        current_birads, current_accession = current_worst_tuple
        current_idx = valid_birads_order.index(str(current_birads).upper().strip())
        new_idx = valid_birads_order.index(str(new_birads).upper().strip())
        
        if new_idx > current_idx:
            return (str(new_birads).upper().strip(), new_accession)
        else:
            return current_worst_tuple
    
    print("Processing previous worst MG...")
    
    # Convert RADIOLOGY_DTM to datetime if it isn't already
    radiology_df['RADIOLOGY_DTM'] = pd.to_datetime(radiology_df['RADIOLOGY_DTM'], errors='coerce')
    
    # Initialize the new columns
    radiology_df['previous_worst_MG'] = None
    radiology_df['previous_worst_MG_accession'] = None
    
    # Sort by patient and date for efficient processing
    df_sorted = radiology_df.sort_values(['PATIENT_ID', 'RADIOLOGY_DTM']).copy()
    
    # Group by patient only
    grouped = df_sorted.groupby(['PATIENT_ID'])
    us_records_processed = 0
    
    # Process each patient
    for patient_id, group in grouped:
        
        # Track the worst MG BI-RADS and accession for each side separately
        # Each is a tuple: (birads_value, accession_number)
        worst_left_mg_tuple = None
        worst_right_mg_tuple = None
        
        # Iterate through records in chronological order
        for idx, row in group.iterrows():
            if row['MODALITY'] == 'MG' and is_valid_birads(row['BI-RADS']):
                birads_value = str(row['BI-RADS']).upper().strip()
                accession = row['ACCESSION_NUMBER']
                laterality = str(row['Study_Laterality']).upper().strip()
                
                # Update worst values based on laterality
                if laterality == 'LEFT':
                    worst_left_mg_tuple = get_worse_birads_with_accession(
                        worst_left_mg_tuple, birads_value, accession)
                elif laterality == 'RIGHT':
                    worst_right_mg_tuple = get_worse_birads_with_accession(
                        worst_right_mg_tuple, birads_value, accession)
                elif laterality == 'BILATERAL':
                    # BILATERAL MG updates both sides with same values
                    worst_left_mg_tuple = get_worse_birads_with_accession(
                        worst_left_mg_tuple, birads_value, accession)
                    worst_right_mg_tuple = get_worse_birads_with_accession(
                        worst_right_mg_tuple, birads_value, accession)
                
            elif row['MODALITY'] == 'US':
                laterality = str(row['Study_Laterality']).upper().strip()
                previous_worst_tuple = None
                
                # Determine previous worst based on US laterality
                if laterality == 'LEFT' and worst_left_mg_tuple is not None:
                    previous_worst_tuple = worst_left_mg_tuple
                elif laterality == 'RIGHT' and worst_right_mg_tuple is not None:
                    previous_worst_tuple = worst_right_mg_tuple
                elif laterality == 'BILATERAL':
                    # BILATERAL US takes the worst of both sides
                    if worst_left_mg_tuple is not None and worst_right_mg_tuple is not None:
                        # Compare the two sides and take the worse one
                        left_birads, left_accession = worst_left_mg_tuple
                        right_birads, right_accession = worst_right_mg_tuple
                        
                        left_idx = valid_birads_order.index(left_birads)
                        right_idx = valid_birads_order.index(right_birads)
                        
                        if right_idx > left_idx:
                            previous_worst_tuple = worst_right_mg_tuple
                        else:
                            previous_worst_tuple = worst_left_mg_tuple
                    elif worst_left_mg_tuple is not None:
                        previous_worst_tuple = worst_left_mg_tuple
                    elif worst_right_mg_tuple is not None:
                        previous_worst_tuple = worst_right_mg_tuple
                
                # Assign both BI-RADS and accession if we found a previous worst
                if previous_worst_tuple is not None:
                    birads_value, accession = previous_worst_tuple
                    radiology_df.loc[idx, 'previous_worst_MG'] = birads_value
                    radiology_df.loc[idx, 'previous_worst_MG_accession'] = accession
                    
                us_records_processed += 1
    
    # Count results
    us_with_prev_mg = radiology_df[
        (radiology_df['MODALITY'] == 'US') & 
        (radiology_df['previous_worst_MG'].notna())
    ]
    
    print(f"Processed {us_records_processed} US records")
    print(f"Found previous MG data for {len(us_with_prev_mg)} US records")
    
    return radiology_df

def extract_modality_guidance(row):
    """
    Extract modality guidance from the DESCRIPTION column by finding the word before ' GUID'.
    Only marks unknowns as 'OTHER' if is_biopsy is 'T'.

    Args:
        row: DataFrame row containing DESCRIPTION and is_biopsy columns

    Returns:
        str: Modality guidance value (US, MAMMO, TOMO, STEREOTACTIC, MR, OTHER, or None)
    """
    description = row['DESCRIPTION']
    is_biopsy = row.get('is_biopsy', 'F')

    if pd.isna(description):
        return None

    # Convert to uppercase for case-insensitive matching
    description_upper = description.upper()

    # Check if ' GUID' exists in the description
    if ' GUID' not in description_upper:
        return None

    # Mapping dictionary
    modality_map = {
        'ULTRASOUND': 'US',
        'US': 'US',
        'MAMMO': 'MAMMO',
        'TOMO': 'TOMO',
        'STEREOTACTIC': 'STEREOTACTIC',
        'STEREO': 'STEREOTACTIC',
        'MR': 'MR'
    }

    # Split text before ' GUID' to get all words
    text_before_guid = description_upper.split(' GUID')[0]
    words = text_before_guid.split()

    if not words:
        # Only mark as OTHER if is_biopsy is 'T'
        return 'OTHER' if is_biopsy == 'T' else None

    # First, check the immediate word before ' GUID'
    immediate_word = words[-1]
    if immediate_word in modality_map:
        return modality_map[immediate_word]

    # If immediate word doesn't match, check all words before ' GUID' from right to left
    for word in reversed(words[:-1]):  # Skip the last word (already checked)
        if word in modality_map:
            return modality_map[word]

    # If no match found, only return OTHER if is_biopsy is 'T'
    return 'OTHER' if is_biopsy == 'T' else None

def apply_biopsy_guidance_modality(radiology_df):
    """
    Apply guidance modality from biopsy rows to nearby exams and calculate days to biopsy.

    For each biopsy with a guidance modality:
    - Apply that modality to exams within -30 to 120 days (if they don't have one)
    - Calculate days_to_biopsy for all exams (days until next biopsy)

    Args:
        radiology_df: DataFrame with PATIENT_ID, RADIOLOGY_DTM, is_biopsy, MODALITY_GUIDANCE columns

    Returns:
        DataFrame with updated MODALITY_GUIDANCE and new days_to_biopsy column
    """
    print("Applying biopsy guidance modality to nearby exams...")

    # Convert RADIOLOGY_DTM to datetime if not already
    radiology_df['RADIOLOGY_DTM'] = pd.to_datetime(radiology_df['RADIOLOGY_DTM'], errors='coerce')

    # Initialize days_to_biopsy column
    radiology_df['days_to_biopsy'] = None

    # Sort by patient and date for efficient processing
    df_sorted = radiology_df.sort_values(['PATIENT_ID', 'RADIOLOGY_DTM']).copy()

    # Group by patient
    grouped = df_sorted.groupby('PATIENT_ID')

    guidance_applied_count = 0
    rows_with_days_to_biopsy = 0

    for patient_id, group in grouped:
        # Get all biopsy rows for this patient
        biopsy_rows = group[group['is_biopsy'] == 'T']

        if biopsy_rows.empty:
            continue

        # Process each row in the group
        for idx, row in group.iterrows():
            row_date = row['RADIOLOGY_DTM']

            # If this row is itself a biopsy, days_to_biopsy is 0
            if row['is_biopsy'] == 'T':
                radiology_df.loc[idx, 'days_to_biopsy'] = 0
                rows_with_days_to_biopsy += 1
            else:
                # Find the next biopsy after this row
                future_biopsies = biopsy_rows[biopsy_rows['RADIOLOGY_DTM'] > row_date]
                if not future_biopsies.empty:
                    next_biopsy = future_biopsies.iloc[0]
                    days_diff = (next_biopsy['RADIOLOGY_DTM'] - row_date).days
                    radiology_df.loc[idx, 'days_to_biopsy'] = days_diff
                    rows_with_days_to_biopsy += 1
                else:
                    # No future biopsy, check for past biopsies
                    past_biopsies = biopsy_rows[biopsy_rows['RADIOLOGY_DTM'] < row_date]
                    if not past_biopsies.empty:
                        # Get the most recent past biopsy
                        most_recent_past_biopsy = past_biopsies.iloc[-1]
                        days_diff = (most_recent_past_biopsy['RADIOLOGY_DTM'] - row_date).days  # This will be negative
                        radiology_df.loc[idx, 'days_to_biopsy'] = days_diff
                        rows_with_days_to_biopsy += 1

            # Apply guidance modality from nearby biopsies (if current row doesn't have one)
            if pd.isna(row['MODALITY_GUIDANCE']) or row['MODALITY_GUIDANCE'] == '':
                # Check all biopsies within the time window
                for biopsy_idx, biopsy_row in biopsy_rows.iterrows():
                    biopsy_date = biopsy_row['RADIOLOGY_DTM']
                    biopsy_guidance = biopsy_row['MODALITY_GUIDANCE']

                    # Skip if biopsy doesn't have guidance modality
                    if pd.isna(biopsy_guidance) or biopsy_guidance == '':
                        continue

                    # Calculate days difference (positive if row is before biopsy)
                    days_diff = (biopsy_date - row_date).days

                    # Check if within window: -30 to 120 days
                    # -30 means row can be up to 30 days AFTER the biopsy
                    # 120 means row can be up to 120 days BEFORE the biopsy
                    if -30 <= days_diff <= 120:
                        radiology_df.loc[idx, 'MODALITY_GUIDANCE'] = biopsy_guidance
                        guidance_applied_count += 1
                        break  # Apply from the first matching biopsy only

    print(f"Applied guidance modality to {guidance_applied_count} rows")
    print(f"Calculated days_to_biopsy for {rows_with_days_to_biopsy} rows")
    append_audit("query_clean_rad.guidance_modality_applied", guidance_applied_count)
    append_audit("query_clean_rad.days_to_biopsy_calculated", rows_with_days_to_biopsy)

    return radiology_df

def extract_addendum(row):
    """
    Extract addendum section from radiology reports. Searches for APPENDED, AMENDMENT, or ADDENDUM
    and returns everything after the first match found.

    Args:
        row: DataFrame row with RADIOLOGY_REPORT and RADIOLOGY_NARRATIVE columns

    Returns:
        str: Text from addendum keyword onwards, or None if not found
    """
    addendum_terms = ['ADDENDUM', 'AMENDMENT', 'APPENDED']
    
    # First try RADIOLOGY_REPORT
    if 'RADIOLOGY_REPORT' in row and not pd.isna(row['RADIOLOGY_REPORT']):
        text = row['RADIOLOGY_REPORT']
        text_upper = text.upper()
        
        # Check each term and find the earliest occurrence
        earliest_pos = len(text)
        earliest_term = None
        
        for term in addendum_terms:
            pos = text_upper.find(term)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                earliest_term = term
        
        # If we found a term, extract from that point onwards
        if earliest_term is not None:
            # Find the actual position in the original text (preserve case)
            addendum_text = text[earliest_pos:].strip()
            return addendum_text
    
    # If not found in RADIOLOGY_REPORT, try RADIOLOGY_NARRATIVE
    if 'RADIOLOGY_NARRATIVE' in row and not pd.isna(row['RADIOLOGY_NARRATIVE']):
        text = row['RADIOLOGY_NARRATIVE']
        text_upper = text.upper()
        
        # Check each term and find the earliest occurrence
        earliest_pos = len(text)
        earliest_term = None
        
        for term in addendum_terms:
            pos = text_upper.find(term)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                earliest_term = term
        
        # If we found a term, extract from that point onwards
        if earliest_term is not None:
            addendum_text = text[earliest_pos:].strip()
            return addendum_text
    
    return None

def add_prior_biopsy_columns(radiology_df):
    """
    Add columns 'left_prior_breast_biopsies' and 'right_prior_breast_biopsies' to track:
    - left_prior_breast_biopsies: count of previous biopsies on LEFT breast (Study_Laterality = LEFT or BILATERAL)
    - right_prior_breast_biopsies: count of previous biopsies on RIGHT breast (Study_Laterality = RIGHT or BILATERAL)
    """

    print("Processing prior breast biopsies by laterality...")

    # Convert RADIOLOGY_DTM to datetime if it isn't already
    radiology_df['RADIOLOGY_DTM'] = pd.to_datetime(radiology_df['RADIOLOGY_DTM'], errors='coerce')

    # Initialize the new columns
    radiology_df['left_prior_breast_biopsies'] = 0
    radiology_df['right_prior_breast_biopsies'] = 0

    # Sort by patient and date for efficient processing
    df_sorted = radiology_df.sort_values(['PATIENT_ID', 'RADIOLOGY_DTM']).copy()

    # Group by patient
    grouped = df_sorted.groupby('PATIENT_ID')

    # Process each patient
    for patient_id, group in grouped:
        # Track number of biopsies separately for each breast
        left_biopsy_count = 0
        right_biopsy_count = 0

        # Iterate through records in chronological order
        for idx, row in group.iterrows():
            # Set the prior values for this row (before updating counters)
            radiology_df.loc[idx, 'left_prior_breast_biopsies'] = left_biopsy_count
            radiology_df.loc[idx, 'right_prior_breast_biopsies'] = right_biopsy_count

            # Check if current row is a biopsy and update counters
            if row['is_biopsy'] == 'T':
                laterality = str(row.get('Study_Laterality', '')).upper().strip()

                if laterality == 'LEFT':
                    left_biopsy_count += 1
                elif laterality == 'RIGHT':
                    right_biopsy_count += 1
                elif laterality == 'BILATERAL':
                    # Bilateral biopsy counts for both sides
                    left_biopsy_count += 1
                    right_biopsy_count += 1

    # Count statistics
    total_with_left_prior_biopsies = (radiology_df['left_prior_breast_biopsies'] > 0).sum()
    total_with_right_prior_biopsies = (radiology_df['right_prior_breast_biopsies'] > 0).sum()

    print(f"Records with left prior biopsies: {total_with_left_prior_biopsies}")
    print(f"Records with right prior biopsies: {total_with_right_prior_biopsies}")
    append_audit("query_clean_rad.records_with_left_prior_biopsies", int(total_with_left_prior_biopsies))
    append_audit("query_clean_rad.records_with_right_prior_biopsies", int(total_with_right_prior_biopsies))

    return radiology_df

# Procedure-description regexes for breast-implant status. Removal is checked
# FIRST because "REMOVAL IMPLANT BREAST" / "REMOVAL TISSUE EXPANDER BREAST"
# also contain the present-implant keywords.
IMPLANT_REMOVED_PATTERN = re.compile(r'REMOVAL.*(?:IMPLANT|TISSUE EXPANDER)')
IMPLANT_PRESENT_PATTERN = re.compile(r'IMPLANT|AUGMENTATION|TISSUE EXPANDER')


def classify_implant_procedure(description):
    """
    Classify a surgical procedure description for breast-implant status:
      - 'removed'  implant/tissue-expander taken out
      - 'present'  implant/tissue-expander placed or retained (augmentation,
                   exchange, reconstruction-with-implant, repositioning, ...)
      - None       neutral -- capsule procedures (capsulectomy/capsulotomy/
                   capsulorrhaphy) and generic 'REVISION BREAST' imply an
                   implant history but don't reliably indicate current status,
                   and all non-implant procedures (lumpectomy, biopsy,
                   mastopexy, flap reconstruction, etc.)
    """
    if pd.isna(description):
        return None
    d = str(description).upper()
    if IMPLANT_REMOVED_PATTERN.search(d):
        return 'removed'
    if IMPLANT_PRESENT_PATTERN.search(d):
        return 'present'
    return None


def add_breast_implant_status(radiology_df, surgery_df):
    """
    Add a 'has_breast_implant' column to each radiology exam:
      - 'unknown'  default (no implant surgery on/before the exam date)
      - 'true'     patient's most recent breast-implant surgery on or before
                   the exam date placed/retained an implant
      - 'removed'  that most recent surgery removed the implant

    Matches on PATIENT_ID + exam date (RADIOLOGY_DTM) against the surgery
    events' clinic number + date (SURGCASE_SURGICAL_OPERATION_END_DTM).
    """
    print("Processing breast implant status...")

    radiology_df['has_breast_implant'] = 'unknown'

    if surgery_df is None or surgery_df.empty:
        print("  No surgery data -- has_breast_implant left as 'unknown'")
        return radiology_df

    # Build a per-patient timeline of classified implant events
    events = surgery_df[[
        'PAT_PATIENT_CLINIC_NUMBER',
        'SURGCASE_SURGICAL_OPERATION_END_DTM',
        'SURGPROC_SURGICAL_PROCEDURE_DESCRIPTION',
    ]].copy()
    events['status'] = events['SURGPROC_SURGICAL_PROCEDURE_DESCRIPTION'].apply(classify_implant_procedure)
    events = events[events['status'].notna()].copy()
    events['_pid_str'] = events['PAT_PATIENT_CLINIC_NUMBER'].astype(str)
    events['event_date'] = pd.to_datetime(events['SURGCASE_SURGICAL_OPERATION_END_DTM'], errors='coerce')
    events = events.dropna(subset=['event_date'])

    if events.empty:
        print("  No breast-implant surgery events detected")
        return radiology_df

    events_by_patient = {pid: g.sort_values('event_date') for pid, g in events.groupby('_pid_str')}

    # Temp columns for matching; leave PATIENT_ID/RADIOLOGY_DTM dtypes untouched
    radiology_df['_pid_str'] = radiology_df['PATIENT_ID'].astype(str)
    radiology_df['_exam_date'] = pd.to_datetime(radiology_df['RADIOLOGY_DTM'], errors='coerce')

    implant_true = 0
    implant_removed = 0
    for pid, patient_events in events_by_patient.items():
        for idx in radiology_df.index[radiology_df['_pid_str'] == pid]:
            exam_date = radiology_df.at[idx, '_exam_date']
            if pd.isna(exam_date):
                continue
            prior = patient_events[patient_events['event_date'] <= exam_date]
            if prior.empty:
                continue
            latest_status = prior.iloc[-1]['status']
            if latest_status == 'present':
                radiology_df.at[idx, 'has_breast_implant'] = 'true'
                implant_true += 1
            elif latest_status == 'removed':
                radiology_df.at[idx, 'has_breast_implant'] = 'removed'
                implant_removed += 1

    radiology_df.drop(columns=['_pid_str', '_exam_date'], inplace=True)

    print(f"  has_breast_implant: {implant_true} exams marked 'true', {implant_removed} marked 'removed'")
    append_audit("query_clean_rad.exams_with_implant_true", implant_true)
    append_audit("query_clean_rad.exams_with_implant_removed", implant_removed)

    return radiology_df


def remove_bad_data(radiology_df, output_path):
    # Count and remove rows with BI-RADS = '0'
    birads_zero_mask = radiology_df['BI-RADS'].isin(['0'])
    birads_zero_count = birads_zero_mask.sum() 
    radiology_df = radiology_df[~birads_zero_mask]
    
    # Remove patients without any 'US' modality exams
    # First, find patients who have at least one 'US' modality
    patients_with_us = radiology_df[radiology_df['MODALITY'] == 'US']['PATIENT_ID'].unique()
    
    # Count patients to be removed
    patients_to_remove = set(radiology_df['PATIENT_ID'].unique()) - set(patients_with_us)
    patients_removed_count = len(patients_to_remove)
    
    # Keep only patients who have at least one 'US' modality
    radiology_df = radiology_df[radiology_df['PATIENT_ID'].isin(patients_with_us)]
    
    
    print(f"Removed {birads_zero_count} rows with BI-RADS = '0'")
    print(f"Removed {patients_removed_count} patients without any 'US' modality exams (after previous removals)")
    append_audit("query_clean_rad.birads_0_removed", birads_zero_count)
    append_audit("query_clean_rad.missing_>=1_US_removed", patients_removed_count)
    
    return radiology_df
    
def extract_concordance(text):
    """Look for 'CONCORDANCE:' followed by Yes/No (case-insensitive, allowing
    whitespace/punctuation in between). Returns 'T' / 'F' / None."""
    if pd.isna(text):
        return None
    m = re.search(r'CONCORDANCE\s*:[\s\W]*(YES|NO)\b', str(text), re.IGNORECASE)
    if not m:
        return None
    return 'T' if m.group(1).upper() == 'YES' else 'F'


def filter_rad_data(radiology_df, output_path, surgery_df=None):
    print("Parsing Radiology Data:")
    
    # Print length
    initial_count = len(radiology_df)
    print(f"Initial dataframe length: {initial_count} rows")
    
    # Audit year range
    temp_dates = pd.to_datetime(radiology_df['RADIOLOGY_DTM'], errors='coerce')
    
    # Extract years into a temporary series
    years = temp_dates.dt.year.dropna()
    year_min = int(years.min())
    year_max = int(years.max())
    append_audit("query_clean_rad.init_year_min", year_min)
    append_audit("query_clean_rad.init_year_max", year_max)

    rename_dict = {'PAT_PATIENT_CLINIC_NUMBER': 'PATIENT_ID',
        'IMGST_ACCESSION_IDENTIFIER_VALUE': 'Accession_Number',
        'IMGST_DESCRIPTION': 'Biopsy_Desc',}
    
    # Rename columns
    radiology_df = radiology_df.rename(columns=rename_dict)
    
    # Remove outside records
    count_before = len(radiology_df)
    radiology_df = remove_outside_records(radiology_df)
    outside_removed = count_before - len(radiology_df)
    append_audit("query_clean_rad.outside_removed", outside_removed)
    
    # Remove axilla records
    radiology_df = remove_axilla_records(radiology_df)
    
    # Apply the extraction functions and create new columns
    radiology_df['Density_Desc'] = radiology_df['RADIOLOGY_REPORT'].apply(extract_density)
    
    # Apply the BI-RADS extraction and create separate columns
    birads_results = radiology_df.apply(extract_birads_and_description, axis=1)
    radiology_df['BI-RADS'] = [result[0] for result in birads_results]
    radiology_df['Biopsy'] = [result[1] for result in birads_results]
    
    # Find Laterality 
    radiology_df['Study_Laterality'] = radiology_df.apply(determine_laterality, axis=1)
    
    # Extract pathology text
    radiology_df['rad_pathology_txt'] = radiology_df['RADIOLOGY_REPORT'].apply(extract_rad_pathology_txt)
    
    # Extract impression text
    radiology_df['rad_impression'] = radiology_df['RADIOLOGY_REPORT'].apply(extract_rad_impression)

    # Extract findings text
    radiology_df['FINDINGS'] = radiology_df.apply(extract_findings_and_fallback, axis=1)

    # Check for biopsy in DESCRIPTION column
    results = radiology_df.apply(check_for_biopsy, axis=1)
    radiology_df['is_biopsy'] = results.str[0]
    radiology_df['is_us_biopsy'] = results.str[1]

    # Check for addendum
    radiology_df['addendum'] = radiology_df.apply(extract_addendum, axis=1)

    # Extract concordance (Yes/No after "CONCORDANCE:") from RADIOLOGY_REPORT
    radiology_df['concordance'] = radiology_df['RADIOLOGY_REPORT'].apply(extract_concordance)

    # Extract modality guidance (must be after is_biopsy is created)
    radiology_df['MODALITY_GUIDANCE'] = radiology_df.apply(extract_modality_guidance, axis=1)

    # Apply guidance modality from biopsies to nearby exams and calculate days_to_biopsy
    radiology_df = apply_biopsy_guidance_modality(radiology_df)

    radiology_df = add_ultrasound_classifications(radiology_df, output_path)
    #radiology_df = add_ultrasound_classifications_bert(radiology_df, output_path)

    # Add previous worst MG column
    radiology_df = add_previous_worst_mg_column(radiology_df)

    # Add prior biopsy and cancer columns
    radiology_df = add_prior_biopsy_columns(radiology_df)

    # Add breast implant status from surgery history (per exam, temporal)
    radiology_df = add_breast_implant_status(radiology_df, surgery_df)

    # Remove bad data
    radiology_df = remove_bad_data(radiology_df, output_path)
    
    # Print final length
    final_count = len(radiology_df)
    append_audit("query_clean_rad.remining_rad_records", final_count)
    print(f"Final dataframe length: {len(radiology_df)} rows")
    
    # Save output
    radiology_df.to_csv(f'{output_path}/parsed_radiology.csv', index=False)
    
    return radiology_df
    
    
if __name__ == "__main__":
    rad_df = pd.read_csv(f'{env}/data/raw_radiology.csv')
    output_path = os.path.join(env, "data")

    surgery_file_path = f'{env}/data/raw_surgery.csv'
    if os.path.exists(surgery_file_path):
        surgery_df = pd.read_csv(surgery_file_path)
        print(f"Loaded surgery data with {len(surgery_df)} records")
    else:
        surgery_df = None
        print("No raw_surgery.csv found -- has_breast_implant will be 'unknown'")

    filter_rad_data(rad_df, output_path, surgery_df=surgery_df)