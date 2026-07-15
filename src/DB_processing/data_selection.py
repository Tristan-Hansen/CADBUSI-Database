from config import CONFIG
from src.DB_processing.image_processing import *
from src.DB_processing.tools import append_audit
from src.DB_processing.database import DatabaseManager
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd
import os
from tools.storage_adapter import save_data, list_files
tqdm.pandas()



def find_nearest_images(subset, image_folder_path):
    if len(subset) == 0:
        return {}
    
    idx = subset.index.to_numpy()
    result = {}
    image_pairs_checked = set()

    # All regions have same coordinates - get them once
    coord_cols = ['region_location_min_x0', 'region_location_min_y0', 
                  'region_location_max_x1', 'region_location_max_y1']
    x, y, x1, y1 = subset.iloc[0][coord_cols].astype(int)
    w, h = x1 - x, y1 - y

    # Load and crop all images once
    cropped_images = {}
    for image_id in idx:
        file_name = subset.loc[image_id, 'image_name']
        full_filename = os.path.join(image_folder_path, file_name)
        img = read_image(full_filename, use_pil=True)
        if img is None:
            tqdm.write(f"  [skip] could not read {file_name}")
            continue
        img = np.array(img).astype(np.uint8)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        rows, cols = img.shape[:2]
        if rows >= y + h and cols >= x + w:
            cropped = img[y:y+h, x:x+w]
        else:
            cropped = np.full((h, w), 255, dtype=np.uint8)

        cropped_images[image_id] = cropped.flatten()

    if len(cropped_images) < 2:
        return {}

    image_ids = list(cropped_images.keys())
    image_matrix = np.array([cropped_images[id] for id in image_ids], dtype=np.uint8)
    
    # scipy's cdist is highly optimized - FASTEST OPTION
    distances = cdist(image_matrix, image_matrix, metric='cityblock') / image_matrix.shape[1]
    np.fill_diagonal(distances, 1000)
    
    # Find nearest neighbors
    for j, current_id in enumerate(image_ids):
        if current_id in image_pairs_checked:
            continue
        
        sister_idx = np.argmin(distances[j])
        sister_id = image_ids[sister_idx]
        distance = distances[j, sister_idx]
        
        result[current_id] = {
            'image_filename': subset.at[current_id, 'image_name'],
            'sister_filename': subset.at[sister_id, 'image_name'],
            'distance': distance
        }
        
        if sister_id not in result:
            result[sister_id] = {
                'image_filename': subset.at[sister_id, 'image_name'],
                'sister_filename': subset.at[current_id, 'image_name'],
                'distance': distance
            }
        
        image_pairs_checked.add(current_id)
        image_pairs_checked.add(sister_id)
    
    return result


def process_nearest_given_ids(pid, subset, image_folder_path):
    # EARLY EXIT:
    subset = subset[subset['photometric_interpretation'] != 'RGB']

    # Validate crop coordinates
    invalid_coords = (
        (subset['region_location_max_x1'] <= subset['region_location_min_x0']) |
        (subset['region_location_max_y1'] <= subset['region_location_min_y0'])
    )
    if invalid_coords.any():
        subset = subset[~invalid_coords]

    # Early termination if no valid images
    if len(subset) < 2:
        return subset

    subset = subset.copy()

    # Group by crop coordinates
    coord_cols = ['region_location_min_x0', 'region_location_min_y0', 'region_location_max_x1', 'region_location_max_y1']

    # Create coordinate groups
    subset['coord_key'] = list(zip(
        subset[coord_cols[0]], 
        subset[coord_cols[1]], 
        subset[coord_cols[2]], 
        subset[coord_cols[3]]
    ))
    coordinate_groups = subset.groupby('coord_key')
    
    #print(f"Accession {pid}: {len(subset)} regions split into {len(coordinate_groups)} coordinate groups")
    
    # Collect all updates
    closest_fn_updates = {}
    distance_updates = {}
    
    for coord_key, group_subset in coordinate_groups:
        has_calipers_in_group = group_subset['has_calipers'].any()
        
        if len(group_subset) >= 2 and has_calipers_in_group:
            group_result = find_nearest_images(group_subset, image_folder_path)
            
            # Collect updates instead of applying immediately
            for i, result in group_result.items():
                closest_fn_updates[i] = result['sister_filename']
                distance_updates[i] = result['distance']
    
    # Apply all updates at once
    if closest_fn_updates:
        subset.loc[list(closest_fn_updates.keys()), 'closest_fn'] = list(closest_fn_updates.values())
        subset.loc[list(distance_updates.keys()), 'distance'] = list(distance_updates.values())
    
    subset = subset.drop('coord_key', axis=1)
    return subset





def apply_filters(CONFIG):
    """
    Apply all quality and relevance filters to the image data.
    Excluded images are saved to the BadImages table with exclusion reasons.

    Args:
        CONFIG: Configuration dictionary
    """
    audit_stats = {}
    excluded_images = []  # Track all excluded images

    with DatabaseManager() as db:
        image_df = db.get_images_dataframe()
        breast_df = db.get_study_cases_dataframe()

        # Track initial counts
        audit_stats['init_images'] = len(image_df)
        audit_stats['init_breasts'] = len(breast_df)

        # Filter 0: Remove images that don't exist on disk
        image_folder = f'{CONFIG["DATABASE_DIR"]}/images/'
        disk_files = set(os.path.basename(f) for f in list_files(image_folder))
        missing_mask = ~image_df['image_name'].isin(disk_files)
        missing_images = image_df[missing_mask][['image_name', 'dicom_hash']].copy()
        missing_images['exclusion_reason'] = 'image does not exist'
        excluded_images.append(missing_images)

        image_df = image_df[~missing_mask]
        audit_stats['missing_file_removed'] = len(missing_images)

        # Merge study_laterality from breast_df for laterality filtering logic
        if 'study_laterality' not in image_df.columns:
            image_df = image_df.merge(
                breast_df[['accession_number', 'study_laterality']],
                on='accession_number',
                how='left'
            )

        # Filter 1: Remove images that are too dark
        darkness_thresh = 75
        darkness_values = image_df['darkness'].round(2).tolist()
        append_audit("export.darkness_values", darkness_values)
        append_audit("export.darkness_thresh", darkness_thresh)

        dark_mask = image_df['darkness'] > darkness_thresh
        dark_images = image_df[dark_mask][['image_name', 'dicom_hash']].copy()
        dark_images['exclusion_reason'] = 'too dark (>75)'
        excluded_images.append(dark_images)

        image_df = image_df[~dark_mask]
        audit_stats['too_dark_removed'] = len(dark_images)

        # Filter 2: Remove non-breast images
        # First, fix unknown areas
        image_df.loc[(image_df['area'] == 'unknown') | (image_df['area'].isna()), 'area'] = 'breast'

        non_breast_mask = image_df['area'] != 'breast'
        non_breast_images = image_df[non_breast_mask][['image_name', 'dicom_hash']].copy()
        non_breast_images['exclusion_reason'] = 'non-breast area'
        excluded_images.append(non_breast_images)

        image_df = image_df[~non_breast_mask]
        audit_stats['non_breast_removed'] = len(non_breast_images)

        # Filter 3: Remove images with unknown laterality in BILATERAL studies
        bilateral_unknown_mask = (
            ((image_df['laterality'] == 'unknown') | (image_df['laterality'].isna())) &
            (image_df['study_laterality'].str.upper() == 'BILATERAL')
        )
        lat_images = image_df[bilateral_unknown_mask][['image_name', 'dicom_hash']].copy()
        lat_images['exclusion_reason'] = 'unknown laterality (bilateral study)'
        excluded_images.append(lat_images)

        image_df = image_df[~bilateral_unknown_mask]
        audit_stats['unknown_lat_removed'] = len(lat_images)

        # Filter 4: Remove images with multiple regions
        #multi_region_mask = image_df['region_count'] > 1
        #region_images = image_df[multi_region_mask][['image_name', 'dicom_hash']].copy()
        #region_images['exclusion_reason'] = 'multiple regions'
        #excluded_images.append(region_images)

        #image_df = image_df[~multi_region_mask]
        #audit_stats['multi_region_removed'] = len(region_images)

        # Update has_malignant for all cases (vectorized)
        both_null = breast_df['left_diagnosis'].isna() & breast_df['right_diagnosis'].isna()
        left_mask = breast_df['study_laterality'] == 'LEFT'
        left_malignant = breast_df['left_diagnosis'].fillna('').str.upper().str.contains('MALIGNANT')
        right_mask = breast_df['study_laterality'] == 'RIGHT'
        right_malignant = breast_df['right_diagnosis'].fillna('').str.upper().str.contains('MALIGNANT')

        breast_df.loc[both_null, 'has_malignant'] = -1
        breast_df.loc[left_mask & ~both_null, 'has_malignant'] = left_malignant[left_mask & ~both_null].astype(int)
        breast_df.loc[right_mask & ~both_null, 'has_malignant'] = right_malignant[right_mask & ~both_null].astype(int)

        # Persist has_malignant back to DB
        db.insert_study_cases_batch(
            breast_df[['accession_number', 'has_malignant']].to_dict('records'),
            update_only=True,
        )

        # Keep only images whose patient_id exists in breast_df
        valid_patient_ids = breast_df['patient_id'].unique()
        image_df = image_df[image_df['patient_id'].isin(valid_patient_ids)]

        # Filter 5: Remove images with missing crop region
        missing_crop_mask = image_df['crop_x'].isna()
        missing_crop_images = image_df[missing_crop_mask][['image_name', 'dicom_hash']].copy()
        missing_crop_images['exclusion_reason'] = 'missing_crop_region'
        excluded_images.append(missing_crop_images)

        image_df = image_df[~missing_crop_mask]
        audit_stats['missing_crop_removed'] = len(missing_crop_images)

        # Filter 6: Remove bad aspect ratios
        min_aspect_ratio = CONFIG.get('MIN_ASPECT_RATIO', 0.5)
        max_aspect_ratio = CONFIG.get('MAX_ASPECT_RATIO', 4.0)

        bad_aspect_mask = (
            (image_df['crop_aspect_ratio'] < min_aspect_ratio) |
            (image_df['crop_aspect_ratio'] > max_aspect_ratio)
        )
        aspect_images = image_df[bad_aspect_mask][['image_name', 'dicom_hash']].copy()
        aspect_images['exclusion_reason'] = f'bad aspect ratio ({min_aspect_ratio}-{max_aspect_ratio})'
        excluded_images.append(aspect_images)

        image_df = image_df[~bad_aspect_mask]
        audit_stats['bad_aspect_removed'] = len(aspect_images)

        # Filter 7: Remove images that are too small
        min_dimension = CONFIG.get('MIN_DIMENSION', 200)

        too_small_mask = (
            (image_df['crop_w'] < min_dimension) |
            (image_df['crop_h'] < min_dimension)
        )
        small_images = image_df[too_small_mask][['image_name', 'dicom_hash']].copy()
        small_images['exclusion_reason'] = f'too small (<{min_dimension}px)'
        excluded_images.append(small_images)

        image_df = image_df[~too_small_mask]
        audit_stats['too_small_removed'] = len(small_images)

        # Track final usable images
        audit_stats['usable_images'] = len(image_df)

        # Save excluded images to BadImages table
        excluded_df = pd.concat(excluded_images, ignore_index=True)
        bad_data = excluded_df[['image_name', 'dicom_hash', 'exclusion_reason']].to_dict('records')
        db.insert_bad_data_batch(bad_data)

        print(f"Saved {len(bad_data)} excluded images to BadImages table")

        # Log audit statistics
        for key, value in audit_stats.items():
            append_audit(f"export.{key}", value)

def Select_Data(database_path):
    with DatabaseManager() as db:
        image_folder_path = f"{database_path}/images/"

        # Load data from database
        db_out = db.get_images_dataframe()
        breast_df = db.get_study_cases_dataframe()

        db_to_process = db_out
        columns_to_update = ['image_name', 'label', 'crop_aspect_ratio', 'closest_fn', 'distance', 'exclusion_reason']
        accession_ids = db_to_process['accession_number'].unique()

        db_to_process['closest_fn'] = '' 
        db_to_process['distance'] = 99999

        grouped_by_accession = db_to_process.groupby('accession_number')
        all_results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor, tqdm(total=len(accession_ids)) as progress:
            futures = {
                executor.submit(
                    process_nearest_given_ids, 
                    pid, 
                    grouped_by_accession.get_group(pid),  # Pass pre-filtered subset
                    image_folder_path
                ): pid for pid in accession_ids
            }

            # Collect results instead of updating immediately
            for future in as_completed(futures):
                result = future.result()
                if result is not None and not result.empty:
                    all_results.append(result)  # Add to list
                progress.update()

        # Update once after all processing is complete
        if all_results:
            updated_df = pd.concat(all_results, ignore_index=False)
            db_to_process.update(updated_df)

        # Build CaliperPairs from near-duplicate pairs (distance <= 5)
        caliper_lookup = db_to_process.set_index('image_name')['has_calipers']

        near_dupes = db_to_process[
            (db_to_process['distance'] <= 5) &
            (db_to_process['closest_fn'] != '')
        ].copy()

        # Deduplicate pairs (A,B) and (B,A)
        near_dupes['pair_key'] = near_dupes.apply(
            lambda r: tuple(sorted([r['image_name'], r['closest_fn']])), axis=1
        )
        near_dupes = near_dupes.drop_duplicates(subset='pair_key')

        # Vectorized sister lookup
        near_dupes['sister_has_calipers'] = near_dupes['closest_fn'].map(caliper_lookup)
        near_dupes['img_has'] = near_dupes['has_calipers'] == 1
        near_dupes['sis_has'] = near_dupes['sister_has_calipers'] == 1

        # Keep only pairs where exactly one has calipers
        valid_pairs = near_dupes[near_dupes['img_has'] != near_dupes['sis_has']]

        clean_pairs = pd.DataFrame({
            'caliper_image_name': valid_pairs.apply(
                lambda r: r['image_name'] if r['img_has'] else r['closest_fn'], axis=1
            ),
            'clean_image_name': valid_pairs.apply(
                lambda r: r['closest_fn'] if r['img_has'] else r['image_name'], axis=1
            )
        })
        clean_pairs = clean_pairs.drop_duplicates(subset='caliper_image_name')

        if not clean_pairs.empty:
            caliper_pairs = clean_pairs.to_dict('records')
            db.insert_caliper_pairs_batch(caliper_pairs)
            print(f"Saved {len(caliper_pairs)} caliper/clean pairs to CaliperPairs table")

        # Merge study_laterality from breast_df for laterality filtering logic
        db_to_process = db_to_process.merge(
            breast_df[['accession_number', 'study_laterality']],
            on='accession_number',
            how='left'
        )

        # Check the aspect ratio of the crop region
        db_to_process['crop_aspect_ratio'] = (db_to_process['crop_w'] / db_to_process['crop_h']).round(2)
        
        total_caliper_images = len(db_to_process[db_to_process['has_calipers'] == 1])
        append_audit("image_processing.total_caliper_images", total_caliper_images)
        caliper_with_duplicates = len(db_to_process[(db_to_process['has_calipers'] == 1) & (db_to_process['distance'] <= 5)])
        append_audit("image_processing.caliper_with_duplicates", caliper_with_duplicates)
        total_near_duplicates = len(db_to_process[db_to_process['distance'] <= 5]) 
        append_audit("image_processing.total_near_duplicates", total_near_duplicates)

        # Convert DataFrame to list of dicts for batch insert
        update_data = db_to_process[columns_to_update].to_dict('records')
        
        # Use batch update for existing records (more efficient than upsert)
        db.insert_images_batch(update_data, update_only=True)
        
        print(f"Updated {len(db_to_process)} images in database")
    
    # Create Exclusion Data
    apply_filters(CONFIG)
        