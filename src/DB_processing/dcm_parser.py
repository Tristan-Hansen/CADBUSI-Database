import os, pydicom
import sqlite3
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd
import io
from tqdm import tqdm
import warnings, logging, cv2
from functools import lru_cache
from PIL import Image
import time
import re
from tools.storage_adapter import *
from src.encrypt_keys import *
from src.DB_processing.tools import append_audit
from src.DB_processing.database import DatabaseManager
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')
env = os.path.dirname(os.path.abspath(__file__))

try:
    import gdcm
    GDCM_AVAILABLE = True
    print("GDCM loaded - enhanced DICOM decompression available")
except ImportError:
    GDCM_AVAILABLE = False
    print("GDCM not available - compressed DICOM handling may fail")


ENCRYPTION_KEY = None


# Define sets at module level for better performance
NAMES_TO_REMOVE = frozenset({
    'SOP Instance UID',
    'Study Instance UID', 'Series Instance UID', 'Private Creator',
    'Media Storage SOP Instance UID', 'Implementation Class UID',
    "Patient's Name", "Referring Physician's Name", "Acquisition DateTime",
    "Institution Name", "Station Name", "Physician(s) of Record",
    "Referenced SOP Class UID", "Referenced SOP Instance UID",
    "Device Serial Number", "Patient Comments", "Issuer of Patient ID",
    "Study ID", "Study Comments", "Current Patient Location",
    "Requested Procedure ID", "Performed Procedure Step ID",
    "Other Patient IDs", "Operators' Name", "Institutional Department Name",
    "Manufacturer", "Requesting Physician",
})

NAMES_TO_ANON_TIME = frozenset({
    'Study Time', 'Series Time', 'Content Time', 'Acquisition Time',
})

# Machine-specific coordinates (as ratios of width, height) to check for color
# Structure: {'Machine Name': {region_count: (x_ratio, y_ratio)}}
MACHINE_COLOR_CHECK_COORDS = {
    'HDI 5000': {
        1: (0.9734, 0.3067),
    },
    'LOGIQE9': {
        1: (0.0219, 0.5278),
        2: (0.0135, 0.4662),
    },
    'LOGIQS8': {
        1: (0.0266, 0.4696),
        2: (0.0117, 0.4708),
    },
    'LOGIQE10': {
        1: (0.0335, 0.45),
        2: (0.0103, 0.45),
    },
    'EPIQ 5G': {
        1: (0.9785, 0.2344),
        2: (0.9932, 0.1888),
    },
    'EPIQ 7G': {
        1: (0.9765, 0.2253),
        2: (0.4912, 0.1940),
    },
    'EPIQ Elite': {
        1: (0.9765, 0.2201),
        2: (0.9893, 0.1836),
    },
    'S3000': {
        2: (0.0049, 0.2747),
    },
    'RS85': {
        1: (0.9703, 0.1747),
    },
    'CX50': {
        1: (0.9625, 0.1917),
    },
    'iU22': {
        1: (0.9697, 0.2148),
    },
    'TUS-AI800': {
        1: (0.9531, 0.2187),
    },
    'TUS-A500': {
        1: (0.0219, 0.1833),
    },
}


# Optimized vectorized color detection using numba for JIT compilation
from numba import jit

@jit(nopython=True)
def is_pixel_colored(image, x, y, min_channel_diff=15):
    """Check if a specific pixel has color (RGB channels differ significantly)"""
    b = image[y, x, 0]
    g = image[y, x, 1]
    r = image[y, x, 2]

    # Check if channels differ significantly (not grayscale)
    max_val = max(r, g, b)
    min_val = min(r, g, b)

    # Has color if: channels differ AND not too dark
    if max_val - min_val >= min_channel_diff:
        return True
    return False

@jit(nopython=True)
def has_blue_pixels(image, n=100, min_b=200):
    height, width = image.shape[:2]
    for i in range(0, height, 10):  # Sample every 10th pixel for speed
        for j in range(0, width, 10):
            b, g, r = image[i, j, 2], image[i, j, 1], image[i, j, 0]
            if b >= min_b and b - r >= n and b - g >= n:
                return True
    return False

@jit(nopython=True)
def has_red_pixels(image, n=100, min_r=200):
    height, width = image.shape[:2]
    for i in range(0, height, 10):  # Sample every 10th pixel for speed
        for j in range(0, width, 10):
            r, g, b = image[i, j, 0], image[i, j, 1], image[i, j, 2]
            if r >= min_r and r - b >= n and r - g >= n:
                return True
    return False


ACQUISITION_TIME_TAG_FALLBACK = ('AcquisitionTime', 'ContentTime', 'SeriesTime', 'StudyTime')


def format_dicom_tm(tm):
    """Format DICOM TM 'HHMMSS[.FFFFFF]' as 'HH:MM:SS[.fff]'. Returns '' if unparseable."""
    if not tm:
        return ''
    s = str(tm).strip()
    if '.' in s:
        hms, frac = s.split('.', 1)
    else:
        hms, frac = s, ''
    if len(hms) < 6 or not hms[:6].isdigit():
        return ''
    h, m, sec = hms[0:2], hms[2:4], hms[4:6]
    if frac:
        ms = (frac + '000')[:3]
        return f"{h}:{m}:{sec}.{ms}"
    return f"{h}:{m}:{sec}"


def pick_acquisition_time(ds):
    """Try AcquisitionTime → ContentTime → SeriesTime → StudyTime, return (formatted_str, source_tag).
    Returns ('', '') if nothing parseable."""
    for tag in ACQUISITION_TIME_TAG_FALLBACK:
        raw = getattr(ds, tag, '') or ''
        if raw:
            formatted = format_dicom_tm(str(raw))
            if formatted:
                return formatted, tag
    return '', ''


def anon_callback(ds, element):
    # Use faster membership testing with frozensets
    if element.name in NAMES_TO_REMOVE:
        del ds[element.tag]
    elif element.VR == "DA":
        element.value = element.value[:4] + "0101"
    elif element.VR == "TM" and element.name not in NAMES_TO_ANON_TIME:
        element.value = "000000"


'''
def manual_decompress(ds):
    """
    Replacement for ds.decompress() for specific edge cases where normal decompression fails
    
    Parameters:
    - ds: pydicom Dataset that needs decompression
    
    Returns:
    - Modified dataset with uncompressed pixel data, or None if decompression fails
    """
    # Check if dataset has pixel data
    if 'PixelData' not in ds:
        print("No pixel data found in dataset")
        return None
    
    pixel_data = ds.PixelData
    
    # Find all JPEG frame start positions
    frame_starts = []
    pos = 0
    while pos < len(pixel_data) - 1:
        if pixel_data[pos] == 0xFF and pixel_data[pos+1] == 0xD8:
            frame_starts.append(pos)
            pos += 2
        else:
            pos += 1
    
    # Return None if no frames found
    if not frame_starts:
        print("No JPEG frames found in pixel data")
        return None
    
    # Get image dimensions from DICOM attributes
    rows = ds.Rows if 'Rows' in ds else 0
    columns = ds.Columns if 'Columns' in ds else 0
    
    # Check if we have enough information
    if rows == 0 or columns == 0:
        print("Missing image dimensions in DICOM")
        return None
    
    # Determine if multi-frame
    is_multi_frame = False
    if 'NumberOfFrames' in ds and ds.NumberOfFrames > 1:
        is_multi_frame = True

    # Process each frame
    frames = []
    for i in range(len(frame_starts)):
        # Calculate frame size
        start = frame_starts[i]
        end = frame_starts[i+1] if i < len(frame_starts)-1 else len(pixel_data)
        
        # Get frame data
        frame_data = pixel_data[start:end]
        
        # Ensure it ends with JPEG EOI marker
        jpeg_end = frame_data.rfind(b'\xFF\xD9')
        if jpeg_end > 0:
            frame_data = frame_data[:jpeg_end+2]
        
        # Skip invalid frames
        if frame_data[:2] != b'\xFF\xD8':
            continue
        
        try:
            # Open JPEG with PIL
            img = Image.open(io.BytesIO(frame_data))
            
            # Ensure dimensions match
            if img.width != columns or img.height != rows:
                # Resize if needed
                img = img.resize((columns, rows))
            
            # Convert to numpy array
            frame_array = np.array(img)
            
            # Append to frames list
            frames.append(frame_array)
        except Exception as e:
            print(f"Error processing frame {i+1}: {e}")
    
    # Return None if no frames were successfully decoded
    if not frames:
        print("No frames could be successfully decompressed")
        return None
    
    # Create uncompressed pixel data
    if is_multi_frame:
        # Stack frames for multi-frame
        pixel_array = np.stack(frames)
    else:
        # Single frame
        pixel_array = frames[0]
    
    # Update dataset
    try:
        # Calculate pixel representation based on array
        if pixel_array.dtype.kind == 'u':  # unsigned integer
            ds.PixelRepresentation = 0
        else:  # signed integer
            ds.PixelRepresentation = 1
        
        # Update bits stored if needed
        max_value = np.max(pixel_array)
        if max_value > 255:
            ds.BitsStored = 16
        else:
            ds.BitsStored = 8
        
        # Update high bit
        ds.HighBit = ds.BitsStored - 1
        
        # Update photometric interpretation if needed
        if len(pixel_array.shape) >= 3 and pixel_array.shape[-1] == 3:
            ds.PhotometricInterpretation = "RGB"
        else:
            # Default to MONOCHROME2 for grayscale
            if 'PhotometricInterpretation' not in ds:
                ds.PhotometricInterpretation = "MONOCHROME2"
        
        # Set uncompressed transfer syntax
        if hasattr(ds, 'file_meta') and ds.file_meta is not None:
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        else:
            ds.file_meta = pydicom.dataset.FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID if 'SOPClassUID' in ds else '1.2.840.10008.5.1.4.1.1.4'  # Default to MR
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID if 'SOPInstanceUID' in ds else pydicom.uid.generate_uid()
            ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
            ds.file_meta.ImplementationVersionName = 'MANUAL_DECOMPRESS'
        
        # Set pixel data from array
        ds.PixelData = pixel_array.tobytes()
        
        return ds
    except Exception as e:
        print(f"Error updating dataset: {e}")
        return None

'''
    
def deidentify_dicom(ds, is_video, is_secondary):
    
    global ENCRYPTION_KEY
    
    ds.PatientID = encrypt_single_id(ENCRYPTION_KEY, ds.PatientID)
    ds.AccessionNumber = encrypt_single_id(ENCRYPTION_KEY, ds.AccessionNumber)

    ds.remove_private_tags()  # take out private tags added by notion or otherwise
    
    # Avoid separate walks by combining them
    ds.walk(anon_callback)
    # Only walk file_meta if it exists
    if hasattr(ds, 'file_meta') and ds.file_meta is not None:
        ds.file_meta.walk(anon_callback)

    y0 = 101
    
    if not is_secondary and (0x0018, 0x6011) in ds:
            y0 = ds['SequenceOfUltrasoundRegions'][0]['RegionLocationMinY0'].value

    if 'OriginalAttributesSequence' in ds:
        del ds.OriginalAttributesSequence

    # Check if Pixel Data is compressed - safely check TransferSyntaxUID
    is_compressed = False
    if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
        is_compressed = ds.file_meta.TransferSyntaxUID.is_compressed
    elif (0x0028, 0x2110) in ds and ds[0x0028, 0x2110].value == '01':
        is_compressed = True
    else:
        print("couldn't determine if compressed")
        return None

    # Attempt to decompress if needed
    if is_compressed:
        try:
            ds.decompress()
        except Exception as e:
            print(f"GDCM decompression failed: {e}")
            try:
                arr = ds.pixel_array  # Sometimes this works even when decompress() fails
                print("Got pixel data despite decompression failure")
            except Exception as e2:
                print(f"Cannot access pixel data: {e2}")
                return None
        
    # crop patient info above US region 
    arr = ds.pixel_array
    
    if is_video:
        arr[:,:y0] = 0
    else:
        arr[:y0] = 0
    
    # Update the Pixel Data
    ds.PixelData = arr.tobytes()
    
    # Keep the original transfer syntax
    ds.file_meta.TransferSyntaxUID = ds.file_meta.TransferSyntaxUID

    return ds



def parse_video_data(dcm, dataset, current_index, parsed_database, video_n_frames, birads_4_accessions=None):

    if video_n_frames <= 0:
        return None

    # Check if we should filter by BI-RADS 4 for videos
    if birads_4_accessions is not None:
        accession = str(dataset.AccessionNumber).strip() if hasattr(dataset, 'AccessionNumber') else None
        if accession and accession not in birads_4_accessions:
            return None  # Skip videos that are not BI-RADS 4
    
    # Extract Software Version
    software_version = ""
    if hasattr(dataset, 'SoftwareVersions'):
        software_version = dataset.SoftwareVersions
    manufacturer_model = ""
    if hasattr(dataset, 'ManufacturerModelName'):
        manufacturer_model = dataset.ManufacturerModelName
        
    data_dict = {}
    region_data_types = []

    for elem in dataset:
        if elem.VR == "SQ":
            # Collect RegionDataType from each region
            if elem.keyword == 'SequenceOfUltrasoundRegions':
                for region in elem:
                    for sub in region:
                        if pydicom.datadict.keyword_for_tag(sub.tag) == "RegionDataType":
                            region_data_types.append(str(sub.value))

            if elem.value and len(elem.value) > 0:
                for sub_elem in elem.value[0]:  # only take the first item in the sequence
                    tag_name = pydicom.datadict.keyword_for_tag(sub_elem.tag)
                    if tag_name == "PixelData":
                        continue
                    data_dict[tag_name] = str(sub_elem.value)
        else:
            tag_name = pydicom.datadict.keyword_for_tag(elem.tag)
            if tag_name == "PixelData":
                continue
            data_dict[tag_name] = str(elem.value)

    #create video folder
    dicom_hash = os.path.splitext(os.path.basename(dcm))[0]
    video_path = dicom_hash
    make_dirs(f"{parsed_database}/videos/{video_path}/")
    
    #get image frames
    image_count = 0
    
   # Get total number of frames
    total_frames = getattr(dataset, 'NumberOfFrames', 1)

    # Extract frames directly from dataset using pydicom
    for i in range(0, total_frames, video_n_frames):  # Process every nth frame
        # Use pydicom to extract the frame
        if hasattr(dataset, 'pixel_array'):
            if total_frames > 1:
                frame = dataset.pixel_array[i]
            else:
                frame = dataset.pixel_array

            # Convert to grayscale if the frame is not already grayscale
            if len(frame.shape) == 3:  # if the frame has 3 channels
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

            frame_name = f"{image_count}.png"

            # Only increment count if frame save succeeds
            if save_data(frame, f"{parsed_database}/videos/{video_path}/{frame_name}"):
                image_count += 1
            else:
                print(f"Failed to save video frame {video_path}/{frame_name}")

    # Only add metadata if at least one frame was saved successfully
    if image_count == 0:
        print(f"No video frames saved successfully for {video_path}, skipping metadata")
        return None

    # Add custom data
    data_dict['DataType'] = 'video'
    data_dict['FileName'] = os.path.join(os.path.basename(os.path.dirname(dcm)), os.path.basename(dcm))
    data_dict['ImagesPath'] = video_path
    data_dict['SavedFrames'] = image_count
    data_dict['DicomHash'] = dicom_hash
    data_dict['RegionDataType'] = ','.join(region_data_types) if region_data_types else ''
    data_dict['SoftwareVersions'] = str(software_version)
    data_dict['ManufacturerModelName'] = str(manufacturer_model)
    data_dict['AcquisitionTime'], _ = pick_acquisition_time(dataset)

    return data_dict



def parse_single_dcm(dcm, current_index, parsed_database, video_n_frames, birads_4_accessions=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Read the DICOM file
            dcm_data = read_binary(dcm)
            dataset = pydicom.dcmread(io.BytesIO(dcm_data), force=True)
            break  # Success, exit retry loop
        except Exception as e:
            if "Checksum mismatch" in str(e) and attempt < max_retries - 1:
                print(f"Checksum error on {dcm}, retrying... (attempt {attempt + 1})")
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                print(f"Failed to read {dcm} after {attempt + 1} attempts: {e}")
                return None
    
    # Not a ultrasound image, likely a image of the settings or some sketch notes
    if not hasattr(dataset, 'SequenceOfUltrasoundRegions'):
        return None
    
    # Safely check for Multi-frame content
    is_video = False
    is_secondary = False
    try:
        if hasattr(dataset, 'file_meta') and 0x00020002 in dataset.file_meta:
            media_type = dataset.file_meta[0x00020002]
            is_video = str(media_type).find('Multi-frame') > -1
            is_secondary = 'Secondary' in str(media_type)
        # Additional check for multi-frame files
        elif hasattr(dataset, 'NumberOfFrames') and dataset.NumberOfFrames > 1:
            is_video = True
        # Method 3: Check SOP Class UID (backup method)
        elif hasattr(dataset, 'SOPClassUID') and 'Multi-frame' in str(dataset.SOPClassUID):
            is_video = True
    except Exception as e:
        print(f"Error determining media type: {e}")
        
    if (is_video and video_n_frames == 0):
        return None
    
    # Anonymize 
    dataset = deidentify_dicom(dataset, is_video, is_secondary)
    
    if (dataset is None):
        return None
    
    # Safely check for Multi-frame content
    if is_video:
        return parse_video_data(dcm, dataset, current_index, parsed_database, video_n_frames, birads_4_accessions)

    
    # Extract Software Version
    software_version = ""
    if hasattr(dataset, 'SoftwareVersions'):
        software_version = dataset.SoftwareVersions
    manufacturer_model = ""
    if hasattr(dataset, 'ManufacturerModelName'):
        manufacturer_model = dataset.ManufacturerModelName
    
    # Continue with image
    data_dict = {}
    region_count = 0
    region_data_types = []

    for elem in dataset:
        # Check if element is a sequence
        if elem.VR == "SQ":
            # Count regions and collect RegionDataType from each
            if elem.keyword == 'SequenceOfUltrasoundRegions':
                for region in elem:
                    region_count += 1
                    for sub in region:
                        if pydicom.datadict.keyword_for_tag(sub.tag) == "RegionDataType":
                            region_data_types.append(str(sub.value))

            #Get Data
            if elem.value and len(elem.value) > 0:
                for sub_elem in elem.value[0]:  # only take the first item in the sequence
                    tag_name = pydicom.datadict.keyword_for_tag(sub_elem.tag)
                    if tag_name == "PixelData":
                        continue
                    data_dict[tag_name] = str(sub_elem.value)

        else:
            tag_name = pydicom.datadict.keyword_for_tag(elem.tag)
            if tag_name == "PixelData":
                continue
            data_dict[tag_name] = str(elem.value)
    
    # get image data
    im = dataset.pixel_array
    if data_dict.get('PhotometricInterpretation', '') == 'RGB':
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        np_im = np.array(im)


        # Check if there are any colored pixels (indicating Doppler flow)
        # Use machine-specific coordinates if available
        is_doppler = False  # Initialize before conditional
        if manufacturer_model in MACHINE_COLOR_CHECK_COORDS:
            machine_coords = MACHINE_COLOR_CHECK_COORDS[manufacturer_model]
            coords = None
            
            if region_count in machine_coords:
                coords = machine_coords[region_count]

            if coords is not None:
                height, width = np_im.shape[:2]
                x_ratio, y_ratio = coords
                x = int(width * x_ratio)
                y = int(height * y_ratio)
                is_doppler = is_pixel_colored(np_im, x, y)
                
        # check if there is any blue pixel
        if not is_doppler and has_red_pixels(np_im) and has_blue_pixels(np_im):
            is_doppler = True
            

        if is_doppler:
            im = cv2.cvtColor(np_im, cv2.COLOR_BGR2RGB)
        else:
            # Convert yellow pixels to white and convert to grayscale
            yellow = [255, 255, 0]  # RGB values for yellow
            mask = np.all(np_im == yellow, axis=-1)
            np_im[mask] = [255, 255, 255]
            im = cv2.cvtColor(np_im, cv2.COLOR_BGR2GRAY)
            data_dict['PhotometricInterpretation'] = 'MONOCHROME2_OVERRIDE'

    dicom_hash = os.path.splitext(os.path.basename(dcm))[0]
    image_name = f"{dicom_hash}.png"

    # Only add metadata if image save succeeds
    if not save_data(im, f"{parsed_database}/images/{image_name}"):
        print(f"Failed to save image {image_name}, skipping metadata")
        return None

    # Add custom data
    data_dict['DataType'] = 'image'
    data_dict['FileName'] = os.path.join(os.path.basename(os.path.dirname(dcm)), os.path.basename(dcm))
    data_dict['ImageName'] = image_name
    data_dict['DicomHash'] = dicom_hash
    data_dict['RegionCount'] = region_count
    data_dict['RegionDataType'] = ','.join(region_data_types) if region_data_types else ''
    data_dict['SoftwareVersions'] = str(software_version)
    data_dict['ManufacturerModelName'] = str(manufacturer_model)
    data_dict['AcquisitionTime'], _ = pick_acquisition_time(dataset)

    return data_dict


def process_batch(batch_data):
    """Process a batch of DICOM files"""
    results = []
    for dcm, current_index, parsed_database, video_n_frames, birads_4_accessions in batch_data:
        try:
            result = parse_single_dcm(dcm, current_index, parsed_database, video_n_frames, birads_4_accessions)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Error processing {dcm}: {e}")
    return results

def parse_files(CONFIG, dcm_files_list, database_path, birads_4_accessions=None, batch_size=100):
    """Optimized parsing with batching and process pooling"""
    print("Parsing DCM Data")
    video_n_frames = CONFIG["VIDEO_SAMPLING"]

    # Check if there are no new files to process
    if len(dcm_files_list) == 0:
        raise ValueError("No *new* DICOM files found to process.")

    # Prepare batches
    batches = []
    for i in range(0, len(dcm_files_list), batch_size):
        batch = []
        for j, dcm in enumerate(dcm_files_list[i:i+batch_size]):
            batch.append((dcm, i+j, database_path, video_n_frames, birads_4_accessions))
        batches.append(batch)
    
    # Process batches in parallel using ProcessPoolExecutor
    data_list = []
    failure_counter = 0
    
    # Use more workers for CPU-bound tasks
    num_workers = min(32, multiprocessing.cpu_count())
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DICOM files"):
            try:
                batch_results = future.result()
                data_list.extend(batch_results)
                # Count failures in batch
                batch_size_actual = len(futures[future])
                failure_counter += batch_size_actual - len(batch_results)
            except Exception as exc:
                print(f'Batch processing exception: {exc}')
                failure_counter += len(futures[future])
    
    if failure_counter > 0:
        print(f'Skipped {failure_counter} DICOMS from missing data or irrelevant data')
    
    append_audit("dicom_parsing.failed_dicoms", failure_counter)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    return df

def to_snake_case(name):
    """Convert any naming convention to snake_case (hyphens become underscores)"""
    # Replace hyphens with underscores first
    name = name.replace('-', '_')
    
    # Handle UPPERCASE -> snake_case
    if name.isupper():
        return name.lower()
    
    # If already contains underscores, just lowercase it
    if '_' in name:
        return name.lower()
    
    # Handle CamelCase/PascalCase -> snake_case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def parse_anon_file(anon_location, image_df):
    """
    Parse and insert data into database.

    Args:
        anon_location: Path to anonymized CSV file
        image_df: DataFrame with image data (ignored if load_from_checkpoint=True)
        load_from_checkpoint: If True, loads processed data from checkpoint instead of using image_df
    """


    # Original processing logic
    # Convert all DataFrame columns to snake_case immediately
    image_df.columns = [to_snake_case(col) for col in image_df.columns]

    # Split into video and image dataframes
    video_df = image_df[image_df['data_type'] == 'video']
    image_df = image_df[image_df['data_type'] == 'image']

    # Define common columns (all snake_case now)
    common_columns = [
        'patient_id', 'accession_number', 'region_spatial_format', 'region_data_type',
        'region_location_min_x0', 'region_location_min_y0', 'region_location_max_x1',
        'region_location_max_y1', 'photometric_interpretation', 'rows', 'columns',
        'file_name', 'dicom_hash', 'software_versions', 'manufacturer_model_name',
        'physical_delta_x', 'acquisition_time'
    ]

    # Keep only necessary columns from dataframes
    if not video_df.empty:
        video_df = video_df[common_columns + ['images_path', 'saved_frames']]

    image_df = image_df[common_columns + ['image_name', 'region_count']]

    # Read CSV and convert columns to snake_case, preserving leading zeros
    anon_location = os.path.normpath(anon_location)
    breast_csv = pd.read_csv(anon_location, dtype={'PATIENT_ID': str, 'ACCESSION_NUMBER': str})
    breast_csv.columns = [to_snake_case(col) for col in breast_csv.columns]
    breast_csv = breast_csv.sort_values('patient_id')

    # Strip whitespace and ensure strings
    image_df[['patient_id', 'accession_number']] = image_df[['patient_id', 'accession_number']].astype(str)
    breast_csv['patient_id'] = breast_csv['patient_id'].astype(str).str.strip()
    breast_csv['accession_number'] = breast_csv['accession_number'].astype(str).str.strip()
    if not video_df.empty:
        video_df[['patient_id', 'accession_number']] = video_df[['patient_id', 'accession_number']].astype(str)

    # Filter to only keep rows where patient_id has at least some images
    # Get unique patient IDs that have images
    image_patient_ids = set(image_df['patient_id'].unique())
    if not video_df.empty:
        video_patient_ids = set(video_df['patient_id'].unique())
        all_patient_ids_with_images = image_patient_ids.union(video_patient_ids)
    else:
        all_patient_ids_with_images = image_patient_ids

    breast_patient_ids = set(breast_csv['patient_id'].unique())

    # Keep ALL rows from breast_csv for any patient_id that has at least one image
    matching_patient_ids = all_patient_ids_with_images.intersection(breast_patient_ids)

    # Filter breast_csv to only patients with images (keeps all rows for those patients)
    breast_csv = breast_csv[breast_csv['patient_id'].isin(matching_patient_ids)]

    # Filter image/video data to only patients in breast_csv
    image_df = image_df[image_df['patient_id'].isin(matching_patient_ids)]
    if not video_df.empty:
        video_df = video_df[video_df['patient_id'].isin(matching_patient_ids)]

    total_breast_patients = len(breast_patient_ids)
    non_matching_breast_patients = len(breast_patient_ids - matching_patient_ids)
    percentage_without_images = (non_matching_breast_patients / total_breast_patients) * 100 if total_breast_patients > 0 else 0
    print(f"{percentage_without_images:.1f}% of breast patients did not have images")

    # Populate has_malignant and has_benign based on left_diagnosis and right_diagnosis
    # Check if either column contains MALIGNANT or BENIGN diagnosis
    breast_csv['has_malignant'] = (
        breast_csv['left_diagnosis'].str.contains('MALIGNANT', na=False) |
        breast_csv['right_diagnosis'].str.contains('MALIGNANT', na=False)
    )
    breast_csv['has_benign'] = (
        breast_csv['left_diagnosis'].str.contains('BENIGN', na=False) |
        breast_csv['right_diagnosis'].str.contains('BENIGN', na=False)
    )

    # CHECKPOINT: Save processed dataframes for quick reprocessing
    checkpoint_dir = os.path.join(os.path.dirname(anon_location), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, 'processed_data.pkl')

    print(f"Saving checkpoint to {checkpoint_file}...")
    import pickle
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            'breast_csv': breast_csv,
            'image_df': image_df,
            'video_df': video_df
        }, f)
    print(f"Checkpoint saved! Use load_from_checkpoint=True to load from checkpoint.")

    # Use DatabaseManager to save to SQLite
    with DatabaseManager() as db:
        # Create schema if it doesn't exist
        db.create_schema()

        # Check existing patient IDs
        existing_patient_ids = db.check_existing_patient_ids()

        # Filter out images from existing patients
        image_df['patient_id'] = image_df['patient_id'].astype(str)
        image_df_new = image_df[~image_df['patient_id'].isin(existing_patient_ids)]

        # Insert study cases (breast data) - dict keys now match DB columns exactly
        study_data = breast_csv.to_dict('records')
        inserted_studies = db.insert_study_cases_batch(study_data)

        # Insert images - dict keys match DB columns
        if not image_df_new.empty:
            # Get all accession numbers that exist in StudyCases (both from breast_csv and already in DB)
            valid_accession_numbers = set(breast_csv['accession_number'].unique())
            existing_accessions = db.get_existing_accession_numbers()
            valid_accession_numbers.update(existing_accessions)

            # Filter images to only those with valid accession numbers
            before_count = len(image_df_new)
            image_df_new = image_df_new[image_df_new['accession_number'].isin(valid_accession_numbers)]
            after_count = len(image_df_new)

            if before_count > after_count:
                print(f"Filtered out {before_count - after_count} images with missing accession numbers in StudyCases")

            if not image_df_new.empty:
                image_data = image_df_new.to_dict('records')
                inserted_images = db.insert_images_batch(image_data)
            else:
                inserted_images = 0
        else:
            inserted_images = 0

        # Insert videos - dict keys match DB columns
        if not video_df.empty:
            # Filter out videos from existing patients (similar to images)
            video_df['patient_id'] = video_df['patient_id'].astype(str)
            video_df_new = video_df[~video_df['patient_id'].isin(existing_patient_ids)]

            if not video_df_new.empty:
                # Filter videos to only those with valid accession numbers
                before_count = len(video_df_new)
                video_df_new = video_df_new[video_df_new['accession_number'].isin(valid_accession_numbers)]
                after_count = len(video_df_new)

                if before_count > after_count:
                    print(f"Filtered out {before_count - after_count} videos with missing accession numbers in StudyCases")

                if not video_df_new.empty:
                    video_data = video_df_new.to_dict('records')
                    inserted_videos = db.insert_videos_batch(video_data)
                else:
                    inserted_videos = 0
            else:
                inserted_videos = 0
        else:
            inserted_videos = 0

        # Update metadata for images/videos from study cases
        db.update_image_metadata_from_studies()
        db.extract_metadata_from_filenames()

        # Get total counts from database
        total_images = len(db.get_images_dataframe())
        total_videos = len(db.get_videos_dataframe())
        total_studies = len(db.get_study_cases_dataframe())

        print(f"Inserted {inserted_images} new images (Total: {total_images})")
        print(f"Inserted {inserted_videos} new videos (Total: {total_videos})")
        print(f"Inserted {inserted_studies} study cases (Total: {total_studies})")

        append_audit("dicom_parsing.images_success", total_images)
        append_audit("dicom_parsing.video_success", total_videos)
        append_audit("dicom_parsing.breast_success", total_studies)
    
def deduplicate_dcm_files(dcm_files_list):
    """Remove duplicate files based on DicomHash (filename without extension)"""
    seen_hashes = set()
    unique_files = []
    duplicate_count = 0

    for dcm_file in dcm_files_list:
        # Generate the same hash that would be created during processing
        dicom_hash = os.path.splitext(os.path.basename(dcm_file))[0]

        if dicom_hash not in seen_hashes:
            seen_hashes.add(dicom_hash)
            unique_files.append(dcm_file)
        else:
            duplicate_count += 1

    print(f'Removed {duplicate_count} duplicate DICOM files')

    return unique_files

def get_existing_study_pairs(db_path):
    """Return the set of (patient_id, accession_number) pairs already stored in
    StudyCases of the given cadbusi.db. Used by incremental --existing-db mode
    to skip DICOMs whose case is already parsed. Read-only; returns empty set if
    db_path is falsy or the table is missing."""
    if not db_path:
        return set()
    if not os.path.exists(db_path):
        print(f"WARNING: --existing-db path does not exist: {db_path}")
        return set()
    try:
        conn = sqlite3.connect(f'file:{os.path.abspath(db_path)}?mode=ro', uri=True)
        rows = conn.execute(
            "SELECT DISTINCT patient_id, accession_number FROM StudyCases"
        ).fetchall()
        conn.close()
        pairs = {(str(p), str(a)) for p, a in rows}
        print(f"Loaded {len(pairs)} existing (patient_id, accession_number) pairs from {db_path}")
        return pairs
    except sqlite3.Error as e:
        print(f"WARNING: could not read StudyCases from {db_path}: {e}")
        return set()


def filter_dcm_files_by_anon_data(dcm_files_list, anon_location, encryption_key, existing_pairs=None):
    """
    Filter DICOM files to only process those that exist in the anonymized input data.

    Parses non-anonymized patient_id and accession_number from file paths,
    encrypts them, and matches against the anonymized CSV data. If
    `existing_pairs` is provided, additionally skips files whose (encrypted
    patient_id, encrypted accession_number) is already in that set.

    Parameters:
    - dcm_files_list: List of DICOM file paths with pattern /{patient_id}_{accession_id}/{hash}.dcm
    - anon_location: Path to the CSV file containing anonymized patient_id and accession_number
    - encryption_key: Key used to encrypt the IDs
    - existing_pairs: Optional set of (patient_id, accession_number) pairs already
      parsed in a prior run (from --existing-db); files matching these are skipped.

    Returns:
    - Filtered list of DICOM file paths that match the anonymized input data
    """
    print("Filtering DICOM files based on anonymized input data...")
    if existing_pairs is None:
        existing_pairs = set()

    # Read the anonymized CSV with dtype=str to preserve leading zeros
    anon_csv = pd.read_csv(anon_location, dtype={'PATIENT_ID': str, 'ACCESSION_NUMBER': str})
    anon_csv.columns = [to_snake_case(col) for col in anon_csv.columns]

    # Strip whitespace and ensure strings
    anon_csv['patient_id'] = anon_csv['patient_id'].astype(str).str.strip()
    anon_csv['accession_number'] = anon_csv['accession_number'].astype(str).str.strip()

    # Create set of anonymized pairs for fast lookup
    anon_pairs = set(zip(anon_csv['patient_id'], anon_csv['accession_number']))

    print(f"Loaded {len(anon_pairs)} unique patient-accession pairs from anonymized data")

    # Filter DICOM files
    filtered_files = []
    skipped_count = 0
    skipped_existing_db = 0
    parse_errors = 0
    found_pairs = set()  # Track which pairs we actually found
    all_file_pairs = set()  # Track all pairs found in files (including those not in CSV)

    for dcm_path in tqdm(dcm_files_list, desc="Filtering DICOM files"):
        # Parse patient_id and accession_number from path
        # Pattern: /{patient_id}_{accession_id}/{hash}.dcm

        # Get the parent directory name which contains {patient_id}_{accession_id}
        path_normalized = dcm_path.replace('\\', '/')
        path_parts = path_normalized.split('/')

        # Find the directory name (second-to-last part before the .dcm file)
        if len(path_parts) < 2:
            parse_errors += 1
            continue

        parent_dir = path_parts[-2]

        # Split by underscore - since IDs never contain underscores, this is unambiguous
        parts = parent_dir.split('_')
        if len(parts) != 2:
            parse_errors += 1
            continue

        patient_id, accession_number = parts
        
        # Encrypt these IDs using the same encryption as deidentify_dicom()
        encrypted_patient_id = encrypt_single_id(encryption_key, patient_id)
        encrypted_accession_number = encrypt_single_id(encryption_key, accession_number)

        # Check if this pair exists in our anonymized data
        pair = (encrypted_patient_id, encrypted_accession_number)
        all_file_pairs.add(pair)  # Track all pairs in files
        
        if pair in anon_pairs:
            if pair in existing_pairs:
                # Already parsed in a prior run; skip per --existing-db.
                skipped_existing_db += 1
                continue
            filtered_files.append(dcm_path)
            found_pairs.add(pair)
        else:
            skipped_count += 1

    # Calculate missing accessions (accessions in CSV but not in DICOM files)
    missing_pairs = anon_pairs - found_pairs
    missing_count = len(missing_pairs)
    total_anon_pairs = len(anon_pairs)
    missing_percentage = (missing_count / total_anon_pairs * 100) if total_anon_pairs > 0 else 0

    print(f"Filtered {len(dcm_files_list)} files down to {len(filtered_files)} files")
    print(f"Found {len(all_file_pairs)} unique patient-accession pairs in file list")
    print(f"Found {len(found_pairs)} unique patient-accession pairs matching CSV")
    print(f"{missing_percentage:.1f}% of anonymized accessions ({missing_count}/{total_anon_pairs}) were not found in DICOM files")
    if existing_pairs:
        print(f"Skipped {skipped_existing_db} files whose (patient_id, accession_number) was already in --existing-db")
        append_audit("dicom_parsing.skipped_existing_db_pairs", skipped_existing_db)

    return filtered_files

# Main Method
def Parse_Dicom_Files(CONFIG, anon_location, lesion_anon_file, birads_anon_file, raw_storage_database, encryption_key, existing_db_path=None):
    """
    Main DICOM processing function.

    Args:
        CONFIG: Configuration dictionary
        anon_location: Path to anonymized CSV file
        lesion_anon_file: Path to lesion data file
        raw_storage_database: Path to raw DICOM storage
        encryption_key: Encryption key for patient IDs
        existing_db_path: Optional path to a previous cadbusi.db. When set,
            (patient_id, accession_number) pairs already in its StudyCases are
            skipped during the DICOM filter step.
    """
    database_path = CONFIG["DATABASE_DIR"]
    data_range = CONFIG["DEBUG_DATA_RANGE"]

    # Set the global encryption key
    global ENCRYPTION_KEY
    ENCRYPTION_KEY = encryption_key

    # Create database dir
    make_dirs(database_path)
    make_dirs(f'{database_path}/images/')
    make_dirs(f'{database_path}/videos/')


    # Get every Dicom File. CADBUSI_DICOM_LIST (a local text file, one bucket path per line) replaces
    # the listing, so a test run can target a few exams without walking the whole download folder.
    dicom_list = os.environ.get("CADBUSI_DICOM_LIST")
    if dicom_list:
        with open(dicom_list) as f:
            dcm_files_list = [line.strip() for line in f if line.strip().endswith('.dcm')]
        print(f'Using explicit DICOM list {dicom_list}')
    else:
        dcm_files_list = get_files_by_extension(raw_storage_database, '.dcm')
    append_audit("dicom_parsing.input_dicoms", len(dcm_files_list))
    print(f'Total Dicoms in Input: {len(dcm_files_list)}')

    # Apply data range only if it's specified
    if data_range and len(data_range) == 2:
        dcm_files_list = dcm_files_list[data_range[0]:data_range[1]]
        print(f'Applied Data Range: {len(dcm_files_list)}')
        append_audit("dicom_parsing.data_range", [data_range[0], data_range[1]])

    # Remove duplicates from current file list
    dcm_files_list = deduplicate_dcm_files(dcm_files_list)
    print(f'Unique DICOM files to process: {len(dcm_files_list)}')

    # Load (patient_id, accession_number) pairs from an existing DB so we can
    # skip already-parsed cases. Empty set if --existing-db wasn't supplied.
    existing_pairs = get_existing_study_pairs(existing_db_path)

    # Filter files to only process those in the anonymized input data
    dcm_files_list = filter_dcm_files_by_anon_data(dcm_files_list, anon_location, encryption_key, existing_pairs)

    # Read anon CSV to get BI-RADS 4 accessions for video filtering
    birads_4_accessions = None
    try:
        anon_csv = pd.read_csv(anon_location, dtype={'PATIENT_ID': str, 'ACCESSION_NUMBER': str})
        anon_csv.columns = [to_snake_case(col) for col in anon_csv.columns]

        if 'bi_rads' in anon_csv.columns:
            # Filter to BI-RADS 4 (includes 4, 4A, 4B, 4C, etc.)
            birads_4_df = anon_csv[anon_csv['bi_rads'].astype(str).str.contains('4', na=False)]
            birads_4_accessions = set(birads_4_df['accession_number'].astype(str).str.strip())
            print(f"Found {len(birads_4_accessions)} BI-RADS 4 accessions - videos will be filtered to these cases only")
        else:
            print("Warning: 'bi_rads' column not found in CSV, all videos will be processed")
    except Exception as e:
        print(f"Warning: Could not load BI-RADS filter data: {e}")

    # Get DCM Data (already returns snake_case columns from updated parse_files)
    image_df = parse_files(CONFIG, dcm_files_list, database_path, birads_4_accessions)

    # Convert all columns to snake_case (defensive - in case parse_files missed any)
    image_df.columns = [to_snake_case(col) for col in image_df.columns]

    # Remove missing IDs (now using snake_case)
    original_row_count = len(image_df)
    # Handle both NaN and empty strings
    image_df['patient_id'] = image_df['patient_id'].replace('', np.nan)
    image_df['accession_number'] = image_df['accession_number'].replace('', np.nan)
    image_df = image_df.dropna(subset=['patient_id', 'accession_number'])
    new_row_count = len(image_df)
    removed_rows = original_row_count - new_row_count
    print(f"Removed {removed_rows} rows with empty values.")
    append_audit("dicom_parsing.missing_ID_removed", removed_rows)

    # Parse and insert study/image/video data
    parse_anon_file(anon_location, image_df)

    # Insert lesion/pathology data into database
    print("Inserting lesion/pathology data")
    lesion_csv = pd.read_csv(lesion_anon_file)
    lesion_csv.columns = [to_snake_case(col) for col in lesion_csv.columns]
    
    # Insert lesion/description data into database
    print("Inserting birad/description data")
    birads_csv = pd.read_csv(birads_anon_file)
    birads_csv.columns = [to_snake_case(col) for col in birads_csv.columns]

    with DatabaseManager() as db:
        # Insert pathology data (dict keys now match DB schema)
        pathology_data = lesion_csv.to_dict('records')
        inserted_pathology = db.insert_pathology_batch(pathology_data)
        print(f"Inserted {inserted_pathology} pathology records")

        # Update StudyCases table with lesion_descriptions from birads CSV
        if not birads_csv.empty:
            # Rename the description column to match our database schema
            if 'birad_descriptions' in birads_csv.columns:
                birads_csv = birads_csv.rename(columns={'birad_descriptions': 'lesion_descriptions'})

            # Update study cases with lesion_descriptions
            study_update_data = birads_csv.to_dict('records')
            updated_studies = db.insert_study_cases_batch(study_update_data, update_only=True)
            print(f"Updated {updated_studies} study cases with lesion descriptions")
        else:
            print("Skipping lesion descriptions update - no birads data available")