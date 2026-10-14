
import threading
thread_local = threading.local()
import os
from tools.storage_adapter import *
import json
import numpy as np
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir) # 2 dirs back

def append_audit(key, value, new_file=False):
    """
    Add or update a key-value pair in a JSON data file.
    
    Args:
        key (str): The key for the data, can use dot notation for nested objects (e.g., "general.num_patients")
        value: The value to store
        new_file (bool): If True, create a new JSON file instead of updating
    """
    # Create the fixed path for the JSON file
    json_file_path = os.path.join(parent_dir, "data", "audit.json")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    
    # Initialize data dictionary
    data = {}
    
    # If not creating a new file and the file exists, read existing data
    if not new_file and os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # If file exists but isn't valid JSON, initialize empty
            data = {}
    
    # Convert NumPy data types to Python native types
    value = convert_numpy_types(value)
    
    # Handle nested keys with dot notation (e.g., "general.num_patients")
    keys = key.split('.')
    current = data
    
    # Navigate to the deepest level of the nested structure
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Set the value at the deepest level
    current[keys[-1]] = value
    
    # Write the updated data back to the file with custom formatting
    with open(json_file_path, 'w') as f:
        # First get the standard JSON formatting
        json_str = json.dumps(data, indent=4)
        
        def compact_numeric_lists(json_text):
            # This function handles finding lists of numbers and compacting them
            lines = json_text.split('\n')
            result = []
            i = 0
            while i < len(lines):
                line = lines[i]
                # Check if this line starts a list
                if '[' in line and ']' not in line:
                    # Start collecting the list
                    list_str = line
                    j = i + 1
                    # Continue collecting until we find the closing bracket
                    while j < len(lines) and ']' not in lines[j]:
                        list_str += lines[j]
                        j += 1
                    if j < len(lines):
                        list_str += lines[j]
                        
                    # Check if this is a numeric list (contains only numbers)
                    if re.search(r'\[\s*(-?\d+(\.\d+)?)(,\s*(-?\d+(\.\d+)?))*\s*\]', list_str):
                        # Compact the list by removing newlines and extra spaces
                        compact_list = re.sub(r'\s+', ' ', list_str)
                        compact_list = re.sub(r'\[\s+', '[', compact_list)
                        compact_list = re.sub(r'\s+\]', ']', compact_list)
                        compact_list = re.sub(r',\s+', ',', compact_list)  # Changed from ', ' to just ','
                        result.append(compact_list)
                        i = j + 1
                        continue
                        
                result.append(line)
                i += 1
                
            return '\n'.join(result)
        
        # Apply the compacting function
        json_str = compact_numeric_lists(json_str)
        f.write(json_str)
    
    return data

def convert_numpy_types(obj):
    """
    Convert NumPy data types and other non-JSON serializable types to Python native types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)  # Convert sets to lists
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

def get_reader():
    # Check if this thread already has a reader
    if not hasattr(thread_local, "reader"):
        # If not, create a new reader and store it in the thread-local storage
        import easyocr
        thread_local.reader = easyocr.Reader(['en'])
    return thread_local.reader


def __getattr__(name):
    # `reader` is built on first access, not at import: EasyOCR loads (and on a
    # fresh machine downloads) its models, which nothing that only needs
    # append_audit should pay for.
    if name == "reader":
        import easyocr
        globals()["reader"] = easyocr.Reader(['en'])
        return globals()["reader"]
    raise AttributeError(name)
