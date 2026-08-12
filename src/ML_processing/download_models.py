import os
import requests
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))

# Read token for the private Hugging Face repo holding our weights. Sourced
# from `hf-credentials.env` (fetched from
# gs://shared-aif-bucket-87d1/Keys/hf-credentials.env) or straight from the
# environment, whichever is present — an already-set variable always wins.
# `src/credentials.py` is shared verbatim with CADBUSI-Production; keep the two
# copies identical rather than editing one of them in place.
_HF_TOKEN_ENV = "HF_TOKEN"


def _hf_headers() -> dict:
    try:
        from src.credentials import load_credentials
    except ImportError:
        # Tolerate being run with this repo's root off sys.path.
        try:
            import importlib.util
            _spec = importlib.util.spec_from_file_location(
                "_cadbusi_credentials",
                Path(current_dir).parent / "credentials.py",
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            load_credentials = _mod.load_credentials
        except Exception:
            load_credentials = None
    if load_credentials is not None:
        try:
            load_credentials()
        except Exception:
            pass  # never let credential discovery break a download path
    token = (os.environ.get(_HF_TOKEN_ENV) or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}

def download_models():
    """
    Download all required models from Hugging Face.
    """
    model_dir = Path(current_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all models to download
    models = {
        "SAMUS": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/SAMUS.pth",
            "filename": "SAMUS.pth"
        },
        "YOLO": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/yolov11_lesion_detector_FP_control_2026_1_26.pt",
            "filename": "yolov11_lesion_detector_FP_control_2026_1_26.pt"
        },
        "Caliper": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/caliper_detect_10_7_25.pt",
            "filename": "caliper_detect_10_7_25.pt"
        },
        "Mask": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/ocr_mask_yolo_2026_4_13.pt",
            "filename": "ocr_mask_yolo_2026_4_13.pt"
        },
        "N2N": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/N2N_7.pth",
            "filename": "N2N_7.pth"
        },
        "Caliper Cropped CLF": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/caliper_pipeline_cropped_clf_2_3_2026.pt",
            "filename": "caliper_pipeline_cropped_clf_2_3_2026.pt"
        },
        "Caliper Locator": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/caliper_pipeline_locator_2_3_2026.pt",
            "filename": "caliper_pipeline_locator_2_3_2026.pt"
        },
        "Caliper Uncropped CLF": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/caliper_pipeline_uncropped_clf_2_3_2026.pt",
            "filename": "caliper_pipeline_uncropped_clf_2_3_2026.pt"
        },
        "US Region Cropper": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/us_region_2026_02_06.pth",
            "filename": "us_region_2026_02_06.pth"
        },
        "LOGIQE Orientation Yolo": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/LOGIQE_ori_yolo_2026_02_06.pt",
            "filename": "LOGIQE_ori_yolo_2026_02_06.pt"
        }
    }
    
    for model_name, model_info in models.items():
        model_path = model_dir / model_info["filename"]
        
        # Check if file already exists
        if model_path.exists():
            continue
        
        print(f"\nDownloading {model_info['filename']} from Hugging Face...")
        
        try:
            response = requests.get(
                model_info["url"], stream=True, verify=False,
                headers=_hf_headers(),
            )
            # A private repo answers an unauthenticated request with 401, and a
            # token lacking access with 403. Both otherwise surface as a bare
            # HTTPError that reads like a network fault — say what it is.
            if response.status_code in (401, 403):
                have = "set" if os.environ.get(_HF_TOKEN_ENV) else "NOT set"
                raise RuntimeError(
                    f"Hugging Face refused the download of "
                    f"{model_info['filename']} (HTTP {response.status_code}). "
                    f"{_HF_TOKEN_ENV} is {have}. The weights repo is private — "
                    f"see hf-credentials.env in the README."
                )
            response.raise_for_status()
            
            # Get file size for progress
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
            print(f"\nSuccessfully downloaded {model_info['filename']} ({downloaded / (1024*1024):.1f} MB)")
            
        except Exception as e:
            print(f"\nError downloading {model_info['filename']}: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise