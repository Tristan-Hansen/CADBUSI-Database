import os
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))

# Weights live in GCS (primary) with the private Hugging Face repo as a fallback.
# Both locations come from CONFIG["weights"]. GCS uses the same Google login as
# everything else (ADC locally, the service account in jobs); Vertex job workers
# cannot reach Hugging Face at all. The HF read token is sourced from
# `hf-credentials.env` (gs://shared-aif-bucket-87d1/Keys/) or the environment.
# `src/credentials.py` is shared verbatim with CADBUSI-Production; keep the two
# copies identical rather than editing one of them in place.
_HF_TOKEN_ENV = "HF_TOKEN"

MODEL_FILES = [
    "SAMUS.pth",
    "yolov11_lesion_detector_FP_control_2026_1_26.pt",
    "caliper_detect_10_7_25.pt",
    "ocr_mask_yolo_2026_4_13.pt",
    "N2N_7.pth",
    "caliper_pipeline_cropped_clf_2_3_2026.pt",
    "caliper_pipeline_locator_2_3_2026.pt",
    "caliper_pipeline_uncropped_clf_2_3_2026.pt",
    "us_region_2026_02_06.pth",
    "LOGIQE_ori_yolo_2026_02_06.pt",
]

# EasyOCR fetches these from GitHub on first use, which jobs cannot reach either.
EASYOCR_FILES = ["craft_mlt_25k.pth", "english_g2.pth"]


def _weights_config():
    from config import CONFIG
    w = CONFIG.get("weights", {})
    return (w.get("source", "gcs"), CONFIG["storage"]["bucket_name"],
            w.get("gcs_prefix", "pretrained_weights/cadbusi").strip("/"),
            w.get("hf_repo", "poofy38/CADBUSI"))


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


def _from_gcs(bucket_name, blob_name, dest: Path):
    from google.cloud import storage
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    blob.download_to_filename(str(dest))
    return dest.stat().st_size


def _from_hf(repo, filename, dest: Path):
    import requests
    response = requests.get(
        f"https://huggingface.co/{repo}/resolve/main/{filename}", stream=True,
        verify=False, headers=_hf_headers(), timeout=60,
    )
    # A private repo answers an unauthenticated request with 401, and a token
    # lacking access with 403. Both otherwise surface as a bare HTTPError that
    # reads like a network fault — say what it is.
    if response.status_code in (401, 403):
        have = "set" if os.environ.get(_HF_TOKEN_ENV) else "NOT set"
        raise RuntimeError(
            f"Hugging Face refused the download of {filename} "
            f"(HTTP {response.status_code}). {_HF_TOKEN_ENV} is {have}. The weights "
            f"repo is private — see hf-credentials.env in the README."
        )
    response.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return dest.stat().st_size


def _fetch(filename, dest: Path, gcs_name=None):
    """Download one file: the configured primary source first, then the other."""
    source, bucket, prefix, repo = _weights_config()
    order = ["gcs", "huggingface"] if source != "huggingface" else ["huggingface", "gcs"]
    errors = []
    for src in order:
        try:
            if src == "gcs":
                size = _from_gcs(bucket, f"{prefix}/{gcs_name or filename}", dest)
            else:
                size = _from_hf(repo, gcs_name or filename, dest)
            print(f"Downloaded {filename} from {src} ({size / (1024 * 1024):.1f} MB)")
            return
        except Exception as e:
            errors.append(f"{src}: {e}")
            if dest.exists():
                dest.unlink()  # clean up partial download
    raise RuntimeError(f"Could not download {filename}; " + " | ".join(errors))


def download_models():
    """
    Download all required model weights into src/ML_processing/models/, plus the
    EasyOCR detector/recognizer into ~/.EasyOCR/model/. Files already on disk are
    skipped.
    """
    model_dir = Path(current_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILES:
        path = model_dir / filename
        if not path.exists():
            _fetch(filename, path)

    easyocr_dir = Path.home() / ".EasyOCR" / "model"
    easyocr_dir.mkdir(parents=True, exist_ok=True)
    for filename in EASYOCR_FILES:
        path = easyocr_dir / filename
        if not path.exists():
            _fetch(filename, path, gcs_name=f"easyocr_model/{filename}")
