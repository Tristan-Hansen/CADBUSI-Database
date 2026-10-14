"""Remove burned-in TEXT annotations from database images (Noise2Noise).

The caliper twin of this file is `inpaint_N2N.py`. Everything about the
runtime is the same -- same U-Net, same grayscale normalisation, same thread
pool, same `CaliperPairs`-style bookkeeping -- and the two are kept separate on
purpose: different weights, different trigger, different output column, so
either can be re-run or rolled back without touching the other.

WHAT IT REMOVES AND WHAT IT KEEPS
  Removes: sonographer annotation burned into the ULTRASOUND REGION --
  NIPPLE, CLIP, SCAR, PALP, clock positions, distances, laterality.

  Keeps: everything outside that region. The scanner sidebar, the banner, and
  the annotation strip below the image are left alone, because the database
  pipeline reads laterality / clock position / distance off them and because
  they never enter the 256px crop the classifier sees. That behaviour is
  trained in rather than masked in: `train_text_N2N.py` pastes stamps only
  inside the ultrasound region, so chrome is identical on both sides of every
  training pair and the loss punishes changing it.

WHY THIS EXISTS
  BUS_framework `experiments/text_leakage/TEXT_LEAKAGE.md` measured the
  classifier's response to burned-in phrases. The bag-level median is
  negligible, but `CLIP` has a real per-frame effect (~0.9 logits at the
  median) and a tail: 4 of 25 tested exams move by more than 0.05 when the
  word alone is greyed out. `RIB`, `SCAR` and `IMPLANT` show smaller tails in
  the same direction. This is the upstream fix for that.

COMPOSITING
  `--mode masked` (default) keeps the original pixels everywhere the network
  did not meaningfully change them, so the output differs from the input only
  where text was. `--mode full` returns the raw network output, which is also
  a whole-frame denoise. The masked form is what the BUS_framework caliper
  inference audit recommended after finding the raw N2N output changes ~13%
  of its pixels outside the markers.

  python -m src.ML_processing.inpaint_text_N2N --preview 12
"""

import torch  # MUST precede numpy/PIL on Windows -- MKL/OpenMP DLL clash
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from training.train_twin_N2N import N2N_Original_Used_UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
env = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(env))          # .../CADBUSI-Database
TEXT_DATA = os.path.join(REPO, "data", "text_inpaint")

MODEL_NAME = "N2N_text_1.pth"
# Pixels whose value moves by less than this are treated as untouched, so the
# masked mode substitutes only where the network actually erased something.
REPLACE_THRESHOLD = 0.05


def load_model(model_path=None):
    path = model_path or f"{env}/models/{MODEL_NAME}"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Train it with "
            f"`python -m training.train_text_N2N`, or add it to "
            f"src/ML_processing/download_models.py once it is on HuggingFace."
        )
    model = N2N_Original_Used_UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def remove_text(image, model, us_rect=None, mode="masked"):
    """Inpaint one PIL image. `us_rect` is (x, y, w, h) of the ultrasound area.

    When `us_rect` is given, edits outside it are discarded -- a belt-and-
    braces guard on top of the trained behaviour, so a bad frame can never
    cost the pipeline its sidebar metadata.
    """
    original = image
    is_rgb = original.mode == "RGB"
    gray = original.convert("L")

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    x = normalize(to_tensor(gray)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
    out = torch.clamp(out.squeeze(0).cpu() * 0.5 + 0.5, 0, 1)

    processed = transforms.ToPILImage()(out).resize(gray.size, Image.BILINEAR)

    src = np.asarray(gray, dtype=np.float32) / 255.0
    net = np.asarray(processed, dtype=np.float32) / 255.0

    if mode == "masked":
        replace = np.abs(src - net) > REPLACE_THRESHOLD
    else:
        replace = np.ones_like(src, dtype=bool)

    if us_rect is not None:
        x0, y0, w, h = (int(v) for v in us_rect)
        keep = np.zeros_like(replace)
        keep[max(0, y0):y0 + h, max(0, x0):x0 + w] = True
        replace &= keep

    result_gray = np.where(replace, net, src)

    if is_rgb:
        rgb = np.asarray(original, dtype=np.float32) / 255.0
        rgb[replace] = result_gray[replace][:, None]
        arr = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr), int(replace.sum())

    arr = np.clip(result_gray * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), int(replace.sum())


def _process_one(args):
    row, input_folder, out_folder, model, mode = args
    try:
        from tools.storage_adapter import read_image, save_data
    except ImportError:  # standalone / preview use
        read_image = lambda p, use_pil=True: Image.open(p)  # noqa: E731
        save_data = lambda im, p: im.save(p)                # noqa: E731

    try:
        src_path = os.path.normpath(os.path.join(input_folder, row["image_name"]))
        image = read_image(src_path, use_pil=True)
        rect = None
        if all(row.get(k) is not None and not pd.isna(row.get(k))
               for k in ("crop_x", "crop_y", "crop_w", "crop_h")):
            rect = (row["crop_x"], row["crop_y"], row["crop_w"], row["crop_h"])

        result, n_px = remove_text(image, model, us_rect=rect, mode=mode)

        name, ext = os.path.splitext(row["image_name"])
        new_name = f"{name}_detexted{ext}"
        save_data(result, os.path.normpath(os.path.join(out_folder, new_name)))
        return {"image_name": row["image_name"], "detexted_version": new_name,
                "pixels_changed": n_px}
    except Exception as e:  # noqa: BLE001
        print(f"Error processing {row.get('image_name')}: {e}")
        return None


def Detext_Dataset_N2N(input_folder, num_workers=8, mode="masked",
                       model_path=None):
    """Run text removal over every image the OCR stage flagged with text.

    Mirrors `Inpaint_Dataset_N2N`: reads the image table, filters to the rows
    worth touching, writes `<name>_detexted.png` into a sibling folder and
    records the mapping back into the database.
    """
    from src.DB_processing.database import DatabaseManager

    base_dir = os.path.dirname(input_folder.rstrip("/\\"))
    out_folder = os.path.join(base_dir, "detexted")
    os.makedirs(out_folder, exist_ok=True)

    with DatabaseManager() as db:
        data = db.get_images_dataframe()
        # Only frames whose ultrasound region is known; without the rect the
        # guard above cannot protect the sidebar.
        todo = data[data["crop_w"].notna() & data["crop_h"].notna()]
        if len(todo) == 0:
            print("No images to process")
            return

        print(f"Removing text from {len(todo)} images with {num_workers} workers")
        model = load_model(model_path)
        model.share_memory()

        args_list = [(r.to_dict(), input_folder, out_folder, model, mode)
                     for _, r in todo.iterrows()]
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(tqdm(ex.map(_process_one, args_list),
                                total=len(args_list), desc="Removing text"))
        rows = [r for r in results if r is not None]

        if rows:
            n = db.insert_images_batch(
                [{"image_name": r["image_name"],
                  "detexted_version": r["detexted_version"]} for r in rows],
                update_only=True)
            print(f"Updated {n} images with detexted_version")


def preview(n, input_folder, out_path, mode="masked", model_path=None,
            on="annotated"):
    """Side-by-side sheet: input | output | 8x difference, for eyeballing.

    `on="annotated"` uses the 227 frames carrying REAL sonographer annotation.
    They were excluded from training precisely because they are not clean, so
    they are held out, and they are the only honest test of removal -- the
    training pairs only ever contain pasted stamps.

    `on="clean"` uses the training pool instead, which tests the opposite
    property: that a frame with nothing to remove comes back unharmed.
    """
    name = "frames_annotated.csv" if on == "annotated" else "frames.csv"
    frames = pd.read_csv(os.path.join(TEXT_DATA, name))
    model = load_model(model_path)
    picks = frames.sample(min(n, len(frames)), random_state=0)

    tiles = []
    for r in picks.itertuples():
        p = os.path.join(input_folder, f"{r.stem}.png")
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("L")
        out, _ = remove_text(im, model,
                             us_rect=(r.us_x, r.us_y, r.us_w, r.us_h), mode=mode)
        a = np.asarray(im, np.int16)
        b = np.asarray(out.convert("L"), np.int16)
        d = np.clip(np.abs(a - b) * 8, 0, 255).astype(np.uint8)
        tiles.append(np.concatenate([a.astype(np.uint8),
                                     np.asarray(out.convert("L")), d], axis=1))
    if not tiles:
        print("nothing to preview")
        return
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]))) for t in tiles]
    Image.fromarray(np.concatenate(tiles, axis=0)).save(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-folder", default="D:/DATA/CADBUSI/database_images")
    ap.add_argument("--mode", choices=["masked", "full"], default="masked")
    ap.add_argument("--model", default=None)
    ap.add_argument("--preview", type=int, default=0,
                    help="write a QC sheet of N frames instead of running the DB pass")
    ap.add_argument("--preview-out",
                    default="data/text_inpaint/inference_preview.png")
    ap.add_argument("--on", choices=["annotated", "clean"], default="annotated",
                    help="annotated: held-out frames with REAL burned-in text "
                         "(tests removal). clean: training-pool frames with "
                         "none (tests that nothing else is damaged).")
    a = ap.parse_args()
    if a.preview:
        preview(a.preview, a.input_folder, a.preview_out, a.mode,
                a.model, a.on)
    else:
        Detext_Dataset_N2N(a.input_folder, mode=a.mode, model_path=a.model)
