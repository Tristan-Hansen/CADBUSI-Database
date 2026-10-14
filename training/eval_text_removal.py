#!/usr/bin/env python3
"""Measure text removal on frames carrying REAL annotation, not pasted stamps.

The training pairs only ever contain stamps this repo pasted, so training loss
cannot say whether the model transfers to annotation as the scanner burned it
in. The 227 frames `build_text_frames.py` excluded for carrying genuine
clinical text are the held-out set that can, and the OCR boxes from the
BUS_framework audit say exactly where the words are.

A BOX IS NOT AUTOMATICALLY A TARGET
  21% of the clinical boxes on these frames sit OUTSIDE the ultrasound
  rectangle -- in the annotation strip under the image, where the database
  pipeline reads laterality and clock position off them and where the
  classifier's 256px crop never reaches. Those are preservation targets, not
  removal targets, and scoring them together makes a correctly-preserved word
  look like a removal failure. So boxes are split by how much of their area
  falls inside the ultrasound rectangle:

    remove   >= 90% inside  -- the model should erase these
    keep     <= 10% inside  -- the model should leave these alone
    partial  in between     -- reported, not graded

METRICS
  change     mean |pixel change| over the box, grey levels 0-255
  residual   how much of the word's own contrast survives: the after/before
             ratio of (box mean - surrounding ring mean). 0 = gone,
             1 = untouched. Computed only where the word had >3 grey levels of
             contrast to begin with, since a ratio off a near-zero
             denominator is noise.
  tissue     mean |change| over the ultrasound region excluding every box --
             the cost the model charges the image it is supposed to protect

Also reported on CLEAN training-pool frames: the same `tissue` number on
frames with nothing to remove at all.

Out: data/text_inpaint/eval_text_removal.csv  (one row per box)

  python -m training.eval_text_removal
"""
from __future__ import annotations

import torch  # MUST precede numpy/PIL on Windows -- MKL/OpenMP DLL clash
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.ML_processing.inpaint_text_N2N import load_model, remove_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data/text_inpaint"
FULL = Path("D:/DATA/CADBUSI/database_images")
PAD = 4          # px of slack around the OCR box; strokes bleed a little
RING = 12        # width of the comparison ring used for contrast


def inside_frac(b, r) -> float:
    x0, y0, x1, y1 = b
    ix = max(0, min(x1, r[0] + r[2]) - max(x0, r[0]))
    iy = max(0, min(y1, r[1] + r[3]) - max(y0, r[1]))
    return (ix * iy) / max(1.0, (x1 - x0) * (y1 - y0))


def contrast(img, box, shape):
    """Box mean minus the mean of the ring around it."""
    x0, y0, x1, y1 = box
    ry0, ry1 = max(0, y0 - RING), min(shape[0], y1 + RING)
    rx0, rx1 = max(0, x0 - RING), min(shape[1], x1 + RING)
    ring = np.ones((ry1 - ry0, rx1 - rx0), bool)
    ring[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = False
    if not ring.any():
        return None
    return float(img[y0:y1, x0:x1].mean() - img[ry0:ry1, rx0:rx1][ring].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", type=Path, default=FULL)
    ap.add_argument("--mode", choices=["masked", "full"], default="masked")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-clean", type=int, default=200)
    args = ap.parse_args()

    model = load_model(args.model)
    ann = pd.read_csv(DATA / "frames_annotated.csv")
    det = pd.read_csv(DATA / "source_detections.csv")
    det = det[det.kind == "clinical"]
    boxes = {s: g[["x0", "y0", "x1", "y1"]].values
             for s, g in det.groupby(det.stem.astype(str)
                                     .str.replace("_inpainted", "", regex=False))}

    rows, tissue = [], []
    for r in ann.itertuples():
        bb = boxes.get(r.stem)
        p = args.full_dir / f"{r.stem}.png"
        if bb is None or not p.exists():
            continue
        im = Image.open(p).convert("L")
        out, _ = remove_text(im, model, us_rect=(r.us_x, r.us_y, r.us_w, r.us_h),
                             mode=args.mode)
        a = np.asarray(im, np.float32)
        b = np.asarray(out, np.float32)
        d = np.abs(a - b)
        rect = (r.us_x, r.us_y, r.us_w, r.us_h)

        covered = np.zeros(a.shape, bool)
        for x0, y0, x1, y1 in bb:
            frac = inside_frac((x0, y0, x1, y1), rect)
            x0, y0 = max(0, int(x0) - PAD), max(0, int(y0) - PAD)
            x1, y1 = min(a.shape[1], int(x1) + PAD), min(a.shape[0], int(y1) + PAD)
            if x1 <= x0 or y1 <= y0:
                continue
            covered[y0:y1, x0:x1] = True
            before = contrast(a, (x0, y0, x1, y1), a.shape)
            after = contrast(b, (x0, y0, x1, y1), a.shape)
            rows.append(dict(
                stem=r.stem, split=r.split,
                role="remove" if frac >= 0.9 else "keep" if frac <= 0.1 else "partial",
                inside_frac=round(frac, 3),
                change=float(d[y0:y1, x0:x1].mean()),
                residual=(after / before) if (before is not None and after is not None
                                              and abs(before) > 3) else np.nan))

        us = np.zeros(a.shape, bool)
        us[int(r.us_y):int(r.us_y + r.us_h), int(r.us_x):int(r.us_x + r.us_w)] = True
        rest = us & ~covered
        if rest.any():
            tissue.append(float(d[rest].mean()))

    e = pd.DataFrame(rows)
    e.to_csv(DATA / "eval_text_removal.csv", index=False)

    print(f"{ann.stem.nunique()} held-out frames with real annotation, "
          f"{len(e)} clinical boxes\n")
    g = e.groupby("role")
    summ = pd.DataFrame({
        "boxes": g.size(),
        "mean |change|": g.change.mean().round(2),
        "median |change|": g.change.median().round(2),
        "median residual": g.residual.median().round(3),
        "residual < 0.5": g.residual.apply(lambda s: (s < 0.5).mean()).round(3),
    })
    print(summ.to_string())
    rem = e[e.role == "remove"]
    print(f"\nremoval targets  : {len(rem)} boxes, median residual "
          f"{rem.residual.median():.3f} (0 = word gone, 1 = untouched)")
    print(f"                   {(rem.residual < 0.5).mean():.1%} reduced to under "
          f"half their original contrast")
    kep = e[e.role == "keep"]
    print(f"preserved targets: {len(kep)} boxes, median |change| "
          f"{kep.change.median():.3f} grey levels")
    print(f"\ntissue inside the ultrasound region, excluding every box: "
          f"mean {np.mean(tissue):.3f} grey levels")

    clean = pd.read_csv(DATA / "frames.csv")
    clean = clean[clean.val == 1].sample(min(args.n_clean, len(clean)),
                                         random_state=0)
    cd = []
    for r in clean.itertuples():
        p = args.full_dir / f"{r.stem}.png"
        if not p.exists():
            continue
        im = Image.open(p).convert("L")
        out, _ = remove_text(im, model, us_rect=(r.us_x, r.us_y, r.us_w, r.us_h),
                             mode=args.mode)
        a, b = np.asarray(im, np.float32), np.asarray(out, np.float32)
        us = np.zeros(a.shape, bool)
        us[int(r.us_y):int(r.us_y + r.us_h),
           int(r.us_x):int(r.us_x + r.us_w)] = True
        cd.append(float(np.abs(a - b)[us].mean()))
    print(f"clean frames with nothing to remove (n={len(cd)}): mean "
          f"{np.mean(cd):.3f} grey levels, p95 {np.percentile(cd, 95):.3f}")


if __name__ == "__main__":
    main()
