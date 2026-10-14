#!/usr/bin/env python3
"""Assemble the best / typical / worst cases into a self-contained report page.

`eval_text_removal.py` gives the distribution; this gives the pictures behind
it. For every clinical box on the 227 held-out frames it re-measures the word's
contrast before and after, crops a window around it, and emits before / after /
8x-difference tiles as base64 so the page needs no image files beside it.

METRIC USED FOR RANKING
  `remaining` = |contrast after| / |contrast before|, where contrast is the
  box mean minus the mean of the ring around it.

    0.0  the word is gone
    1.0  untouched
    >1   worse than untouched -- either the model dug a hole where the word was
         (overshoot) or it brightened the patch

  The signed ratio in eval_text_removal.csv goes negative on overshoot, which
  sorts a bad case next to a good one. Taking the magnitude makes the ranking
  monotone in "how wrong is this", which is what a best/worst gallery needs.

Out: data/text_inpaint/cases.json

  python -m training.build_case_gallery
"""
from __future__ import annotations

import torch  # MUST precede numpy/PIL on Windows -- MKL/OpenMP DLL clash
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.ML_processing.inpaint_text_N2N import load_model, remove_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data/text_inpaint"
FULL = Path("D:/DATA/CADBUSI/database_images")
PAD, RING = 4, 12
CTX = 46          # px of context around the box in the cropped tile


def contrast(img, box):
    x0, y0, x1, y1 = box
    ry0, ry1 = max(0, y0 - RING), min(img.shape[0], y1 + RING)
    rx0, rx1 = max(0, x0 - RING), min(img.shape[1], x1 + RING)
    ring = np.ones((ry1 - ry0, rx1 - rx0), bool)
    ring[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = False
    if not ring.any():
        return None
    return float(img[y0:y1, x0:x1].mean() - img[ry0:ry1, rx0:rx1][ring].mean())


def b64(arr, q=88):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "JPEG", quality=q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def tile(a, b, box):
    """Crop a window around the box; return (before, after, 8x diff) as b64."""
    x0, y0, x1, y1 = box
    cy0, cy1 = max(0, y0 - CTX), min(a.shape[0], y1 + CTX)
    cx0, cx1 = max(0, x0 - CTX), min(a.shape[1], x1 + CTX)
    A, B = a[cy0:cy1, cx0:cx1], b[cy0:cy1, cx0:cx1]
    D = np.clip(np.abs(A - B) * 8, 0, 255)
    return b64(A), b64(B), b64(D)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", type=Path, default=FULL)
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    model = load_model(args.model)
    ann = pd.read_csv(DATA / "frames_annotated.csv").set_index("stem")
    det = pd.read_csv(DATA / "source_detections.csv")
    det = det[det.kind == "clinical"].copy()
    det["s"] = det.stem.astype(str).str.replace("_inpainted", "", regex=False)

    recs = []
    for stem, g in det.groupby("s"):
        if stem not in ann.index:
            continue
        p = args.full_dir / f"{stem}.png"
        if not p.exists():
            continue
        r = ann.loc[stem]
        r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        im = Image.open(p).convert("L")
        out, _ = remove_text(im, model, us_rect=(r.us_x, r.us_y, r.us_w, r.us_h))
        a = np.asarray(im, np.float32)
        b = np.asarray(out, np.float32)
        rect = (r.us_x, r.us_y, r.us_w, r.us_h)

        for t in g.itertuples():
            x0, y0 = max(0, int(t.x0) - PAD), max(0, int(t.y0) - PAD)
            x1 = min(a.shape[1], int(t.x1) + PAD)
            y1 = min(a.shape[0], int(t.y1) + PAD)
            if x1 <= x0 or y1 <= y0:
                continue
            ix = max(0, min(x1, rect[0] + rect[2]) - max(x0, rect[0]))
            iy = max(0, min(y1, rect[1] + rect[3]) - max(y0, rect[1]))
            frac = (ix * iy) / max(1.0, (x1 - x0) * (y1 - y0))
            cb, ca = contrast(a, (x0, y0, x1, y1)), contrast(b, (x0, y0, x1, y1))
            if cb is None or ca is None or abs(cb) <= 3:
                continue
            recs.append(dict(
                stem=stem, text=str(t.text), split=str(r.split),
                role="remove" if frac >= 0.9 else "keep" if frac <= 0.1 else "partial",
                before=round(cb, 1), after=round(ca, 1),
                remaining=round(abs(ca) / abs(cb), 3),
                change=round(float(np.abs(a - b)[y0:y1, x0:x1].mean()), 1),
                box=[x0, y0, x1, y1], arr_a=a, arr_b=b))

    df = pd.DataFrame(recs)
    rem = df[df.role == "remove"].drop_duplicates(subset=["stem", "text"])
    keep = df[df.role == "keep"].drop_duplicates(subset=["stem", "text"])
    n = args.per_group

    med = rem.remaining.median()
    groups = {
        "best": rem.nsmallest(n, "remaining"),
        "typical": rem.iloc[(rem.remaining - med).abs().argsort()[:n]],
        "worst": rem.nlargest(n, "remaining"),
        "preserved": keep.sample(min(n, len(keep)), random_state=0),
    }

    out = {"stats": {
        "n_frames": int(df.stem.nunique()),
        "n_boxes": int(len(df)),
        "remove_n": int(len(rem)),
        "keep_n": int(len(keep)),
        "median_remaining": float(rem.remaining.median()),
        "p25_remaining": float(rem.remaining.quantile(.25)),
        "p75_remaining": float(rem.remaining.quantile(.75)),
        "frac_under_half": float((rem.remaining < 0.5).mean()),
        "frac_under_quarter": float((rem.remaining < 0.25).mean()),
        "keep_median_change": float(keep.change.median()),
    }, "hist": np.histogram(rem.remaining.clip(0, 2), bins=20,
                            range=(0, 2))[0].tolist(), "groups": {}}

    for name, gg in groups.items():
        items = []
        for r in gg.itertuples():
            A, B, D = tile(r.arr_a, r.arr_b, r.box)
            items.append(dict(text=r.text, split=r.split, before=r.before,
                              after=r.after, remaining=r.remaining,
                              change=r.change, stem=r.stem[:10],
                              a=A, b=B, d=D))
        out["groups"][name] = items

    (DATA / "cases.json").write_text(json.dumps(out), encoding="utf-8")
    mb = (DATA / "cases.json").stat().st_size / 1e6
    print(f"{len(df)} boxes on {df.stem.nunique()} frames")
    print(f"remove n={len(rem)}  median remaining contrast "
          f"{rem.remaining.median():.3f}  "
          f"<0.5 {(rem.remaining < 0.5).mean():.1%}  "
          f"<0.25 {(rem.remaining < 0.25).mean():.1%}")
    print(f"keep   n={len(keep)}  median |change| {keep.change.median():.3f}")
    print(f"wrote {DATA / 'cases.json'} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
