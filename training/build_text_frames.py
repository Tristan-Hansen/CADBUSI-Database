#!/usr/bin/env python3
"""Pick the clean frames the text-N2N model trains on.

A Noise2Noise pair is the SAME frame with two independent sets of stamps
pasted on. That only teaches removal if the frame is text-free to begin with:
any annotation already burned into it appears identically in the input and the
target, so the loss rewards KEEPING it -- the exact opposite of the goal.

So frames are taken only from the OCR'd sample in the BUS_framework audit,
where every frame has a verdict, and only those it cleared of clinical
annotation are kept. Scanner chrome is deliberately NOT excluded: it is
identical on both sides of the pair, which is how the network learns to leave
it alone. That matters -- the database pipeline reads metadata off the chrome.

Each row also carries the ultrasound region rectangle, in full-resolution
coordinates. Stamps are placed inside it because that is where a sonographer
writes; the surrounding black and the sidebar stay untouched.

OUT: data/text_inpaint/frames.csv  (stem, us rect, val)

  python training/build_text_frames.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "data/text_inpaint"
AUDIT = Path("F:/CODE/CADBUSI/BUS_framework/experiments/text_leakage/results")
FULL = Path("D:/DATA/CADBUSI/database_images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=AUDIT / "analysis_frames.csv")
    ap.add_argument("--full-dir", type=Path, default=FULL)
    ap.add_argument("--min-side", type=int, default=320,
                    help="drop frames whose ultrasound region is smaller than "
                         "this on either side -- a 512 crop off them would be "
                         "mostly background")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    f0 = pd.read_csv(args.frames)
    f = f0
    n0 = len(f)

    f = f[f.has_clin == 0]                       # the whole point of this file
    n1 = len(f)
    f = f.dropna(subset=["db_x", "db_y", "db_w", "db_h"])
    f = f[(f.db_w >= args.min_side) & (f.db_h >= args.min_side)]
    n2 = len(f)

    have = {p.stem for p in args.full_dir.glob("*.png")}
    f = f[f.stem.isin(have)]

    out = pd.DataFrame({
        "stem": f.stem,
        "us_x": f.db_x.astype(int), "us_y": f.db_y.astype(int),
        "us_w": f.db_w.astype(int), "us_h": f.db_h.astype(int),
        # The audit's split is v7's, so a frame never crosses from the
        # classifier's train set into this model's val set or back.
        "val": (f.split != "train").astype(int),
        "scanner": f.scanner,
    })
    out.to_csv(OUT / "frames.csv", index=False)

    # The frames excluded above are the only REAL test of removal: they carry
    # genuine sonographer annotation rather than a pasted stamp. They are never
    # trained on -- that is why they were excluded -- so they double as a
    # held-out QC set for `inpaint_text_N2N.py --preview`.
    ann = f0[(f0.has_clin == 1) & f0.db_w.notna() & f0.stem.isin(have)]
    pd.DataFrame({
        "stem": ann.stem,
        "us_x": ann.db_x.astype(int), "us_y": ann.db_y.astype(int),
        "us_w": ann.db_w.astype(int), "us_h": ann.db_h.astype(int),
        "clin_text": ann.clin_text, "split": ann.split,
    }).to_csv(OUT / "frames_annotated.csv", index=False)
    print(f"   plus {len(ann):,} genuinely annotated frames held out for QC "
          f"-> frames_annotated.csv")

    print(f"{n0:,} OCR'd frames")
    print(f"  {n0 - n1:,} dropped: carry a clinical annotation already")
    print(f"  {n1 - n2:,} dropped: ultrasound region under {args.min_side}px")
    print(f"  {n2 - len(out):,} dropped: no full-res file on disk")
    print(f"-> {len(out):,} usable  ({int(out.val.sum()):,} val)")
    print(f"\nultrasound region px: w {out.us_w.median():.0f}  h {out.us_h.median():.0f}")
    print(out.scanner.value_counts().head(8).to_string())
    print(f"\nwrote {OUT / 'frames.csv'}")


if __name__ == "__main__":
    main()
