#!/usr/bin/env python3
"""Harvest burned-in TEXT annotations as RGBA stamps, for N2N text inpainting.

WHY THIS EXISTS
  `train_twin_N2N.py` learns to remove calipers by pasting caliper shapes from
  `PairExport/unique_caliper_shapes` onto a clean frame twice, independently,
  and asking the network to map one paste to the other. The shapes are the
  training signal. This script produces the equivalent asset for TEXT: a
  library of real burned-in annotation glyphs cut out of real frames, with an
  alpha channel, so the same Noise2Noise trick can be run against text.

  Text is NOT calipers and is deliberately kept separate. The BUS_framework
  audit (experiments/text_leakage/TEXT_LEAKAGE.md) found the classifier
  responds to a handful of sonographer annotations -- CLIP most of all, then
  RIB / SCAR / IMPLANT in the tail -- independently of any caliper mark.

SOURCE
  The OCR pass in that audit already located every annotation at full
  resolution and classified it `clinical` (sonographer overlay) vs `chrome`
  (the scanner's own UI). Its detection CSVs are the input here. Only
  `clinical` rows are harvested: chrome must be PRESERVED by the inpainter,
  because the database pipeline reads scanner metadata off it.

HOW THE ALPHA IS BUILT
  Burned-in text is a thin bright stroke laid over tissue. A grayscale
  morphological top-hat (patch minus its opening) responds to exactly that:
  strokes thinner than the kernel survive, tissue and speckle do not. The
  alpha is the normalised top-hat; the RGB is the untouched patch, so a paste
  reproduces the original pixel values rather than a synthetic white.

  Cutting a rectangle and pasting it opaquely would carry a patch of donor
  tissue with it and teach the network to erase rectangles. The alpha is what
  makes the paste a glyph.

OUT
  data/text_inpaint/stamps/<phrase>_<n>.png   RGBA
  data/text_inpaint/stamps_manifest.csv       one row per kept stamp
  data/text_inpaint/stamps_contact_sheet.png  visual QC

  python training/build_text_stamps.py --contact-sheet
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "data/text_inpaint"
STAMPS = OUT / "stamps"

# The audit lives in the sibling research repo; only its detection CSVs are
# read, and the rows actually used are copied into data/text_inpaint/ so this
# repo can rebuild the library without it.
AUDIT = Path("F:/CODE/CADBUSI/BUS_framework/experiments/text_leakage/results")
FULL = Path("D:/DATA/CADBUSI/database_images")

MARGIN = 3          # px of context kept around the OCR box
TOPHAT_K = 9        # must exceed stroke width; strokes are 2-4px at full res
ALPHA_FLOOR = 0.32  # below this the top-hat is speckle, not stroke

# The audit's classifier files these as `other`, but they are sonographer
# words, and two of them -- RIB and IMPLANT -- are in its section-8 tail. The
# rest of the `other` bucket really is chrome it misfiled (`+`, `TAC1`,
# `DYN R 65`, `P LOW`, `X4`, `58%`), so it is an allow-list, not a wider net.
EXTRA_CLINICAL = {"RIB", "IMPLANT", "INPLANT", "CALC", "BREAST"}


def norm_phrase(t: str) -> str:
    """Collapse OCR variants into a filesystem-safe phrase tag."""
    t = str(t).upper().strip()
    if re.fullmatch(r"(NIP|NPL|NIPPLE)\|?", t):
        return "NIPPLE"
    if t.startswith("CLIP"):
        return "CLIP"
    if re.fullmatch(r"\d{1,2}[:.]\d{2}", t):
        return "CLOCK"
    if re.fullmatch(r"\d{1,2}\s?CM\s?FN", t.replace("CMFN", " CMFN")):
        return "DIST"
    if "CMFN" in t.replace(" ", ""):
        return "DIST"
    if t in {"RT", "LT", "RIGHT BREAST", "LEFT BREAST", "RIGHT", "LEFT"}:
        return "LAT"
    if t in {"RAD", "ANTI-RADIAL", "RADIAL", "TRANS", "LONG", "SAG"}:
        return "ORIENT"
    return re.sub(r"[^A-Z0-9]+", "", t) or "OTHER"


def cut(gray: np.ndarray, x0, y0, x1, y1):
    """Return (rgba, coverage, contrast) or None if the cut is not a glyph."""
    h, w = gray.shape
    x0 = max(0, int(x0) - MARGIN); y0 = max(0, int(y0) - MARGIN)
    x1 = min(w, int(x1) + MARGIN); y1 = min(h, int(y1) + MARGIN)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    patch = gray[y0:y1, x0:x1]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_K, TOPHAT_K))
    top = cv2.morphologyEx(patch, cv2.MORPH_TOPHAT, k).astype(np.float32)
    hi = float(np.percentile(top, 99.0))
    if hi < 12:                      # nothing stands off the background
        return None
    alpha = np.clip(top / hi, 0.0, 1.0)
    alpha[alpha < ALPHA_FLOOR] = 0.0

    solid = alpha > 0.5
    coverage = float(solid.mean())
    if not (0.015 <= coverage <= 0.40):
        return None                  # empty cut, or a bright blob not text
    contrast = float(patch[solid].mean() - patch[~solid].mean())
    if contrast < 25:
        return None

    rgba = np.dstack([patch, patch, patch,
                      (alpha * 255).astype(np.uint8)]).astype(np.uint8)
    # Trim fully transparent rows/cols so the stamp is tight around the glyph.
    ys, xs = np.where(rgba[:, :, 3] > 0)
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgba, coverage, contrast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", nargs="+", type=Path,
                    default=[AUDIT / "analysis_detections.csv",
                             AUDIT / "analysis_detections_inpainted.csv"])
    ap.add_argument("--full-dir", type=Path, default=FULL)
    ap.add_argument("--contact-sheet", action="store_true")
    args = ap.parse_args()

    STAMPS.mkdir(parents=True, exist_ok=True)
    det = pd.concat([pd.read_csv(p) for p in args.detections if p.exists()])
    det["phrase"] = det.text.map(norm_phrase)
    det = det[(det.kind == "clinical") |
              ((det.kind == "other") & det.phrase.isin(EXTRA_CLINICAL))].copy()
    det["stem_file"] = det.stem.astype(str).str.replace("_inpainted", "",
                                                        regex=False)
    print(f"{len(det):,} clinical detections over "
          f"{det.stem_file.nunique():,} frames")

    rows, seen, n_img = [], set(), 0
    for stem, g in det.groupby("stem_file"):
        p = args.full_dir / f"{stem}.png"
        if not p.exists():
            continue
        gray = np.asarray(Image.open(p).convert("L"))
        n_img += 1
        for r in g.itertuples():
            got = cut(gray, r.x0, r.y0, r.x1, r.y1)
            if got is None:
                continue
            rgba, cov, con = got
            # Dedup on a coarse alpha signature: the same annotation recurs on
            # every frame of an exam and would otherwise dominate the library.
            sig = cv2.resize(rgba[:, :, 3], (24, 8),
                             interpolation=cv2.INTER_AREA)
            key = hashlib.md5((sig > 96).tobytes()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            name = f"{r.phrase}_{len(rows):04d}.png"
            Image.fromarray(rgba, "RGBA").save(STAMPS / name)
            rows.append(dict(file=name, phrase=r.phrase, text=r.text,
                             stem=r.stem_file, h=rgba.shape[0], w=rgba.shape[1],
                             coverage=round(cov, 4), contrast=round(con, 1),
                             conf=r.conf))

    man = pd.DataFrame(rows)
    man.to_csv(OUT / "stamps_manifest.csv", index=False)
    det.to_csv(OUT / "source_detections.csv", index=False)
    print(f"\nread {n_img:,} frames -> {len(man):,} unique stamps")
    print(man.phrase.value_counts().to_string())
    print(f"\nsize px  w {man.w.median():.0f} (p05 {man.w.quantile(.05):.0f} "
          f"p95 {man.w.quantile(.95):.0f})   h {man.h.median():.0f}")
    print(f"wrote {OUT / 'stamps_manifest.csv'}")

    if args.contact_sheet and len(man):
        cell, cols = 160, 10
        pick = man.groupby("phrase", group_keys=False).head(6).head(cols * 12)
        rows_n = int(np.ceil(len(pick) / cols))
        sheet = Image.new("RGB", (cell * cols, cell * rows_n), (18, 18, 18))
        for i, r in enumerate(pick.itertuples()):
            im = Image.open(STAMPS / r.file).convert("RGBA")
            s = min(cell / im.width, cell / im.height, 3.0)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                           Image.NEAREST)
            # Composite over mid-grey so the alpha edges are visible.
            bg = Image.new("RGBA", im.size, (110, 110, 110, 255))
            bg.alpha_composite(im)
            sheet.paste(bg.convert("RGB"),
                        ((i % cols) * cell, (i // cols) * cell))
        sheet.save(OUT / "stamps_contact_sheet.png")
        print(f"wrote {OUT / 'stamps_contact_sheet.png'}")


if __name__ == "__main__":
    main()
