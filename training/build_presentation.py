#!/usr/bin/env python3
"""Render the text-inpainting results page from cases.json + train_log.csv.

Writes a single self-contained HTML file next to the assets it describes. The
case tiles are already base64 inside cases.json, so the page needs nothing
beside it and can be opened straight off disk or published as-is.

  python -m training.build_presentation
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data/text_inpaint"
TPL = HERE / "presentation_template.html"


def main() -> None:
    cases = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))
    log = pd.read_csv(DATA / "train_log.csv")
    cases["train"] = {
        "epoch": log.epoch.tolist(),
        "val_clean": [round(v, 6) for v in log.val_clean],
        "train": [round(v, 6) for v in log.train],
    }
    # Absolute-legibility figures recomputed in eval_boxes_magnitude.csv; they
    # are what the headline band quotes, so they travel with the payload.
    b = pd.read_csv(DATA / "eval_boxes_magnitude.csv")
    rem = b[b.role == "remove"]
    cases["abs"] = {
        "n": int(len(rem)),
        "median_before": round(float(rem.before.abs().median()), 1),
        "median_after": round(float(rem.after.abs().median()), 1),
        "untouched": round(float((rem.remaining >= 0.99).mean()), 4),
        "under5": round(float((rem.after.abs() < 5).mean()), 4),
        "under10": round(float((rem.after.abs() < 10).mean()), 4),
    }
    by = (rem.assign(P=rem.text.str.upper().str.extract(
            r"(NIPPLE|NIP|CLIP|SCAR|RIB|IMPLANT)", expand=False).fillna("other"))
          .groupby("P").agg(n=("remaining", "size"),
                            med=("remaining", "median")).reset_index())
    by = by[by.n >= 5].sort_values("n", ascending=False)
    cases["by_phrase"] = [
        {"phrase": r.P, "n": int(r.n), "median": round(float(r.med), 3)}
        for r in by.itertuples()]

    html = TPL.read_text(encoding="utf-8").replace(
        "__PAYLOAD__", json.dumps(cases, separators=(",", ":")))
    out = DATA / "presentation.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
