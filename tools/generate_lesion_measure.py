"""Generate a lesion-measurement database from stored caliper coordinates.

For every Images row that has caliper_coordinates AND a closest_fn (the paired
clean/no-caliper frame), pair the caliper points into measurement lines
(find_smallest_perpendicular_pairs) and convert pixel distances to cm via the
caliper frame's physical_delta_x.

Output rows are keyed to the CLEAN image (the closest_fn): image_name and
dicom_hash come from the closest_fn's Images row. One output row per
measurement line, so an image with a perpendicular lesion (2 lines) or
multiple lesions produces multiple rows.

Usage:
    python tools/generate_lesion_measure.py [--db PATH] [--out PATH]
"""
import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from src.ML_processing.caliper_coordinates import find_smallest_perpendicular_pairs

DEFAULT_DB = r"F:\CODE\CADBUSI\CADBUSI-Database\data\cadbusi_2026_7_16.db"
DEFAULT_OUT = r"F:\CODE\CADBUSI\CADBUSI-Database\data\lesion_measurements.db"


def parse_caliper_coordinates(raw):
    """Parse caliper_coordinates into [(x, y), ...].

    Handles both formats found in the DB:
      - JSON list: [{"x": 417.7, "y": 347.9, "score": 0.99}, ...]
      - Legacy string: "x,y;x,y;...;"
    """
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            pts = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [(float(p["x"]), float(p["y"])) for p in pts
                if isinstance(p, dict) and "x" in p and "y" in p]
    points = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            continue
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB, help="input cadbusi sqlite db")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output sqlite db")
    args = ap.parse_args()

    src = sqlite3.connect(args.db)
    cur = src.cursor()

    # dicom_hash lookup for the clean (closest_fn) images
    hash_of = dict(cur.execute(
        "SELECT image_name, dicom_hash FROM Images WHERE dicom_hash IS NOT NULL"))

    rows = cur.execute("""
        SELECT image_name, closest_fn, caliper_coordinates, physical_delta_x
        FROM Images
        WHERE caliper_coordinates IS NOT NULL AND caliper_coordinates != ''
          AND closest_fn IS NOT NULL AND closest_fn != ''
    """).fetchall()
    src.close()
    print(f"{len(rows)} caliper rows with a closest_fn to process")

    records = []           # (image_name, dicom_hash, lesion_cm)
    n_missing_clean = 0
    n_no_delta = 0
    n_too_few_points = 0

    for caliper_name, closest_fn, coords_raw, delta_x in tqdm(rows, desc="Measuring"):
        clean_hash = hash_of.get(closest_fn)
        if clean_hash is None:
            n_missing_clean += 1
            continue
        if delta_x is None or delta_x <= 0:
            n_no_delta += 1
            continue

        points = parse_caliper_coordinates(coords_raw)
        if len(points) < 2:
            n_too_few_points += 1
            continue

        groups = find_smallest_perpendicular_pairs(points)
        arr = np.asarray(points, dtype=float)
        for group in groups:
            for i, j in group["connections"]:
                px = float(np.linalg.norm(arr[i] - arr[j]))
                lesion_cm = round(px * delta_x, 4)
                records.append((closest_fn, clean_hash, lesion_cm))

    print(f"skipped: {n_missing_clean} closest_fn not in Images, "
          f"{n_no_delta} without physical_delta_x, "
          f"{n_too_few_points} with <2 caliper points")
    print(f"{len(records)} measurements from "
          f"{len(set(r[0] for r in records))} distinct clean images")

    if os.path.exists(args.out):
        os.remove(args.out)
    out = sqlite3.connect(args.out)
    out.execute("""
        CREATE TABLE LesionMeasurements (
            measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            dicom_hash TEXT NOT NULL,
            lesion_cm REAL NOT NULL
        )
    """)
    out.execute("CREATE INDEX idx_lm_image_name ON LesionMeasurements(image_name)")
    out.execute("CREATE INDEX idx_lm_dicom_hash ON LesionMeasurements(dicom_hash)")
    out.executemany(
        "INSERT INTO LesionMeasurements (image_name, dicom_hash, lesion_cm) VALUES (?, ?, ?)",
        records)
    out.commit()
    out.close()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
