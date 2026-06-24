"""Merge two cadbusi.db files via INSERT OR IGNORE per table."""
import os
import re
import sqlite3

import pandas as pd

from src.DB_processing.database import DatabaseManager


def _to_snake_case(name):
    """Local copy of dcm_parser.to_snake_case so the merge command doesn't pull
    in pydicom/cv2 just for one string utility."""
    name = name.replace('-', '_')
    if name.isupper():
        return name.lower()
    if '_' in name:
        return name.lower()
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


TABLES = [
    'StudyCases',
    'Images',
    'Videos',
    'Pathology',
    'Lesions',
    'BadImages',
    'CaliperPairs',
    'RegionLabels',
]


def _refresh_studycases_from_anon(dest_path, anon_path):
    """Refresh existing StudyCases rows in DEST from anon_data.csv.

    Updates clinical fields (laterality, BI-RADS, diagnoses, findings, etc.)
    on rows whose accession_number already exists in DEST.StudyCases. Rows in
    the CSV whose accession_number is NOT already in DEST are skipped (no
    placeholder rows are inserted). Uses the same column-mapping and
    has_malignant/has_benign derivation as the parse path so the data lands
    identically.
    """
    if not os.path.exists(anon_path):
        raise FileNotFoundError(f"anon_data.csv not found: {anon_path}")

    print(f"\nRefreshing StudyCases from {anon_path} ...")
    df = pd.read_csv(anon_path, dtype={'PATIENT_ID': str, 'ACCESSION_NUMBER': str})
    df.columns = [_to_snake_case(c) for c in df.columns]
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    df['accession_number'] = df['accession_number'].astype(str).str.strip()

    # Derive has_malignant / has_benign the same way parse_anon_file does
    # (dcm_parser.py:792-799) so the columns aren't NULL after the update.
    if 'left_diagnosis' in df.columns and 'right_diagnosis' in df.columns:
        df['has_malignant'] = (
            df['left_diagnosis'].str.contains('MALIGNANT', na=False) |
            df['right_diagnosis'].str.contains('MALIGNANT', na=False)
        )
        df['has_benign'] = (
            df['left_diagnosis'].str.contains('BENIGN', na=False) |
            df['right_diagnosis'].str.contains('BENIGN', na=False)
        )

    # Restrict the CSV to accessions that already exist in DEST.StudyCases so
    # update_only=True doesn't waste work on rows it would skip anyway. Also
    # gives us an accurate "rows updated" count.
    conn = sqlite3.connect(dest_path)
    existing = {r[0] for r in conn.execute("SELECT accession_number FROM StudyCases").fetchall()}
    conn.close()
    total_csv = len(df)
    df = df[df['accession_number'].isin(existing)].copy()
    skipped = total_csv - len(df)

    records = df.to_dict('records')
    dm = DatabaseManager()
    dm.db_file = dest_path
    with dm as d:
        before = d.conn.execute("SELECT COUNT(*) FROM StudyCases").fetchone()[0]
        d.insert_study_cases_batch(records, upsert=False, update_only=True)
        after = d.conn.execute("SELECT COUNT(*) FROM StudyCases").fetchone()[0]

    print(f"  StudyCases: {len(records):,} existing rows refreshed, "
          f"{skipped:,} CSV rows skipped (no matching accession in DEST), "
          f"row count change: {after - before:+,}")


def _premerge_cleanup(dest_path):
    """Drop the stale `inpainted_from` column from DEST's Images table if both
    `inpainted_from` and `inpainted_version` are present. Without this,
    DatabaseManager's _migrate_inpainted_column step crashes with a duplicate-
    column error while trying to RENAME inpainted_from -> inpainted_version.
    Any data in inpainted_from is discarded; inpainted_version carries forward.
    """
    conn = sqlite3.connect(dest_path)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(Images)").fetchall()]
        if 'inpainted_from' in cols and 'inpainted_version' in cols:
            print("Pre-merge cleanup: dropping stale `inpainted_from` column from DEST.Images")
            conn.execute("ALTER TABLE Images DROP COLUMN inpainted_from")
            conn.commit()
    finally:
        conn.close()


def _table_columns(conn, table, schema='main'):
    """Return ordered list of column names for {schema}.{table}, or [] if missing.

    Skips INTEGER PRIMARY KEY columns (autoincrement IDs). Copying these from
    SRC would collide with DEST's existing autoincrement IDs and `INSERT OR
    IGNORE` would silently drop the row. Letting SQLite assign fresh IDs in
    the destination avoids that.
    """
    rows = conn.execute(f"PRAGMA {schema}.table_info({table})").fetchall()
    # row = (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows
            if not (r[5] == 1 and r[2].upper() == 'INTEGER')]


def merge_databases(src_path, dest_path, anon_data_path=None):
    """Copy every row from each table in SRC into DEST using INSERT OR IGNORE,
    preserving existing DEST rows on conflict. Explicitly lists the column
    intersection so schema drift between SRC and DEST (e.g. new columns added
    to the destination after SRC was built) doesn't break the merge.

    If anon_data_path is provided, after the table-by-table merge, also
    refreshes clinical fields on existing DEST.StudyCases rows
    (left/right_diagnosis, BI-RADS, findings, etc.). Rows in the CSV whose
    accession_number is not already in DEST are skipped — no placeholder
    rows are inserted.
    """
    src_path = os.path.abspath(src_path)
    dest_path = os.path.abspath(dest_path)
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"SRC database not found: {src_path}")
    if not os.path.exists(dest_path):
        raise FileNotFoundError(f"DEST database not found: {dest_path}")

    print(f"Merging {src_path}")
    print(f"   into {dest_path}")
    print()

    # Pre-flight cleanup on the destination so DatabaseManager's migration
    # doesn't crash. Specifically: if DEST has BOTH inpainted_from and
    # inpainted_version columns, _migrate_inpainted_column would try to
    # RENAME inpainted_from -> inpainted_version and fail on the duplicate.
    # In a merge context the stale inpainted_from column carries pointer data
    # in the old direction; we drop it and let inpainted_version carry forward.
    _premerge_cleanup(dest_path)

    # Bring the destination's schema up to current (creates missing tables/columns
    # and runs any pending migrations).
    print("Syncing destination schema (idempotent)...")
    dm = DatabaseManager()
    dm.db_file = dest_path
    with dm:
        pass

    conn = sqlite3.connect(dest_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"ATTACH DATABASE '{src_path}' AS src")

    try:
        for tbl in TABLES:
            try:
                dest_cols = _table_columns(conn, tbl, schema='main')
                src_cols  = _table_columns(conn, tbl, schema='src')
                if not dest_cols:
                    print(f"  {tbl:<14} SKIPPED (table missing in DEST)")
                    continue
                if not src_cols:
                    print(f"  {tbl:<14} SKIPPED (table missing in SRC)")
                    continue

                shared = [c for c in dest_cols if c in src_cols]
                dropped_src  = [c for c in src_cols  if c not in dest_cols]
                missing_dest = [c for c in dest_cols if c not in src_cols]
                col_list = ", ".join(shared)

                before = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                conn.execute(
                    f"INSERT OR IGNORE INTO {tbl} ({col_list}) "
                    f"SELECT {col_list} FROM src.{tbl}"
                )
                after = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]

                note = ""
                if missing_dest:
                    note += f"  [dest cols not in src, will be NULL: {', '.join(missing_dest)}]"
                if dropped_src:
                    note += f"  [src cols dropped: {', '.join(dropped_src)}]"
                print(f"  {tbl:<14} +{after - before:>7,} rows (now {after:>10,}){note}")
            except sqlite3.Error as e:
                print(f"  {tbl:<14} SKIPPED ({e})")
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE src")
        conn.close()

    # After the table-by-table merge, optionally refresh StudyCases from the
    # latest anon_data.csv so existing rows pick up new clinical info even if
    # the incremental --existing-db parse skipped them.
    if anon_data_path:
        _refresh_studycases_from_anon(dest_path, anon_data_path)

    print("\nMerge complete.")
    print("Reminder: copy `images/` and `videos/` from the source DB's directory")
    print("into the destination's directory so the merged DB rows still point at")
    print("files that exist on disk.")
