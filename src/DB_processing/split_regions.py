"""Fill in per-region crop boxes for dual-region ultrasound frames.

A dual-region frame shows two panes side by side but carries a single
whole-frame crop box spanning both -- meaningless as one sample. This cuts that
box at the machine's divider so crop_* becomes region 0 (left) and crop2_*
becomes region 1 (right).

Both boxes are in full-frame coordinates against the original image file: no
image is read, written, or modified. Frame width comes from the DICOM-derived
Images.columns value.
"""
from src.DB_processing.tools import append_audit
from src.DB_processing.database import DatabaseManager


# Where the divider between the two regions sits, per machine.
#
# LOGIQ frames have asymmetric chrome (the right margin is ~104px wider than the
# left), so the divider sits a FIXED 52px left of frame centre no matter the
# frame size. A width ratio only holds at the size it was calibrated on: 0.4665
# was exact at 1552px wide but drifted 3px at 1456 and 13px at 1164.
SPLIT_OFFSETS_FROM_CENTER = {
    'LOGIQE9':  -52,
    'LOGIQE10': -52,
}

# EPIQ frames split essentially dead centre (0.501 => only ~1.5px right of
# centre at these sizes). Kept as a ratio: it's verified working, and we have no
# multi-width EPIQ samples to derive a fixed offset from.
SPLIT_RATIOS = {
    'EPIQ 5G':    0.501,
    'EPIQ 7G':    0.501,
    'EPIQ Elite': 0.501,
}

SUPPORTED_MODELS = list(SPLIT_OFFSETS_FROM_CENTER) + list(SPLIT_RATIOS)

# RegionDataType values to exclude (spectral doppler -- not a side-by-side pair)
SPECTRAL_TYPES = {'3', '4'}

CROP2_COLUMNS = ('crop2_x', 'crop2_y', 'crop2_w', 'crop2_h')


def compute_split_x(model, width):
    """Pixel x at which the frame divides into its two regions."""
    if model in SPLIT_OFFSETS_FROM_CENTER:
        return width // 2 + SPLIT_OFFSETS_FROM_CENTER[model]
    return int(width * SPLIT_RATIOS[model])


def split_crop_at(split_x, crop_x, crop_y, crop_w, crop_h):
    """Cut a whole-frame crop box at split_x into left and right boxes.

    Returns (left, right) as (x, y, w, h) in full-frame coordinates. A side is
    None when the crop box doesn't reach across the divider into it.
    """
    lx0, lx1 = crop_x, min(crop_x + crop_w, split_x)
    rx0, rx1 = max(crop_x, split_x), max(crop_x + crop_w, split_x)

    left = (lx0, crop_y, lx1 - lx0, crop_h) if lx1 > lx0 else None
    right = (rx0, crop_y, rx1 - rx0, crop_h) if rx1 > rx0 else None
    return left, right


def _ensure_crop2_columns(cursor, conn):
    """Add the region-1 crop columns to Images if this DB predates them."""
    cursor.execute("PRAGMA table_info(Images)")
    existing = {row[1] for row in cursor.fetchall()}
    for col in CROP2_COLUMNS:
        if col not in existing:
            cursor.execute(f"ALTER TABLE Images ADD COLUMN {col} INTEGER")
            print(f"Added column '{col}' to Images")
    conn.commit()


def split_regions_in_db():
    """
    Split the whole-frame crop of every dual-region image into two region crops.

    Only rows where crop2_x IS NULL are processed, so re-running is a no-op
    instead of re-splitting an already-split crop. The original whole-frame crop
    stays recoverable as [crop_x -> crop2_x + crop2_w].

    Run after crop regions exist (generate_crop_regions) and before Select_Data,
    which re-derives crop_aspect_ratio and applies the aspect filters.
    """
    with DatabaseManager() as db:
        cursor = db.conn.cursor()
        _ensure_crop2_columns(cursor, db.conn)

        placeholders = ','.join('?' * len(SUPPORTED_MODELS))
        cursor.execute(f"""
            SELECT image_name, manufacturer_model_name, region_data_type,
                   columns, crop_x, crop_y, crop_w, crop_h
            FROM Images
            WHERE region_count = 2
              AND manufacturer_model_name IN ({placeholders})
              AND crop2_x IS NULL
              AND crop_x IS NOT NULL AND crop_y IS NOT NULL
              AND crop_w IS NOT NULL AND crop_h IS NOT NULL
              AND columns IS NOT NULL
              AND region_data_type IS NOT NULL AND region_data_type != ''
        """, SUPPORTED_MODELS)

        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        if not rows:
            print("No dual-region images to split.")
            return

        print(f"Found {len(rows)} dual-region images to split...")

        updates = []
        skipped_spectral = 0
        skipped_degenerate = 0

        for row in rows:
            if set(str(row['region_data_type']).split(',')) & SPECTRAL_TYPES:
                skipped_spectral += 1
                continue

            split_x = compute_split_x(row['manufacturer_model_name'], int(row['columns']))
            left, right = split_crop_at(
                split_x,
                int(row['crop_x']), int(row['crop_y']),
                int(row['crop_w']), int(row['crop_h']),
            )

            # The crop box has to straddle the divider to yield two regions
            if left is None or right is None:
                skipped_degenerate += 1
                continue

            # crop_aspect_ratio describes crop_*, so it changes with the split
            aspect = round(left[2] / left[3], 2) if left[3] else None
            updates.append((*left, aspect, *right, row['image_name']))

        if updates:
            cursor.executemany("""
                UPDATE Images
                SET crop_x = ?, crop_y = ?, crop_w = ?, crop_h = ?, crop_aspect_ratio = ?,
                    crop2_x = ?, crop2_y = ?, crop2_w = ?, crop2_h = ?
                WHERE image_name = ?
            """, updates)
            db.conn.commit()

        print(f"Split {len(updates)} images into two crop regions")
        if skipped_spectral:
            print(f"  Skipped {skipped_spectral} spectral doppler images")
        if skipped_degenerate:
            print(f"  Skipped {skipped_degenerate} images whose crop does not straddle the divider")

        append_audit("split_regions.images_split", len(updates))
        append_audit("split_regions.skipped_spectral", skipped_spectral)
        append_audit("split_regions.skipped_degenerate", skipped_degenerate)
