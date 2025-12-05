# src/scraper/augment_fights_with_event_info.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Append event fields (date, city, country, venue) to fights.csv via event_id."
    )
    ap.add_argument("--fights", default="data/curated/fights.csv", help="Path to fights.csv (will be overwritten)")
    ap.add_argument("--events", default="data/curated/events.csv", help="Path to events.csv (source of fields)")
    args = ap.parse_args()

    fights_path = Path(args.fights)
    events_path = Path(args.events)

    if not fights_path.exists():
        sys.exit(f"[ERR] Fights file not found: {fights_path}")
    if not events_path.exists():
        sys.exit(f"[ERR] Events file not found: {events_path}")

    # Load
    fights = pd.read_csv(fights_path, dtype=str)  # preserve IDs exactly
    events = pd.read_csv(events_path, dtype=str)

    if "event_id" not in fights.columns:
        sys.exit("[ERR] fights.csv is missing 'event_id' column")
    need_cols = {"event_id", "date", "city", "country", "venue"}
    missing = need_cols - set(events.columns)
    if missing:
        sys.exit(f"[ERR] events.csv missing columns: {sorted(missing)}")

    # Keep only the columns we need from events
    events_keep = events.loc[:, ["event_id", "date", "city", "country", "venue"]].copy()

    # Merge (left join to keep all fights)
    merged = fights.merge(events_keep, on="event_id", how="left", validate="m:1")

    # Ensure new columns are appended at the END in the requested order
    base_cols = [c for c in fights.columns if c in merged.columns]
    new_cols = ["date", "city", "country", "venue"]
    # Remove any that already existed (unlikely) to avoid duplicates in final order
    new_cols = [c for c in new_cols if c not in base_cols]
    final_cols = base_cols + new_cols
    merged = merged.loc[:, final_cols]

    # Replace NaN with empty strings for the new fields (cleaner CSV)
    merged[new_cols] = merged[new_cols].fillna("")

    # Write back atomically to the SAME FILE (no new file)
    tmp_path = fights_path.with_suffix(fights_path.suffix + ".tmp")
    merged.to_csv(tmp_path, index=False)
    os.replace(tmp_path, fights_path)

    # Simple report
    total = len(merged)
    missing_ev = merged["date"].eq("").sum() if "date" in merged.columns else 0
    print(f"[OK] Updated {fights_path} with event fields. Rows: {total}. "
          f"Missing event info for {missing_ev} fights.")


if __name__ == "__main__":
    main()