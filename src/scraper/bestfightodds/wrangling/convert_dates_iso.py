# src/scraper/convert_dates_iso.py
from __future__ import annotations

import argparse
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

DAY_SUFFIX = re.compile(r"(st|nd|rd|th)", re.IGNORECASE)

def parse_date(date_str: str) -> str | None:
    """Convert 'Apr 3rd 2009' -> '2009-04-03'. Return None if parsing fails."""
    if not date_str or pd.isna(date_str):
        return None
    cleaned = DAY_SUFFIX.sub("", str(date_str)).strip()
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def main():
    ap = argparse.ArgumentParser(description="Convert BFO date strings to ISO format (yyyy-mm-dd).")
    ap.add_argument("--input", default="data/curated/fighters_odds_bfo.csv",
                    help="Input CSV with original dates")
    ap.add_argument("--output", default="data/curated/fighters_odds_bfo_iso.csv",
                    help="Output CSV with ISO dates")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "date" not in df.columns:
        raise SystemExit(f"{args.input} must have a 'date' column")

    df["date_iso"] = df["date"].apply(parse_date)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[DONE] Wrote {len(df)} rows with ISO dates to {out_path}")

if __name__ == "__main__":
    main()