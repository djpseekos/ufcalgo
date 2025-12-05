# src/scraper/filter_odds_by_fights.py
from __future__ import annotations

import argparse
import os
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def norm_name(s: str) -> str:
    """Lowercase, strip accents, keep letters/digits/spaces/['-], collapse spaces."""
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_date(s: str):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Keep only odds rows whose (fighter_name, date_iso) exist in fights.csv "
                    "(as red/blue on same date, or within ±N days)."
    )
    ap.add_argument("--odds", default="data/curated/fighters_odds_bfo_iso.csv",
                    help="Input/output odds CSV (will be filtered in place)")
    ap.add_argument("--fights", default="data/curated/fights.csv",
                    help="Fights CSV (must include date, r_fighter_name, b_fighter_name)")
    ap.add_argument("--tolerance-days", type=int, default=1,
                    help="Date tolerance window; keep if within ±N days (default: 1)")
    args = ap.parse_args()

    odds_path = Path(args.odds)
    fights_path = Path(args.fights)
    if not odds_path.exists():
        raise SystemExit(f"[ERR] Not found: {odds_path}")
    if not fights_path.exists():
        raise SystemExit(f"[ERR] Not found: {fights_path}")

    # Load
    odds = pd.read_csv(odds_path, dtype=str)
    fights = pd.read_csv(fights_path, dtype=str)

    # Column checks
    for col in ("fighter_name", "date_iso"):
        if col not in odds.columns:
            raise SystemExit(f"[ERR] {odds_path} missing '{col}' column")
    for col in ("date", "r_fighter_name", "b_fighter_name"):
        if col not in fights.columns:
            raise SystemExit(f"[ERR] {fights_path} missing '{col}' column")

    # Build name -> set(date) map from fights (both corners)
    fights = fights.fillna("")
    name_to_dates: dict[str, set] = {}

    def add_pair(name: str, date_str: str):
        d = to_date(date_str)
        if not d:
            return
        key = norm_name(name)
        name_to_dates.setdefault(key, set()).add(d)

    for _, row in fights.iterrows():
        add_pair(row["r_fighter_name"], row["date"])
        add_pair(row["b_fighter_name"], row["date"])

    # Helper: does (name, date) exist within ±t days?
    tol = abs(int(args.tolerance_days))

    def matches_within(name: str, date_str: str) -> tuple[bool, int]:
        """Return (matched, offset_days_used). offset_days_used = 0/±1/… or 999 if no match."""
        d = to_date(date_str)
        if not d:
            return (False, 999)
        dates = name_to_dates.get(norm_name(name))
        if not dates:
            return (False, 999)
        # exact first
        if d in dates:
            return (True, 0)
        # then ±N
        for k in range(1, tol + 1):
            if d - timedelta(days=k) in dates:
                return (True, -k)
            if d + timedelta(days=k) in dates:
                return (True, +k)
        return (False, 999)

    # Apply filter
    odds = odds.fillna("")
    before = len(odds)
    keep_flags = []
    offsets = []  # for reporting: 0, ±1, etc.

    for _, r in odds.iterrows():
        ok, off = matches_within(r["fighter_name"], r["date_iso"])
        keep_flags.append(ok)
        offsets.append(off)

    filtered = odds.loc[keep_flags].copy()
    after = len(filtered)

    # Stats
    exact = sum(1 for o in offsets if o == 0)
    tol_kept = sum(1 for o in offsets if o != 999 and o != 0)
    removed = before - after

    # Write back in place
    tmp = odds_path.with_suffix(odds_path.suffix + ".tmp")
    filtered.to_csv(tmp, index=False)
    os.replace(tmp, odds_path)

    print(f"[OK] Filtered odds by fights with ±{tol}-day tolerance.")
    print(f"     Kept {after}/{before} rows; removed {removed}.")
    print(f"     Matches: exact={exact}, within_tolerance={tol_kept}")

if __name__ == "__main__":
    main()