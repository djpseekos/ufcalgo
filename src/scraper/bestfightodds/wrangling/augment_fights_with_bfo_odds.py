# src/scraper/bestfightodds/wrangling/augment_fights_with_bfo_odds.py
from __future__ import annotations

import argparse
import csv
import html
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_FIGHTS = "data/curated/fights.csv"
DEFAULT_ODDS   = "data/curated/fighters_odds_bfo_iso.csv"
DEFAULT_OUT    = DEFAULT_FIGHTS


# ---------------------- helpers ----------------------

def norm_name(s: str) -> str:
    s = html.unescape(str(s or ""))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def to_iso_date(s: str):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None


# ---------------------- loader / index ----------------------

def build_odds_index(odds_df: pd.DataFrame) -> Dict[str, Dict[str, Tuple[str, str, str]]]:
    """
    Build: name_norm -> date_iso(str) -> (open, close_low, close_high)
    """
    idx: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
    for _, r in odds_df.iterrows():
        name = norm_name(r.get("fighter_name", ""))
        d = str(r.get("date_iso", "")).strip()
        if not name or not d:
            continue
        open_s = str(r.get("open", "") or "")
        low_s  = str(r.get("close_low", "") or "")
        high_s = str(r.get("close_high", "") or "")
        idx.setdefault(name, {})[d] = (open_s, low_s, high_s)
    return idx


def lookup_with_tolerance(
    idx: Dict[str, Dict[str, Tuple[str, str, str]]],
    fighter_name: str,
    event_date_iso: str,
    tol_days: int = 1,
) -> Optional[Tuple[str, str, str]]:
    """
    Try exact date first; if absent, search within ±tol_days and
    return the nearest by absolute day difference.
    """
    n = norm_name(fighter_name)
    date = to_iso_date(event_date_iso)
    if not n or not date:
        return None

    by_date = idx.get(n)
    if not by_date:
        return None

    # exact
    exact = by_date.get(event_date_iso)
    if exact:
        return exact

    # tolerance
    best: Optional[Tuple[int, Tuple[str, str, str]]] = None
    for ds, odds in by_date.items():
        d = to_iso_date(ds)
        if not d:
            continue
        delta = abs((d - date).days)
        if delta <= tol_days:
            if best is None or delta < best[0]:
                best = (delta, odds)

    return best[1] if best else None


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Augment fights.csv with BestFightOdds odds columns (per fighter)."
    )
    ap.add_argument("--fights", default=DEFAULT_FIGHTS)
    ap.add_argument("--odds", default=DEFAULT_ODDS)
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output path (default overwrites fights.csv)")
    ap.add_argument("--tolerance-days", type=int, default=1, help="Date tolerance for odds match (±days)")
    args = ap.parse_args()

    fights = pd.read_csv(args.fights, dtype=str).fillna("")
    odds   = pd.read_csv(args.odds, dtype=str).fillna("")

    idx = build_odds_index(odds)

    # Ensure columns exist; we’ll overwrite if present
    for col in [
        "r_bfo_open", "r_bfo_close_a", "r_bfo_close_b",
        "b_bfo_open", "b_bfo_close_a", "b_bfo_close_b",
    ]:
        if col not in fights.columns:
            fights[col] = ""

    filled_r = filled_b = 0
    total = len(fights)

    for i, r in fights.iterrows():
        date_iso = str(r.get("date", "")).strip()
        r_name = str(r.get("r_fighter_name", "")).strip()
        b_name = str(r.get("b_fighter_name", "")).strip()

        # Red
        ro = lookup_with_tolerance(idx, r_name, date_iso, args.tolerance_days)
        if ro:
            fights.at[i, "r_bfo_open"]    = ro[0]
            fights.at[i, "r_bfo_close_a"] = ro[1]
            fights.at[i, "r_bfo_close_b"] = ro[2]
            filled_r += 1

        # Blue
        bo = lookup_with_tolerance(idx, b_name, date_iso, args.tolerance_days)
        if bo:
            fights.at[i, "b_bfo_open"]    = bo[0]
            fights.at[i, "b_bfo_close_a"] = bo[1]
            fights.at[i, "b_bfo_close_b"] = bo[2]
            filled_b += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    fights.to_csv(tmp, index=False)
    Path(tmp).replace(out_path)

    print("=== BFO odds augmentation complete ===")
    print(f"Rows processed     : {total}")
    print(f"Red side filled    : {filled_r}")
    print(f"Blue side filled   : {filled_b}")
    print(f"Wrote → {out_path}")

if __name__ == "__main__":
    main()