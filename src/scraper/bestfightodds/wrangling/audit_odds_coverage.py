# src/scraper/audit_odds_coverage.py
from __future__ import annotations

import argparse
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Set, Dict

import pandas as pd


def norm_name(s: str) -> str:
    """Lowercase, strip accents, keep letters/digits/spaces/['-], collapse spaces."""
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


DAY_SUFFIX = re.compile(r"(st|nd|rd|th)\b", re.IGNORECASE)

def parse_date_any(s: str) -> Optional[datetime.date]:
    """
    Parse either ISO 'YYYY-MM-DD' or BFO-style 'Apr 3rd 2009'/'October 5th 2025'.
    Return a date() or None.
    """
    if not s or pd.isna(s):
        return None
    txt = str(s).strip()
    # ISO fast path
    try:
        return datetime.strptime(txt, "%Y-%m-%d").date()
    except Exception:
        pass
    # Remove ordinals and try month formats
    cleaned = DAY_SUFFIX.sub("", txt)
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def build_name_to_dates(df: pd.DataFrame, name_col: str, date_col: str) -> Dict[str, Set[datetime.date]]:
    m: Dict[str, Set[datetime.date]] = {}
    for _, row in df[[name_col, date_col]].fillna("").iterrows():
        n = norm_name(row[name_col])
        d = parse_date_any(row[date_col])
        if not n or not d:
            continue
        m.setdefault(n, set()).add(d)
    return m


def matches_within(date_set: Set[datetime.date], d: Optional[datetime.date], tol: int) -> int:
    """
    Return offset days used for a match (0, ±1, …), or 999 if no match.
    """
    if not date_set or not d:
        return 999
    if d in date_set:
        return 0
    for k in range(1, tol + 1):
        if (d - timedelta(days=k)) in date_set:
            return -k
        if (d + timedelta(days=k)) in date_set:
            return +k
    return 999


def main():
    ap = argparse.ArgumentParser(
        description="Audit coverage between fights.csv and odds (crude & cleaned). "
                    "Reports missing fighters and per-fight matching (±N day tolerance)."
    )
    ap.add_argument("--fights", default="data/curated/fights.csv")
    ap.add_argument("--odds-crude", default="data/curated/fighters_odds_bfo.csv",
                    help="Raw odds (date column may be 'date'); ISO also OK.")
    ap.add_argument("--odds-clean", default="data/curated/fighters_odds_bfo_iso.csv",
                    help="Cleaned odds after filtering (must have 'date_iso' or 'date').")
    ap.add_argument("--tolerance-days", type=int, default=1,
                    help="Date tolerance when auditing per-fight matches (default: 1)")
    ap.add_argument("--show-missing", type=int, default=25,
                    help="Print up to N sample missing fighter names")
    args = ap.parse_args()

    fights_path = Path(args.fights)
    crude_path = Path(args.odds_crude)
    clean_path = Path(args.odds_clean)

    # Load CSVs
    fights = pd.read_csv(fights_path, dtype=str).fillna("")
    crude = pd.read_csv(crude_path, dtype=str).fillna("")
    clean = pd.read_csv(clean_path, dtype=str).fillna("")

    # --- Validate minimal columns
    need_fights = {"date", "r_fighter_name", "b_fighter_name"}
    if not need_fights.issubset(fights.columns):
        raise SystemExit(f"[ERR] fights.csv missing columns: {sorted(need_fights - set(fights.columns))}")

    # Determine date column names in odds files
    def pick_date_col(df: pd.DataFrame) -> str:
        if "date_iso" in df.columns:
            return "date_iso"
        if "date" in df.columns:
            return "date"
        raise SystemExit("[ERR] odds file needs a 'date_iso' or 'date' column")

    crude_date_col = pick_date_col(crude)
    clean_date_col = pick_date_col(clean)

    # --- Fighter name sets
    fighters_in_fights = set(
        norm_name(x) for x in pd.concat([fights["r_fighter_name"], fights["b_fighter_name"]], ignore_index=True).tolist()
        if x
    )
    fighters_in_crude = set(norm_name(x) for x in crude["fighter_name"].tolist() if x)
    fighters_in_clean = set(norm_name(x) for x in clean["fighter_name"].tolist() if x)

    miss_vs_crude = sorted(fighters_in_fights - fighters_in_crude)
    miss_vs_clean = sorted(fighters_in_fights - fighters_in_clean)

    print("\n=== Fighter presence (unique normalized names) ===")
    print(f"fighters.csv : {len(fighters_in_fights)}")
    print(f"odds_crude   : {len(fighters_in_crude)}   missing from crude: {len(miss_vs_crude)}")
    if miss_vs_crude:
        print("  e.g.:", ", ".join(miss_vs_crude[:args.show_missing]))
    print(f"odds_clean   : {len(fighters_in_clean)}   missing from clean: {len(miss_vs_clean)}")
    if miss_vs_clean:
        print("  e.g.:", ", ".join(miss_vs_clean[:args.show_missing]))

    # --- Expected rows vs actual
    total_fights = len(fights)
    expected_odds_rows = 2 * total_fights
    actual_clean_rows = len(clean)
    shortfall = expected_odds_rows - actual_clean_rows

    print("\n=== Rows expectation ===")
    print(f"fights.csv rows                 : {total_fights}")
    print(f"expected odds rows (2x fights)  : {expected_odds_rows}")
    print(f"cleaned odds rows               : {actual_clean_rows}")
    print(f"SHORTFALL                       : {shortfall}")

    # --- Build name->dates maps for per-fight matching
    # Use cleaned odds for primary audit; crude for comparison to detect over-filtering.
    clean_map = build_name_to_dates(clean, "fighter_name", clean_date_col)
    crude_map = build_name_to_dates(crude, "fighter_name", crude_date_col)

    # --- Per-fight coverage audit (match each side within ±tol)
    tol = max(0, int(args.tolerance_days))
    two_matched = one_matched = zero_matched = 0
    two_matched_crude = one_matched_crude = zero_matched_crude = 0

    # No-contest heuristic: look at 'method' (or empty winner_corner)
    method_col = "method" if "method" in fights.columns else None
    nc_flags = []
    for _, r in fights.iterrows():
        rname = norm_name(r["r_fighter_name"])
        bname = norm_name(r["b_fighter_name"])
        d = parse_date_any(r["date"])

        # cleaned odds matching
        r_off = matches_within(clean_map.get(rname, set()), d, tol)
        b_off = matches_within(clean_map.get(bname, set()), d, tol)
        matched = sum(1 for off in (r_off, b_off) if off != 999)
        if matched == 2:
            two_matched += 1
        elif matched == 1:
            one_matched += 1
        else:
            zero_matched += 1

        # crude odds matching (to detect over-filtering)
        r_off_c = matches_within(crude_map.get(rname, set()), d, tol)
        b_off_c = matches_within(crude_map.get(bname, set()), d, tol)
        matched_c = sum(1 for off in (r_off_c, b_off_c) if off != 999)
        if matched_c == 2:
            two_matched_crude += 1
        elif matched_c == 1:
            one_matched_crude += 1
        else:
            zero_matched_crude += 1

        # NC flag
        is_nc = False
        if method_col:
            if "no contest" in r[method_col].lower():
                is_nc = True
        if not is_nc and "winner_corner" in fights.columns:
            # sometimes NC has blank winner corner
            if (r["winner_corner"] or "").strip() == "":
                # avoid mislabeling scheduled/never-happened; best-effort heuristic
                if method_col and r[method_col]:
                    is_nc = True
        nc_flags.append(is_nc)

    print("\n=== Per-fight coverage (date tolerance ±{} days) ===".format(tol))
    print("CLEANED odds match per fight:")
    print(f"  both fighters matched : {two_matched}")
    print(f"  one fighter matched   : {one_matched}")
    print(f"  zero matched          : {zero_matched}")

    print("\nCRUDE odds match per fight (pre-filter; helps spot over-filtering):")
    print(f"  both fighters matched : {two_matched_crude}")
    print(f"  one fighter matched   : {one_matched_crude}")
    print(f"  zero matched          : {zero_matched_crude}")

    # Over-filtering signal: if crude matches >> cleaned matches, our cleaning step is too strict
    delta_two = two_matched_crude - two_matched
    delta_one = one_matched_crude - one_matched
    delta_zero = zero_matched - zero_matched_crude
    print("\nOver-filtering signal (crude - cleaned):")
    print(f"  Δ both matched : {delta_two:+}")
    print(f"  Δ one matched  : {delta_one:+}")
    print(f"  Δ zero matched : {delta_zero:+}")

    # No Contest bias
    fights["__nc__"] = nc_flags
    nc_total = fights["__nc__"].sum()
    non_nc_total = len(fights) - nc_total

    # Coverage for NC vs non-NC
    def coverage_for(mask):
        idx = fights.index[mask]
        # quick recompute for those indices
        c2 = c1 = c0 = 0
        for i in idx:
            r = fights.loc[i]
            rname = norm_name(r["r_fighter_name"])
            bname = norm_name(r["b_fighter_name"])
            d = parse_date_any(r["date"])
            r_off = matches_within(clean_map.get(rname, set()), d, tol)
            b_off = matches_within(clean_map.get(bname, set()), d, tol)
            m = sum(1 for off in (r_off, b_off) if off != 999)
            if m == 2: c2 += 1
            elif m == 1: c1 += 1
            else: c0 += 1
        return c2, c1, c0, len(idx)

    c_nc = coverage_for(fights["__nc__"])
    c_non = coverage_for(~fights["__nc__"])

    print("\n=== No Contest vs non-NC coverage (cleaned) ===")
    print(f"NC fights    : total={c_nc[3]}  both={c_nc[0]}  one={c_nc[1]}  zero={c_nc[2]}")
    print(f"Non-NC fights: total={c_non[3]} both={c_non[0]} one={c_non[1]} zero={c_non[2]}")

    print("\nDone.")


if __name__ == "__main__":
    main()