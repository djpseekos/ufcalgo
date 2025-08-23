from __future__ import annotations
import argparse, re
from typing import Optional, List, Dict
import pandas as pd

from .common.http import get_html
from .common.parse import soup
from .common.io import load_csv, upsert_csv, update_manifest
from .common.names import canon, pair_key

FIGHTS_CSV = "data/curated/fights.csv"
ODDS_CSV = "data/curated/odds_snapshot.csv"
BFO_LOOKUP = "data/lookups/bfo_event_urls.csv"  # columns: event_id,event_url

def _moneyline_to_decimal(ml: int | str) -> Optional[float]:
    try:
        ml = int(str(ml).strip())
    except:
        return None
    if ml > 0:
        return 1.0 + ml / 100.0
    else:
        return 1.0 + 100.0 / abs(ml)

def _de_vig_two_way(p1: float, p2: float) -> tuple[float,float]:
    # Normalize two implied probs so they sum to 1 (basic de-vig)
    s = p1 + p2
    if s <= 0:
        return p1, p2
    return p1/s, p2/s

def _parse_bfo_event(event_url: str) -> List[Dict]:
    """
    Parse a BestFightOdds event page, returning entries with book/open/close moneylines for each side.
    We keep fighter display names here; mapping to fight_id is done later by name matching.
    """
    html = get_html(event_url, cache_key=f"bfo_{hash(event_url)}", ttl_hours=6)
    doc = soup(html)

    fights: list[Dict] = []

    # BFO markup changes; we look for blocks that contain two fighter names and open/close lines.
    # Heuristic: rows with class containing 'fight' and nested book odds tables.
    for block in doc.find_all(True, class_=re.compile(r"fight|matchup", re.I)):
        txt = block.get_text(" ", strip=True)
        # Grab fighter names: look for two largest <a> or strong tags
        names = [canon(a.get_text(" ", strip=True)) for a in block.select("a")][:2]
        if len(names) < 2:
            continue

        # For each sportsbook row, try to find open/close
        for tr in block.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td","th"])]
            if len(cells) < 3:
                continue
            book = cells[0].strip()
            if not book or book.lower() in ("book", "sportsbook"):
                continue

            # Try to find "open" and "close" moneylines in the rest of the cells
            # Many BFO tables format as: book | open | close | best | ...
            ml_open = None
            ml_close = None
            for c in cells[1:]:
                m = re.search(r"([+-]\d{2,4})", c.replace(" ", ""))
                if m and ml_open is None:
                    ml_open = int(m.group(1))
                elif m and ml_close is None:
                    ml_close = int(m.group(1))

            if ml_open is None and ml_close is None:
                continue

            fights.append({
                "book": book,
                "name_a": names[0],
                "name_b": names[1],
                "ml_open": ml_open,
                "ml_close": ml_close,
            })

    return fights

def _map_to_fight_ids(rows: List[Dict], fights_df: pd.DataFrame, event_id: str) -> List[Dict]:
    out = []
    # Build an index of canonical pair -> fight_id for this event
    sub = fights_df[fights_df["event_id"] == event_id]
    idx = {}
    for _, r in sub.iterrows():
        key = pair_key(r["r_fighter_name"], r["b_fighter_name"])
        idx[key] = r["fight_id"]

    for r in rows:
        key = pair_key(r["name_a"], r["name_b"])
        fight_id = idx.get(key)
        if not fight_id:
            # try reversed (pair_key sorts but canon may differ)
            fight_id = idx.get(pair_key(r["name_b"], r["name_a"]))
        if not fight_id:
            continue

        # Compute implied probabilities
        dec_a_open = _moneyline_to_decimal(r["ml_open"]) if r["ml_open"] is not None else None
        dec_a_close = _moneyline_to_decimal(r["ml_close"]) if r["ml_close"] is not None else None
        # Assume symmetrical (two-way market); derive opponent side implicitly
        # We store only side A; downstream can reconstruct side B by 1-p
        recs = []
        for tag, dec in [("open", dec_a_open), ("close", dec_a_close)]:
            if dec is None:
                continue
            p_raw_a = 1.0 / dec
            p_raw_b = 1.0 - p_raw_a
            p_fair_a, p_fair_b = _de_vig_two_way(p_raw_a, p_raw_b)
            recs.append({
                "fight_id": fight_id,
                "book": r["book"],
                "tag": tag,
                "price_format": "moneyline",
                "moneyline": r[f"ml_{tag}"],
                "decimal_price": dec,
                "implied_prob_raw": p_raw_a,
                "implied_prob_fair": p_fair_a,
            })
        out.extend(recs)
    return out

def scrape_bfo() -> pd.DataFrame:
    fights_df = load_csv(FIGHTS_CSV)
    if fights_df.empty:
        raise SystemExit("fights.csv not found — run event_fights first.")
    m = load_csv(BFO_LOOKUP)
    if m.empty or "event_id" not in m or "event_url" not in m:
        raise SystemExit("Missing lookup: data/lookups/bfo_event_urls.csv with columns: event_id,event_url")

    all_rows = []
    for _, row in m.iterrows():
        event_id = row["event_id"]
        url = row["event_url"]
        parsed = _parse_bfo_event(url)
        mapped = _map_to_fight_ids(parsed, fights_df, event_id)
        all_rows.extend(mapped)

    df = pd.DataFrame(all_rows, columns=[
        "fight_id","book","tag","price_format","moneyline","decimal_price","implied_prob_raw","implied_prob_fair"
    ])
    return df

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scrape BestFightOdds (with event URL lookup) → odds_snapshot.csv")
    ap.add_argument("--out", default=ODDS_CSV)
    args = ap.parse_args(argv)

    df = scrape_bfo()
    if df.empty:
        print("[odds] parsed 0 rows; check lookup CSV and site structure.")
        return 0

    upsert_csv(df, args.out, keys=["fight_id","book","tag"])
    update_manifest("odds_snapshot.csv", rows=len(df))
    print(f"[odds] wrote {len(df)} rows → {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())