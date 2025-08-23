from __future__ import annotations
import argparse, re
from typing import Optional, Dict, List
import pandas as pd

from .common.http import get_html
from .common.parse import soup, to_date, height_to_cm, reach_to_cm
from .common.io import load_csv, upsert_csv, update_manifest

FIGHTS_CSV = "data/curated/fights.csv"
FIGHTERS_CSV = "data/curated/fighters.csv"

def _parse_profile(fighter_id: str, url: str) -> Dict:
    html = get_html(url, cache_key=f"fighter_{fighter_id}")
    doc = soup(html)

    name = doc.select_one("span.b-content__title-highlight")
    name = name.get_text(" ", strip=True) if name else None

    dob = None
    height_cm = None
    reach_cm = None
    stance = None

    for li in doc.select("ul.b-list__box-list li"):
        txt = li.get_text(" ", strip=True)
        if ":" not in txt:
            continue
        label, val = [t.strip() for t in txt.split(":", 1)]
        L = label.lower()
        if L == "height":
            height_cm = height_to_cm(val)
        elif L == "reach":
            reach_cm = reach_to_cm(val.replace("in.", "").replace("in", "").strip())
        elif L in ("stance", "striking style"):
            stance = val
        elif L in ("dob", "date of birth"):
            # value often like "Jan 1, 1990"
            d = to_date(val)
            dob = d or dob

    return {
        "fighter_id": fighter_id,
        "name": name,
        "dob": dob,
        "height_cm": height_cm,
        "reach_cm": reach_cm,
        "stance": stance,
        "profile_url": url,
    }

def scrape_fighters(limit: Optional[int] = None) -> pd.DataFrame:
    fights = load_csv(FIGHTS_CSV)
    if fights.empty:
        raise SystemExit("fights.csv not found — run fights scraper first.")
    # Collect unique fighter ids + URLs
    urls = []
    for _, row in fights.iterrows():
        for side in ("r","b"):
            fid = row[f"{side}_fighter_id"]
            # Reconstruct profile URL from known pattern
            if pd.notna(fid):
                urls.append((fid, f"http://www.ufcstats.com/fighter-details/{fid}"))
    uniq = dict(urls)  # dedup by fid

    items = list(uniq.items())
    if limit:
        items = items[:limit]

    rows = []
    for fid, url in items:
        rows.append(_parse_profile(fid, url))

    df = pd.DataFrame(rows, columns=["fighter_id","name","dob","height_cm","reach_cm","stance","profile_url"])
    return df

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scrape fighter profiles → fighters.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=FIGHTERS_CSV)
    args = ap.parse_args(argv)

    df = scrape_fighters(limit=args.limit)
    if df.empty:
        print("[fighters] parsed 0 rows")
        return 0

    upsert_csv(df, args.out, keys=["fighter_id"])
    update_manifest("fighters.csv", rows=len(df))
    print(f"[fighters] wrote {len(df)} rows → {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())