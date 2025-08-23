from __future__ import annotations
import argparse, time
import pandas as pd
from typing import Optional, Dict

from .common.http import get_html
from .common.parse import soup
from .common.io import load_csv, upsert_csv, update_manifest

FIGHTERS_CSV = "data/curated/fighters.csv"
TAPOLOGY_LOOKUP = "data/lookups/tapology_fighter_urls.csv"  # columns: fighter_id,profile_url

def _parse_tapology_profile(url: str) -> Dict[str, Optional[str]]:
    html = get_html(url, cache_key=f"tapology_{hash(url)}", ttl_hours=24)
    doc = soup(html)
    camp = None
    stance = None

    # Heuristics: Tapology profiles often list "Fighting out of" / "Gym/Team"
    text = doc.get_text(" ", strip=True)
    # very loose patterns:
    for label in ["Team:", "Gym:", "Fighting out of:", "Association:"]:
        if label in text:
            idx = text.find(label)
            camp = text[idx+len(label): idx+len(label)+80].split("  ")[0].strip()
            break

    # stance often appears as "Stance: Orthodox/Southpaw/Switch"
    if "Stance:" in text:
        idx = text.find("Stance:")
        stance = text[idx+len("Stance:"): idx+len("Stance:")+30].split("  ")[0].strip()

    return {"camp_name": camp, "stance_alt": stance}

def fill_from_tapology(limit: Optional[int] = None) -> pd.DataFrame:
    f = load_csv(FIGHTERS_CSV)
    map_df = load_csv(TAPOLOGY_LOOKUP)
    if f.empty or map_df.empty:
        raise SystemExit("Need fighters.csv and tapology_fighter_urls.csv")

    need = f[f["stance"].isna() | (f.get("camp_name") is None)]
    df = need.merge(map_df, on="fighter_id", how="inner")
    if limit:
        df = df.head(limit)

    updates = []
    for _, row in df.iterrows():
        parsed = _parse_tapology_profile(row["profile_url"])
        updates.append({
            "fighter_id": row["fighter_id"],
            "camp_name": parsed.get("camp_name"),
            "stance": row["stance"] or parsed.get("stance_alt"),
        })
        time.sleep(0.5)  # be polite

    upd = pd.DataFrame(updates)
    if upd.empty:
        return upd

    # Upsert: overwrite only nulls
    cur = f.set_index("fighter_id")
    for _, r in upd.set_index("fighter_id").iterrows():
        if pd.isna(cur.loc[r.name, "stance"]) and pd.notna(r["stance"]):
            cur.loc[r.name, "stance"] = r["stance"]
        if ("camp_name" in cur.columns) and (pd.isna(cur.loc[r.name, "camp_name"])) and pd.notna(r["camp_name"]):
            cur.loc[r.name, "camp_name"] = r["camp_name"]

    out = cur.reset_index()
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tapology fills for fighters.csv (optional)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    out = fill_from_tapology(limit=args.limit)
    if out.empty:
        print("[tapology] nothing to update.")
        return 0

    upsert_csv(out, FIGHTERS_CSV, keys=["fighter_id"])
    update_manifest("fighters.csv", rows=len(out))
    print(f"[tapology] updated fighters.csv rows: {len(out)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())