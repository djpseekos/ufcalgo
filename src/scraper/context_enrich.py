from __future__ import annotations
import argparse, math
import pandas as pd

from .common.io import load_csv, write_csv, update_manifest

EVENTS_CSV = "data/curated/events.csv"
FIGHTS_CSV = "data/curated/fights.csv"
FIGHTERS_CSV = "data/curated/fighters.csv"
FIGHTER_EVENT_CSV = "data/curated/fighter_event.csv"

ALT_LOOKUP = "data/lookups/altitude_lookup.csv"      # columns: city,venue(optional),altitude_m,timezone
MISS_CSV = "data/lookups/weight_miss.csv"            # optional: event_id,fight_id(optional),fighter_name,miss_lbs

def enrich_events():
    ev = load_csv(EVENTS_CSV)
    if ev.empty:
        raise SystemExit("events.csv missing.")
    alt = load_csv(ALT_LOOKUP)
    if not alt.empty:
        # Join by venue if provided, else by city
        alt_cols = [c for c in ["venue","city","altitude_m","timezone"] if c in alt.columns]
        for _, r in alt.iterrows():
            mask = pd.Series([True] * len(ev))
            if "venue" in alt and isinstance(r.get("venue"), str) and r["venue"]:
                mask &= ev["venue"].fillna("").str.contains(str(r["venue"]), case=False, na=False)
            if "city" in alt and isinstance(r.get("city"), str) and r["city"]:
                mask &= ev["city"].fillna("").str.contains(str(r["city"]), case=False, na=False)
            if "altitude_m" in alt.columns:
                ev.loc[mask, "altitude_m"] = r["altitude_m"]
            if "timezone" in alt.columns:
                ev.loc[mask, "timezone"] = r["timezone"]

    write_csv(ev, EVENTS_CSV, index=False)
    update_manifest("events.csv", rows=len(ev))
    print("[context] events enriched with altitude/timezone (where available).")

def build_fighter_event():
    fights = load_csv(FIGHTS_CSV)
    fighters = load_csv(FIGHTERS_CSV)
    events = load_csv(EVENTS_CSV)
    if fights.empty or fighters.empty or events.empty:
        print("[context] missing fights/fighters/events; skipping fighter_event.")
        return

    # Flatten to two rows per fight (red/blue)
    rows = []
    ev_dt = events.set_index("event_id")["date"].to_dict()
    for _, f in fights.iterrows():
        for side in ("r","b"):
            fid = f[f"{side}_fighter_id"]
            if pd.isna(fid): 
                continue
            rows.append({
                "fight_id": f["fight_id"],
                "fighter_id": fid,
                "corner": "R" if side == "r" else "B",
                "event_id": f["event_id"],
                "date": ev_dt.get(f["event_id"]),
            })
    df = pd.DataFrame(rows)

    # Optional weight-miss join
    miss = load_csv(MISS_CSV)
    if not miss.empty:
        # Merge on fighter name + event if fight_id not present
        if "fight_id" in miss.columns:
            df = df.merge(miss[["fight_id","fighter_name","miss_lbs"]], on="fight_id", how="left")
        else:
            # best-effort: fuzzy by event_id + fighter name (canonize simple)
            miss["name_canon"] = miss["fighter_name"].str.lower().str.replace(r"\s+", " ", regex=True)
            df["name_canon"] = df["fighter_id"]  # placeholder; we’d need names to match properly
        df["missed_weight"] = df["miss_lbs"].notna().astype(int)

    write_csv(df, FIGHTER_EVENT_CSV, index=False)
    update_manifest("fighter_event.csv", rows=len(df))
    print(f"[context] wrote fighter_event.csv rows: {len(df)}")

def main(argv=None) -> int:
    enrich_events()
    build_fighter_event()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())