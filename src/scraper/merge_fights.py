# src/scraper/merge_fights.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_csv(path: Path) -> pd.DataFrame:
    # Keep everything as string so IDs/empty cells don't get mangled
    return pd.read_csv(path, dtype=str, keep_default_na=False).replace({np.nan: ""})

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def merge_and_update(
    fights_path: Path,
    events_path: Path,
    out_path: Path,
    strict: bool = False,
) -> None:
    import pandas as pd
    import numpy as np

    fights = load_csv(fights_path)
    events = load_csv(events_path)

    # normalize minimal whitespace on IDs/strings
    for df in (fights, events):
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # quick sanity checks
    overlap = len(set(fights["fight_id"]) & set(events["fight_id"]))
    print(f"[merge] fight_id overlap: {overlap}")

    # how many rows could actually be filled for b_* from events?
    import numpy as np, pandas as pd
    e_pick = events[["fight_id","b_fighter_id","b_fighter_name"]].copy()
    m = fights.merge(e_pick, on="fight_id", how="inner", suffixes=("","_ev"))
    could_fill_b_id   = ((m["b_fighter_id"].eq("") ) & (m["b_fighter_id_ev"].ne(""))).sum()
    could_fill_b_name = ((m["b_fighter_name"].eq("")) & (m["b_fighter_name_ev"].ne(""))).sum()
    print(f"[merge] rows where events has b_fighter_id to fill:   {int(could_fill_b_id)}")
    print(f"[merge] rows where events has b_fighter_name to fill: {int(could_fill_b_name)}")

    if "fight_id" not in fights.columns or "fight_id" not in events.columns:
        raise SystemExit("Both CSVs must contain a 'fight_id' column.")

    print(f"[merge] loaded {len(fights)} rows in 'fights.csv'")
    print(f"[merge] loaded {len(events)} rows in 'fights_event.csv'")

    # Columns to update = intersection (except key)
    common_cols = [c for c in events.columns if c in fights.columns and c != "fight_id"]
    if not common_cols:
        print("[merge] no common columns to update (besides fight_id). Nothing to do.")
        save_csv(fights, out_path)
        return

    # Index by fight_id so .update can align rows
    f = fights.set_index("fight_id")
    e = events.set_index("fight_id")[common_cols].copy()

    # If not strict, treat empty strings in events as "missing" so they don't overwrite
    if not strict:
        e = e.replace({"": pd.NA})

    # Track changes
    before = f[common_cols].copy()

    # In-place update of only the overlapping cells
    f.update(e)

    # Count changes per column
    changed = (f[common_cols].ne(before)).sum()

    # Restore order/column set and save
    out = f.reset_index()[fights.columns]  # keep original fights columns/order
    save_csv(out, out_path.resolve())

    print(f"[merge] wrote {len(out)} rows → {out_path.resolve()}")
    for col in common_cols:
        print(f"[merge] updated {int(changed[col]):6d} values in '{col}'")

def main():
    parser = argparse.ArgumentParser(
        description="Update fights.csv columns using fights_event.csv on fight_id."
    )
    default_base = Path("data/curated")
    parser.add_argument(
        "--fights",
        type=Path,
        default=default_base / "fights.csv",
        help="Path to fights.csv (default: data/curated/fights.csv)",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=default_base / "fights_event.csv",
        help="Path to fights_event.csv (default: data/curated/fights_event.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_base / "fights.csv",
        help="Output path (default: overwrite fights.csv)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Overwrite even with empty strings from fights_event.csv.",
    )
    args = parser.parse_args()
    merge_and_update(args.fights, args.events, args.out, strict=args.strict)

if __name__ == "__main__":
    main()