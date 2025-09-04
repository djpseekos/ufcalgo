# src/elo_model/fighter_counts.py
from __future__ import annotations
import argparse
import pandas as pd

def main(fights_csv: str, threshold: int = 5):
    df = pd.read_csv(fights_csv)

    # Exclude no-contests to mirror training
    if "winner_corner" in df.columns:
        df = df[df["winner_corner"] != "NC"].copy()

    # Stack red/blue fighters into one column
    long = pd.concat([
        df[["r_fighter_id"]].rename(columns={"r_fighter_id": "fighter_id"}),
        df[["b_fighter_id"]].rename(columns={"b_fighter_id": "fighter_id"})
    ], ignore_index=True)

    # Drop missing ids just in case
    long = long.dropna(subset=["fighter_id"])

    counts = long.value_counts("fighter_id").rename("n_fights").reset_index()

    total_fighters = len(counts)
    over = counts[counts["n_fights"] > threshold]
    at_or_below = total_fighters - len(over)

    print(f"Total unique fighters (NC excluded): {total_fighters}")
    print(f"Fighters with > {threshold} UFC fights: {len(over)} "
          f"({len(over)/total_fighters:.2%})")
    print(f"Fighters with ≤ {threshold} UFC fights: {at_or_below} "
          f"({at_or_below/total_fighters:.2%})")

    # Optional: quick breakdown for context
    bins = [1,2,3,4,5,6,7,8,9,10,20,50]
    counts["bin"] = pd.cut(counts["n_fights"], bins=[0]+bins+[10**9], right=True)
    dist = counts["bin"].value_counts().sort_index()
    print("\nDistribution (number of fighters by fights played):")
    for interval, freq in dist.items():
        lo = int(interval.left) + 1  # since we used (0,1], (1,2], ...
        hi = "∞" if interval.right > 1000 else int(interval.right)
        print(f"  {lo:>2}–{hi}: {freq}")

    # If you want to save the per-fighter table:
    counts.sort_values("n_fights", ascending=False).to_csv(
        "outputs/fighter_fight_counts.csv", index=False
    )
    print("\nSaved per-fighter counts to outputs/fighter_fight_counts.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Count UFC fights per fighter.")
    p.add_argument("--fights", required=True, help="Path to fights.csv")
    p.add_argument("--threshold", type=int, default=5, help="Count fighters with > threshold fights")
    args = p.parse_args()
    main(args.fights, args.threshold)