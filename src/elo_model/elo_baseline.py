# src/elo_model/elo_sweep.py
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

def eval_fixed_k(df: pd.DataFrame, K: float) -> float:
    """Binary accuracy for fixed-K Elo (ignores draws & NCs)."""
    ratings = defaultdict(lambda: 1500.0)
    preds, outcomes = [], []

    for _, r in df.iterrows():
        fi, fj = r["r_fighter_id"], r["b_fighter_id"]
        Ri, Rj = ratings[fi], ratings[fj]

        # Expected score for red
        Ei = 1.0 / (1.0 + 10 ** ((Rj - Ri) / 400.0))

        # Pred class (red wins if Ei > 0.5)
        preds.append(1 if Ei > 0.5 else 0)
        outcomes.append(1 if r["winner_corner"] == "R" else 0)

        # Update
        Si = 1.0 if r["winner_corner"] == "R" else 0.0
        Sj = 1.0 - Si
        ratings[fi] = Ri + K * (Si - Ei)
        ratings[fj] = Rj + K * (Sj - (1.0 - Ei))

    return float(np.mean(np.array(preds) == np.array(outcomes)))

def main(fights_csv: str, events_csv: str, Ks: list[float], out_csv: str | None):
    fights = pd.read_csv(fights_csv)
    events = pd.read_csv(events_csv, parse_dates=["date"])
    df = fights.merge(events[["event_id", "date"]], on="event_id", how="left")

    # Keep only decisive fights (ignore draws and NCs for binary accuracy)
    df = df[df["winner_corner"].isin(["R", "B"])].copy()
    df = df.sort_values("date").reset_index(drop=True)

    rows = []
    for K in Ks:
        acc = eval_fixed_k(df, K)
        rows.append({"K": K, "accuracy": acc})
        print(f"Elo(K={K}) accuracy: {acc:.3f}")

    res = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    print("\nTop K values by accuracy:")
    print(res.head(10).to_string(index=False, formatters={"K": "{:.1f}".format, "accuracy": "{:.3f}".format}))

    if out_csv:
        res.to_csv(out_csv, index=False)
        print(f"\nSaved sweep results to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sweep fixed-K Elo accuracies.")
    ap.add_argument("--fights", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default="outputs/elo_k_sweep.csv")
    ap.add_argument(
        "--Ks",
        nargs="+",
        type=float,
        default=[8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 80, 96],
        help="List of K values to sweep",
    )
    args = ap.parse_args()
    main(args.fights, args.events, args.Ks, args.out)