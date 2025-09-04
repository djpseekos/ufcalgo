from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict

from .train import EloBTDTrainer
from .model import FightRow, btd_probs, K_value

def replay_probs_and_counts(rows: List[FightRow], params: np.ndarray, K_cap: float = 128.0):
    """One pass through fights, returning per-fight: (p_i,p_d,p_j,true_label,count_i,count_j)."""
    a, Kmin, Kmax, nu = params[:4]; beta = params[4:]
    R_class: Dict[Tuple[str,str], float] = {}
    R_global: Dict[str, float] = {}
    class_means: Dict[str, float] = {}
    class_counts: Dict[str, int] = {}
    fight_counts = defaultdict(int)

    out = []  # list of dicts per fight

    for f in rows:
        ki, kj = (f.i, f.wc), (f.j, f.wc)
        # init if needed
        if ki not in R_class:
            mu_c = class_means.get(f.wc, 1500.0)
            R_class[ki] = a * R_global.get(f.i,1500.0) + (1 - a) * mu_c
            class_means[f.wc] = (class_means.get(f.wc,0.0) * class_counts.get(f.wc,0) + R_class[ki]) / (class_counts.get(f.wc,0) + 1)
            class_counts[f.wc] = class_counts.get(f.wc,0) + 1
        if kj not in R_class:
            mu_c = class_means.get(f.wc, 1500.0)
            R_class[kj] = a * R_global.get(f.j,1500.0) + (1 - a) * mu_c
            class_means[f.wc] = (class_means.get(f.wc,0.0) * class_counts.get(f.wc,0) + R_class[kj]) / (class_counts.get(f.wc,0) + 1)
            class_counts[f.wc] = class_counts.get(f.wc,0) + 1

        Ri, Rj = R_class[ki], R_class[kj]
        p_i, p_d, p_j = btd_probs(Ri, Rj, nu)
        true = {"i":0,"d":1,"j":2}[f.outcome]

        out.append({
            "p_i": p_i, "p_d": p_d, "p_j": p_j,
            "true": true,
            "cnt_i": fight_counts[f.i],
            "cnt_j": fight_counts[f.j],
        })

        # update ratings
        S_i = 1.0 if f.outcome == "i" else 0.0 if f.outcome == "j" else 0.5
        S_hat_i = p_i + 0.5 * p_d
        Kt = min(K_value(Kmin, Kmax, beta, f.x), K_cap)
        R_class[ki] = Ri + Kt * (S_i - S_hat_i)
        R_class[kj] = Rj + Kt * ((1.0 - S_i) - (p_j + 0.5 * p_d))

        # increment counts AFTER fight
        fight_counts[f.i] += 1
        fight_counts[f.j] += 1

    return pd.DataFrame(out)

def metric_from_probs(df: pd.DataFrame, min_fights: int):
    mask = (df["cnt_i"] >= min_fights) & (df["cnt_j"] >= min_fights)
    if not mask.any():
        return {"threshold": min_fights, "n": 0, "acc": None, "logloss": None}
    sub = df[mask].copy()
    probs = sub[["p_i","p_d","p_j"]].to_numpy()
    true = sub["true"].to_numpy()
    pred = probs.argmax(axis=1)
    acc = float((pred == true).mean())
    eps = 1e-12
    ll = -float(np.log(np.maximum(probs[np.arange(len(sub)), true], eps)).mean())
    return {"threshold": min_fights, "n": int(mask.sum()), "acc": acc, "logloss": ll}

def fixed_k_elo_subset(fights_csv: str, events_csv: str, K: float, min_fights: int):
    """Binary Elo baseline evaluated only when both fighters already had ≥ min_fights past bouts."""
    fights = pd.read_csv(fights_csv)
    events = pd.read_csv(events_csv, parse_dates=["date"])
    df = fights.merge(events[["event_id","date"]], on="event_id", how="left")
    df = df[df["winner_corner"].isin(["R","B"])].sort_values("date").reset_index(drop=True)

    ratings = defaultdict(lambda: 1500.0)
    counts = defaultdict(int)

    preds, trues = [], []
    for _, r in df.iterrows():
        fi, fj = r["r_fighter_id"], r["b_fighter_id"]
        Ri, Rj = ratings[fi], ratings[fj]
        Ei = 1 / (1 + 10 ** ((Rj - Ri)/400))
        # evaluate only if both had ≥ min_fights
        if counts[fi] >= min_fights and counts[fj] >= min_fights:
            preds.append(1 if Ei > 0.5 else 0)
            trues.append(1 if r["winner_corner"] == "R" else 0)
        # update
        Si = 1.0 if r["winner_corner"]=="R" else 0.0
        ratings[fi] = Ri + K * (Si - Ei)
        ratings[fj] = Rj + K * ((1.0 - Si) - (1.0 - Ei))
        counts[fi] += 1; counts[fj] += 1

    if not preds:
        return None
    return float((np.array(preds) == np.array(trues)).mean())

def main(fights_csv: str, events_csv: str, params_path: str, max_threshold: int = 10):
    trainer = EloBTDTrainer()
    rows = trainer.load_data(fights_csv, events_csv)
    rows = trainer.build_features(rows)
    params = np.load(params_path)

    dfp = replay_probs_and_counts(rows, params, K_cap=trainer.K_cap)

    results = [metric_from_probs(dfp, t) for t in range(0, max_threshold+1)]
    res_df = pd.DataFrame(results)
    print("Accuracy/log-loss by min prior fights (both fighters):")
    print(res_df.to_string(index=False, formatters={"acc":"{:.3f}".format, "logloss":"{:.3f}".format}))

    # Compare to Elo(K=96) on the same filter (binary)
    k96 = fixed_k_elo_subset(fights_csv, events_csv, K=96, min_fights=5)
    if k96 is not None:
        print(f"\nBinary Elo baseline at K=96, min_fights=5 → accuracy: {k96:.3f}")

    # Save for paper
    res_df.to_csv("outputs/accuracy_by_experience.csv", index=False)
    print("\nSaved outputs/accuracy_by_experience.csv")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Experience-threshold accuracy curve")
    ap.add_argument("--fights", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--params", default="outputs/best_params_trval.npy")
    ap.add_argument("--max_threshold", type=int, default=10)
    args = ap.parse_args()
    main(args.fights, args.events, args.params, max_threshold=args.max_threshold)