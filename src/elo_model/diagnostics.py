# src/elo_model/diagnostics.py
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .train import EloBTDTrainer
from .feature_builder import FeatureBuilder
from .helpers import parse_judge_margin, is_decision
from .time_utils import actual_elapsed_seconds
from .model import btd_probs


def _hdr(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_raw_csv(fights_csv: str) -> None:
    """Basic raw CSV sanity checks (no modeling)."""
    _hdr("RAW CSV CHECKS")
    df = pd.read_csv(fights_csv)

    n_total = len(df)
    vc_win = df["winner_corner"].value_counts(dropna=False)
    draw_rate = (df["winner_corner"] == "D").mean()

    print(f"Rows in fights.csv: {n_total}")
    print("\nwinner_corner value counts (incl. NaN):")
    print(vc_win.to_string())

    print(f"\nEmpirical draw rate in raw CSV: {draw_rate:.4%}")

    # scheduled rounds distribution
    if "scheduled_rounds" in df.columns:
        print("\nScheduled rounds distribution (raw):")
        print(df["scheduled_rounds"].value_counts(dropna=False).to_string())

    # Methods with 'decision'
    if "method" in df.columns:
        dec_mask = df["method"].fillna("").str.lower().str.contains("decision")
        print(f"\nRows whose method contains 'decision': {dec_mask.sum()} ({dec_mask.mean():.2%})")


def check_loader_maps_and_filter(trainer: EloBTDTrainer, fights_csv: str, events_csv: str) -> List:
    """Run trainer.load_data and summarize outcomes, dates, NC filter, etc."""
    _hdr("LOADER CHECKS (load_data)")
    rows = trainer.load_data(fights_csv, events_csv)
    n = len(rows)
    print(f"Rows after load & NC filtering: {n}")

    # outcome counts
    cnt = {"i": 0, "j": 0, "d": 0}
    sr_set = set()
    dates = []
    wc_set = set()
    for r in rows:
        cnt[r.outcome] += 1
        sr_set.add(r.scheduled_rounds)
        dates.append(r.date)
        wc_set.add(r.wc)

    print("Outcome counts:", cnt)
    print(f"Empirical draw rate (post-load): {cnt['d'] / max(1, n):.4%}")
    print("Scheduled rounds set (post-load):", sorted(sr_set))
    print(f"Unique weight classes: {len(wc_set)}")

    # date monotonicity
    dates = pd.to_datetime(pd.Series(dates))
    decreasing = (dates.diff().dt.total_seconds().fillna(0) < 0).sum()
    if decreasing > 0:
        print(f"WARNING: {decreasing} date regressions detected after sort (dates not monotonic).")
    else:
        print("Dates are non-decreasing after merge/sort ✅")

    return rows


def check_feature_builder(rows: List) -> Tuple[pd.DataFrame, np.ndarray]:
    """Rebuild features exactly like training and validate z-scores & transforms."""
    _hdr("FEATURE BUILDER CHECKS")
    fb = FeatureBuilder()
    raw = fb.collect_raw_features(rows)
    fb.fit_scalers(raw)
    X = fb.transform(raw)  # list of np arrays, aligned with raw rows

    df = raw.copy()
    Xmat = np.vstack(X)

    # Reconstruct the z/tanh pieces to verify:
    to_z = ["round_early", "experience", "layoff", "x_cards_raw"]
    means = {k: float(df[k].mean()) for k in to_z}
    stds = {k: float(df[k].std(ddof=0)) if float(df[k].std(ddof=0)) > 1e-8 else 1.0 for k in to_z}
    df["round_early_z"] = (df["round_early"] - means["round_early"]) / stds["round_early"]
    df["experience_z"] = (df["experience"] - means["experience"]) / stds["experience"]
    df["layoff_z"] = (df["layoff"] - means["layoff"]) / stds["layoff"]
    df["x_cards_tanhz"] = np.tanh((df["x_cards_raw"] - means["x_cards_raw"]) / stds["x_cards_raw"])

    # Summaries
    def s(col):  # summary line
        m, sd = float(df[col].mean()), float(df[col].std(ddof=0))
        print(f"{col:>18s}: mean={m:+.4f}, std={sd:.4f}")

    print("Z-score sanity (should be ~0 mean, ~1 std):")
    s("round_early_z"); s("experience_z"); s("layoff_z")
    print("\nCards transform (tanh(z)) — bounded in [-1,1], mean near 0:")
    s("x_cards_tanhz")

    # Check judge-score parsing plausibility
    S_list, J_list = [], []
    for sc in df["fight"].map(lambda f: f.judge_scores):
        S, J = parse_judge_margin(sc)
        S_list.append(S); J_list.append(J)
    S_arr, J_arr = np.array(S_list), np.array(J_list)
    print("\nJudge score parsing:")
    print(f"  Share with any parsed totals: {(J_arr > 0).mean():.2%}")
    print(f"  Median sum of abs margins (S|J>0): {np.median(S_arr[J_arr>0]) if (J_arr>0).any() else 0:.2f}")
    print(f"  95th pct S (|J>0): {np.percentile(S_arr[J_arr>0],95) if (J_arr>0).any() else 0:.2f}")

    return df, Xmat


def check_elapsed_time_logic(df_raw_feats: pd.DataFrame) -> None:
    """For 'decision' methods, elapsed time should equal scheduled_rounds*5*60."""
    _hdr("ELAPSED-TIME LOGIC CHECKS")
    # We stored both T_act (raw) and T_star (floored) in raw features
    # Recompute the expected full-time for decisions
    # We need access to original fights to get method/scheduled_rounds (in df_raw_feats we kept them via df_raw_feats["fight"])
    methods = df_raw_feats["fight"].map(lambda f: f.method)
    sr = df_raw_feats["fight"].map(lambda f: f.scheduled_rounds)
    t_act = df_raw_feats["T_act"]
    dec_mask = methods.map(is_decision)

    expected_full = sr * 5 * 60
    ok = (t_act[dec_mask].values == expected_full[dec_mask].values)
    mismatches = (~ok).sum()
    if mismatches > 0:
        print(f"WARNING: {mismatches} decision fights where T_act != scheduled*300")
    else:
        print("Decision fights have full elapsed time as expected ✅")


def probability_sanity(rows: List, sample: int = 50) -> None:
    """Quick check: BTD probabilities are valid and sum to 1 on a sample, using flat 1500 ratings."""
    _hdr("BTD PROBABILITY SANITY")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)

    bad_sum = 0
    for idx in pick:
        Ri, Rj = 1500.0, 1500.0  # flat ratings for sanity
        p_i, p_d, p_j = btd_probs(Ri, Rj, nu=0.01)
        s = p_i + p_d + p_j
        if not (abs(s - 1.0) < 1e-12 and min(p_i, p_d, p_j) >= 0.0):
            bad_sum += 1

    if bad_sum == 0:
        print("p_i + p_d + p_j == 1 for sampled fights, and all are ≥ 0 ✅")
    else:
        print(f"WARNING: {bad_sum} sampled rows had invalid probability sums")


def run_all(fights_csv: str, events_csv: str, head: int | None = None) -> None:
    """Run the whole diagnostics suite."""
    check_raw_csv(fights_csv)

    trainer = EloBTDTrainer()
    rows = trainer.load_data(fights_csv, events_csv)
    if head is not None:
        rows = rows[:head]

    rows = check_loader_maps_and_filter(trainer, fights_csv, events_csv)
    if head is not None:
        rows = rows[:head]

    df_raw_feats, X = check_feature_builder(rows)
    check_elapsed_time_logic(df_raw_feats)
    probability_sanity(rows, sample=50)

    # Optional: quick peek at feature matrix distribution
    _hdr("FEATURE MATRIX SNAPSHOT")
    print("X shape:", X.shape)
    colnames = ["(intercept)","finish","round_early_z","x_cards_tanhz","experience_z","layoff_z","newWC","r5","title","flash_ko"]
    if X.shape[1] == len(colnames):
        desc = pd.DataFrame(X, columns=colnames).describe().T[["mean","std","min","25%","50%","75%","max"]]
    else:
        desc = pd.DataFrame(X).describe().T
    print(desc.to_string(max_rows=20))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline diagnostics for UFC Elo-BTD model")
    parser.add_argument("--fights", required=True, help="Path to fights.csv")
    parser.add_argument("--events", required=True, help="Path to events.csv")
    parser.add_argument("--head", type=int, default=None, help="Use only first N after loading (for speed)")
    args = parser.parse_args()

    run_all(args.fights, args.events, head=args.head)