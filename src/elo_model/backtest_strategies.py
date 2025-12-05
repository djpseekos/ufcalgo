# src/elo_model/backtest_strategies.py
from __future__ import annotations

import argparse
import math
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .train import EloBTDTrainer
from .model import FightRow, btd_probs, K_value
from .feature_builder import FeatureBuilder


# ----------------- odds / profit helpers -----------------

def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def american_implied_prob(odds: float) -> Optional[float]:
    """American odds -> implied probability (with vig kept)."""
    o = _to_float(odds)
    if o is None:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return None

def profit_from_odds(odds: float, won: Optional[bool], stake: float = 1.0) -> Optional[float]:
    """
    Return net profit for a single $stake bet at American odds.
    won=True  -> win payout;  won=False -> lose stake;  won=None -> push (0.0).
    """
    o = _to_float(odds)
    if o is None:
        return None
    if won is None:
        return 0.0
    if won:
        return stake * (o / 100.0) if o > 0 else stake * (100.0 / (-o))
    else:
        return -stake

def pick_odds(row: pd.Series, side: str) -> Optional[float]:
    """
    Choose the 'best available close' line first, else fall back to open.
    Robust to several column name variants.
    """
    side = side.upper()

    def first(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in row and pd.notna(row[k]) and str(row[k]).strip() != "":
                try:
                    return float(row[k])
                except Exception:
                    pass
        return None

    if side == "R":
        return first([
            "r_bfo_close_b", "r_close_b", "r_bfo_closeB",
            "r_bfo_close_a", "r_close_a", "r_bfo_closeA",
            "r_bfo_open", "r_open", "bfo_open_r"
        ])
    if side == "B":
        return first([
            "b_bfo_close_b", "b_close_b", "b_bfo_closeB",
            "b_bfo_close_a", "b_close_a", "b_bfo_closeA",
            "b_bfo_open", "b_open", "bfo_open_b"
        ])
    return None


# ----------------- time-respecting split + features -----------------

def chronological_folds(rows: List[FightRow], test_frac=0.15, folds=3):
    """
    Rolling, time-respecting CV:
      - TEST = last test_frac of fights.
      - Prefix is split into `folds` consecutive blocks.
      - For k=1..folds: TRAIN = blocks[:k-1], VAL = block k
        (k=1 uses first half of block 1 as a seed TRAIN).
    """
    n = len(rows)
    n_test = int(round(n * test_frac))
    te = rows[-n_test:] if n_test > 0 else []
    prefix = rows[:-n_test] if n_test > 0 else rows[:]

    idx_blocks = np.array_split(np.arange(len(prefix)), max(1, folds))
    blocks = [prefix[idx[0]:idx[-1]+1] if len(idx) else [] for idx in idx_blocks]

    folds_out = []
    for k in range(len(blocks)):
        val = blocks[k]
        if k == 0:
            half = max(1, len(blocks[0]) // 2)
            tr = blocks[0][:half]
            va = blocks[0][half:]
        else:
            tr = sum(blocks[:k], [])
            va = val
        folds_out.append((tr, va))
    return folds_out, te

def build_features_seq(train_rows, val_rows, test_rows):
    """
    One FeatureBuilder per fold:
      - fit scalers on TRAIN only,
      - carry time-dependent state across VAL and TEST,
      - transform all three.
    """
    fb = FeatureBuilder()

    raw_tr = fb.collect_raw_features(train_rows)
    fb.fit_scalers(raw_tr)
    X_tr = fb.transform(raw_tr)
    for r, x in zip(raw_tr["fight"].tolist(), X_tr):
        r.x = x

    raw_va = fb.collect_raw_features(val_rows)
    X_va = fb.transform(raw_va)
    for r, x in zip(raw_va["fight"].tolist(), X_va):
        r.x = x

    raw_te = fb.collect_raw_features(test_rows)
    X_te = fb.transform(raw_te)
    for r, x in zip(raw_te["fight"].tolist(), X_te):
        r.x = x

    return train_rows, val_rows, test_rows


# ----------------- pre-fight probabilities from tuned params -----------------

def pre_fight_probs(rows: List[FightRow], params: np.ndarray, K_cap: float) -> Dict[str, Tuple[float,float,float]]:
    """
    Replay the full (TRAIN+VAL+TEST) chronology, returning pre-fight
    probs for each fight keyed by (red_id, blue_id, date_ts).
    """
    a, Kmin, Kmax, nu = params[:4]
    beta = params[4:]

    def key_for(f: FightRow) -> str:
        return f"{f.i}-{f.j}-{int(pd.Timestamp(f.date).value)}"

    R_class: Dict[Tuple[str, str], float] = {}
    R_global: Dict[str, float] = {}
    class_means: Dict[str, float] = {}
    class_counts: Dict[str, int] = {}

    probs: Dict[str, Tuple[float, float, float]] = {}

    for f in rows:
        key_i = (f.i, f.wc); key_j = (f.j, f.wc)
        Ri_g = R_global.get(f.i, 1500.0); Rj_g = R_global.get(f.j, 1500.0)

        if key_i not in R_class:
            mu_c = class_means.get(f.wc, 1500.0)
            R_class[key_i] = a * Ri_g + (1.0 - a) * mu_c
            class_means[f.wc] = (class_means.get(f.wc, 0.0) * class_counts.get(f.wc, 0) + R_class[key_i]) / (class_counts.get(f.wc, 0) + 1)
            class_counts[f.wc] = class_counts.get(f.wc, 0) + 1
        if key_j not in R_class:
            mu_c = class_means.get(f.wc, 1500.0)
            R_class[key_j] = a * Rj_g + (1.0 - a) * mu_c
            class_means[f.wc] = (class_means.get(f.wc, 0.0) * class_counts.get(f.wc, 0) + R_class[key_j]) / (class_counts.get(f.wc, 0) + 1)
            class_counts[f.wc] = class_counts.get(f.wc, 0) + 1

        Ri = R_class[key_i]; Rj = R_class[key_j]
        p_i, p_d, p_j = btd_probs(Ri, Rj, nu)

        # store pre-update probs
        probs[key_for(f)] = (p_i, p_d, p_j)

        # update with observed outcome
        if f.outcome == "i": y_i = 1.0
        elif f.outcome == "j": y_i = 0.0
        else: y_i = 0.5
        S_hat_i = p_i + 0.5 * p_d
        Kt = min(K_value(Kmin, Kmax, beta, f.x), K_cap)
        R_class[key_i] = Ri + Kt * (y_i - S_hat_i)
        R_class[key_j] = Rj + Kt * ((1.0 - y_i) - (p_j + 0.5 * p_d))

    return probs


# ----------------- strategy runner -----------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare strategies on second-fold TEST: value(fav-only) vs always-favorite vs always-underdog vs random."
    )
    ap.add_argument("--fights", default="data/curated/fights.csv")
    ap.add_argument("--params", default="outputs/best_params_trva.npy",
                    help="Tuned params from tuner (a, Kmin, Kmax, nu, beta...)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--K-cap", type=float, default=1000.0)

    # Random baseline controls
    ap.add_argument("--random-trials", type=int, default=1000, help="Number of random simulations")
    ap.add_argument("--random-seed", type=int, default=None, help="Optional seed (omit for non-deterministic)")

    args = ap.parse_args()

    # Load fights and create chronological FightRows
    trainer = EloBTDTrainer(K_cap=args.K_cap)
    rows = trainer.load_data(args.fights)  # your version uses fights.csv only and sorts by date
    if not rows:
        raise SystemExit("No fights loaded from fights.csv")

    # Build folds; use “second fold” TRAIN/VAL just to honor your earlier convention
    folds_list, test_rows = chronological_folds(rows, test_frac=args.test_frac, folds=args.folds)
    fold_index = 1 if len(folds_list) >= 2 else 0
    train_rows, val_rows = folds_list[fold_index]

    # Features with TRAIN-only scalers + state carry
    tr_b, va_b, te_b = build_features_seq(train_rows[:], val_rows[:], test_rows[:])

    # Read tuned params and compute pre-fight probabilities for all fights we just built
    params = np.load(args.params)
    all_rows_in_order = tr_b + va_b + te_b
    probs_map = pre_fight_probs(all_rows_in_order, params, K_cap=args.K_cap)

    # Build a test DataFrame that exactly matches te_b by (red_id, blue_id, date)
    fights_df = pd.read_csv(args.fights)
    fights_df["date"] = pd.to_datetime(fights_df["date"])
    fights_df = fights_df.sort_values(["date", "event_id"]).reset_index(drop=True)

    def fkey_series(r: pd.Series) -> str:
        return f"{str(r['r_fighter_id'])}-{str(r['b_fighter_id'])}-{int(pd.Timestamp(r['date']).value)}"

    fights_df["_key"] = fights_df.apply(fkey_series, axis=1)

    test_keys = set(f"{f.i}-{f.j}-{int(pd.Timestamp(f.date).value)}" for f in te_b)
    fights_test = fights_df.loc[fights_df["_key"].isin(test_keys)].copy()

    # Strategy stats accumulators
    stats = {
        "value_fav": {"bets": 0, "staked": 0.0, "profit": 0.0, "wins": 0},
        "favorite":  {"bets": 0, "staked": 0.0, "profit": 0.0, "wins": 0},
        "underdog":  {"bets": 0, "staked": 0.0, "profit": 0.0, "wins": 0},
    }

    # Random baseline setup
    rng = random.Random()
    if args.random_seed is None:
        try:
            rng.seed(int.from_bytes(os.urandom(8), "little"))
        except Exception:
            rng.seed(None)
    else:
        rng.seed(args.random_seed)

    # Collect per-trial random ROI
    random_rois: List[float] = []
    random_profits: List[float] = []
    random_bets: List[int] = []

    # Helper: outcome to won/lose/push
    def outcome_to_won(side: str, winner_corner: str) -> Optional[bool]:
        wc = str(winner_corner).strip()
        if wc in ("D", "NC"):
            return None
        return (wc == "R") if side == "R" else (wc == "B")

    # Loop once to compute the three deterministic strategies
    for _, r in fights_test.iterrows():
        key = r["_key"]
        if key not in probs_map:
            continue
        p_red, p_draw, p_blue = probs_map[key]

        # market odds/implied
        o_r = pick_odds(r, "R");  o_b = pick_odds(r, "B")
        imp_r = american_implied_prob(o_r) if o_r is not None else None
        imp_b = american_implied_prob(o_b) if o_b is not None else None
        if imp_r is None and imp_b is None:
            continue

        # ---------- Value (fav-only): bet only if model improves the market favorite ----------
        fav_side = None
        if (imp_r is not None) and (imp_b is not None):
            fav_side = "R" if imp_r >= imp_b else "B"
        elif imp_r is not None:
            fav_side = "R"
        elif imp_b is not None:
            fav_side = "B"

        if fav_side == "R" and o_r is not None and imp_r is not None and p_red is not None:
            if p_red > imp_r:  # only if model thinks favorite is even stronger
                won = outcome_to_won("R", r.get("winner_corner", ""))
                prof = profit_from_odds(o_r, won, stake=args.stake)
                if prof is not None:
                    s = stats["value_fav"]
                    s["bets"] += 1; s["staked"] += args.stake; s["profit"] += prof
                    if won: s["wins"] += 1
        elif fav_side == "B" and o_b is not None and imp_b is not None and p_blue is not None:
            if p_blue > imp_b:
                won = outcome_to_won("B", r.get("winner_corner", ""))
                prof = profit_from_odds(o_b, won, stake=args.stake)
                if prof is not None:
                    s = stats["value_fav"]
                    s["bets"] += 1; s["staked"] += args.stake; s["profit"] += prof
                    if won: s["wins"] += 1

        # ---------- Always Favorite ----------
        if fav_side == "R" and o_r is not None:
            won = outcome_to_won("R", r.get("winner_corner", ""))
            prof = profit_from_odds(o_r, won, stake=args.stake)
            if prof is not None:
                s = stats["favorite"]
                s["bets"] += 1; s["staked"] += args.stake; s["profit"] += prof
                if won: s["wins"] += 1
        elif fav_side == "B" and o_b is not None:
            won = outcome_to_won("B", r.get("winner_corner", ""))
            prof = profit_from_odds(o_b, won, stake=args.stake)
            if prof is not None:
                s = stats["favorite"]
                s["bets"] += 1; s["staked"] += args.stake; s["profit"] += prof
                if won: s["wins"] += 1

        # ---------- Always Underdog ----------
        dog_side = None
        if (imp_r is not None) and (imp_b is not None):
            dog_side = "R" if imp_r < imp_b else "B"
        elif imp_r is not None:
            dog_side = "R"
        elif imp_b is not None:
            dog_side = "B"

        if dog_side == "R" and o_r is not None:
            won = outcome_to_won("R", r.get("winner_corner", ""))
            prof = profit_from_odds(o_r, won, stake=args.stake)
            if prof is not None:
                s = stats["underdog"]
                s["bets"] += 1; s["staked"] += args.stake; s["profit"] += prof
                if won: s["wins"] += 1
        elif dog_side == "B" and o_b is not None:
            won = outcome_to_won("B", r.get("winner_corner", ""))
            prof = profit_from_odds(o_b, won, stake=args.stake)
            if prof is not None:
                s = stats["underdog"]
                s["bets"] += 1; s["staked"] += args.stake; s["profit"] += prof
                if won: s["wins"] += 1

    # ---------- Random baseline (simulate per trial, reusing the same TEST set) ----------
    # Non-deterministic by default; set --random-seed for reproducible results
    for _ in range(args.random_trials):
        total_profit = 0.0
        total_bets = 0
        for _, r in fights_test.iterrows():
            o_r = pick_odds(r, "R");  o_b = pick_odds(r, "B")
            avail = []
            if o_r is not None: avail.append(("R", o_r))
            if o_b is not None: avail.append(("B", o_b))
            if not avail:
                continue
            side, odds = rng.choice(avail)
            won = outcome_to_won(side, r.get("winner_corner", ""))
            prof = profit_from_odds(odds, won, stake=args.stake)
            if prof is not None:
                total_profit += prof
                total_bets += 1
        staked = total_bets * args.stake
        random_bets.append(total_bets)
        random_profits.append(total_profit)
        random_rois.append((total_profit / staked) if staked > 0 else float("nan"))

    # ---------- print compact comparison table ----------
    def fmt_line(name: str, s: Dict[str, float]) -> str:
        bets = s["bets"]; staked = s["staked"]; profit = s["profit"]; wins = s["wins"]
        roi = (profit / staked * 100.0) if staked > 0 else float("nan")
        hit = (wins / bets * 100.0) if bets > 0 else float("nan")
        return f"{name:<19} | {bets:5d} | ${staked:8.2f} | ${profit:9.2f} | {roi:6.2f}% | {hit:6.2f}%"

    def pct(vals: List[float], q: float) -> float:
        arr = sorted([v for v in vals if not (v is None or (isinstance(v, float) and math.isnan(v)))])
        if not arr: return float("nan")
        k = (len(arr) - 1) * q
        f = math.floor(k); c = math.ceil(k)
        if f == c: return arr[int(k)]
        return arr[f] + (k - f) * (arr[c] - arr[f])

    print("\n=== Test-period strategy comparison (second fold) ===")
    print("Strategy            | Bets  |   Staked   |    Profit   |   ROI   |  Hit% ")
    print("--------------------+-------+------------+-------------+---------+--------")
    print(fmt_line("Value (fav-only)", stats["value_fav"]))
    print(fmt_line("Always Favorite",  stats["favorite"]))
    print(fmt_line("Always Underdog",  stats["underdog"]))

    print(f"\nRandom baseline ({args.random_trials} trials; "
          f"{'seeded' if args.random_seed is not None else 'non-deterministic'})")
    if random_bets:
        print("  bets mean   : {:,.1f}".format(float(np.mean(random_bets))))
        print("  profit mean : ${:,.2f}".format(float(np.mean(random_profits))))
        roi_mean = float(np.mean([r for r in random_rois if not math.isnan(r)])) if random_rois else float("nan")
        print("  ROI mean    : {:6.2f}%".format(roi_mean * 100 if not math.isnan(roi_mean) else float('nan')))
        print("  ROI p05/50/95: {:6.2f}% / {:6.2f}% / {:6.2f}%".format(
            pct(random_rois, 0.05) * 100 if not math.isnan(pct(random_rois, 0.05)) else float('nan'),
            pct(random_rois, 0.50) * 100 if not math.isnan(pct(random_rois, 0.50)) else float('nan'),
            pct(random_rois, 0.95) * 100 if not math.isnan(pct(random_rois, 0.95)) else float('nan'),
        ))
    else:
        print("  (no eligible fights for random baseline)")

if __name__ == "__main__":
    main()