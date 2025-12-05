from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .train import EloBTDTrainer
from .model import btd_probs, K_value, FightRow


def build_features_all(rows: List[FightRow]) -> List[FightRow]:
    """
    Build features on the full chronology (fit scalers on full data, carry state).
    This is fine for historical export since params are already fixed.
    """
    from .feature_builder import FeatureBuilder
    fb = FeatureBuilder()
    raw = fb.collect_raw_features(rows)
    fb.fit_scalers(raw)            # fit on ALL rows for export
    X = fb.transform(raw)
    for r, x in zip(raw["fight"].tolist(), X):
        r.x = x
    return raw["fight"].tolist()


def export_elo_history(
    fights_csv: str,
    events_csv: str,
    fighters_csv: str,
    params_npy: str,
    out_csv: str,
    K_cap: float = 1e6,
) -> None:
    # --- load timeline & build features ---
    trainer = EloBTDTrainer(K_cap=K_cap)
    rows = trainer.load_data(fights_csv)       # already sorted by date in load_data()
    if not len(rows):
        raise SystemExit("No fights loaded from fights CSV.")
    rows = build_features_all(rows)

    # --- load canonical dates & fighters ---
    events = pd.read_csv(events_csv, parse_dates=["date"])
    event_dates = sorted(pd.to_datetime(events["date"]).dt.date.unique())
    fighters = pd.read_csv(fighters_csv, dtype=str)
    fighter_ids = fighters["fighter_id"].astype(str).tolist()

    # --- load tuned params ---
    params = np.load(params_npy)
    a, Kmin, Kmax, nu = params[:4]
    beta = params[4:]

    # --- state (mirrors training) ---
    R_class: Dict[Tuple[str, str], float] = {}     # (fighter_id, weight_class) -> rating
    R_global: Dict[str, float] = {}                # (unused in updates; kept for symmetry)
    class_means: Dict[str, float] = {}
    class_counts: Dict[str, int] = {}
    last_wc: Dict[str, str] = {}                  # last observed WC per fighter

    # --- group fights by date for fast replay ---
    fights_by_date: Dict[object, List[FightRow]] = defaultdict(list)
    for f in rows:
        fights_by_date[pd.to_datetime(f.date).date()].append(f)

    # --- helper: get a single Elo for a fighter (current WC if known, else 1500) ---
    def current_elo(fid: str) -> float:
        wc = last_wc.get(fid)
        if wc is not None and (fid, wc) in R_class:
            return R_class[(fid, wc)]
        return 1500.0

    # --- pre-allocate wide table (we’ll fill column by column) ---
    out = pd.DataFrame({"fighter_id": fighter_ids})

    # --- iterate event dates chronologically ---
    for d in event_dates:
        # 1) process all fights on this date
        for f in fights_by_date.get(d, []):
            key_i = (f.i, f.wc)
            key_j = (f.j, f.wc)

            # initialize class ratings on first appearance in this WC
            if key_i not in R_class:
                mu_c = class_means.get(f.wc, 1500.0)
                Ri_g = R_global.get(f.i, 1500.0)
                R_class[key_i] = a * Ri_g + (1.0 - a) * mu_c
                class_means[f.wc] = (class_means.get(f.wc, 0.0) * class_counts.get(f.wc, 0) + R_class[key_i]) / (
                    class_counts.get(f.wc, 0) + 1
                )
                class_counts[f.wc] = class_counts.get(f.wc, 0) + 1

            if key_j not in R_class:
                mu_c = class_means.get(f.wc, 1500.0)
                Rj_g = R_global.get(f.j, 1500.0)
                R_class[key_j] = a * Rj_g + (1.0 - a) * mu_c
                class_means[f.wc] = (class_means.get(f.wc, 0.0) * class_counts.get(f.wc, 0) + R_class[key_j]) / (
                    class_counts.get(f.wc, 0) + 1
                )
                class_counts[f.wc] = class_counts.get(f.wc, 0) + 1

            # outcomes & probs
            Ri = R_class[key_i]; Rj = R_class[key_j]
            p_i, p_d, p_j = btd_probs(Ri, Rj, nu)

            if f.outcome == "i":
                S_i = 1.0
            elif f.outcome == "j":
                S_i = 0.0
            else:
                S_i = 0.5

            S_hat_i = p_i + 0.5 * p_d
            Kt = min(K_value(Kmin, Kmax, beta, f.x), trainer.K_cap)

            # apply updates
            R_class[key_i] = Ri + Kt * (S_i - S_hat_i)
            R_class[key_j] = Rj + Kt * ((1.0 - S_i) - (p_j + 0.5 * p_d))

            # track last WC
            last_wc[f.i] = f.wc
            last_wc[f.j] = f.wc

        # 2) snapshot after the date’s fights
        col_name = str(d)  # YYYY-MM-DD
        out[col_name] = [current_elo(fid) for fid in fighter_ids]

    # forward-fill across dates: already snapshots every date post-event, but
    # if you later add non-event dates, you could ffill here. Kept as-is.

    # write CSV
    out.to_csv(out_csv, index=False)
    print(f"Wrote Elo history → {out_csv}  (shape={out.shape})")


def main():
    ap = argparse.ArgumentParser(description="Export wide Elo history matrix (fighters × event dates).")
    ap.add_argument("--fights", default="data/curated/fights.csv")
    ap.add_argument("--events", default="data/curated/events.csv")
    ap.add_argument("--fighters", default="data/curated/fighters.csv")
    ap.add_argument("--params", default="outputs/best_params_trva.npy",
                    help="Tuned parameters npy (a, Kmin, Kmax, nu, beta...)")
    ap.add_argument("--out", default="data/curated/elo_history.csv")
    ap.add_argument("--K-cap", type=float, default=1e6, help="Safety clamp during export")
    args = ap.parse_args()

    export_elo_history(
        fights_csv=args.fights,
        events_csv=args.events,
        fighters_csv=args.fighters,
        params_npy=args.params,
        out_csv=args.out,
        K_cap=args.K_cap,
    )


if __name__ == "__main__":
    main()