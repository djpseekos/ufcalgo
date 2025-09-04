# src/elo_model/tuner.py
from __future__ import annotations

import math, json, os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .train import EloBTDTrainer
from .model import btd_probs, K_value, safe_log, FightRow

# ---------- metrics / replay ----------

def build_features_seq(train_rows, val_rows, test_rows):
    """
    Build time-dependent features sequentially with a single FeatureBuilder:
    - collect on TRAIN, fit scalers on TRAIN only, transform TRAIN
    - continue collecting on VAL (state carries over), transform VAL with train scalers
    - continue collecting on TEST, transform TEST with train scalers
    Returns (train_rows, val_rows, test_rows) with .x assigned.
    """
    from .feature_builder import FeatureBuilder
    fb = FeatureBuilder()

    # TRAIN
    raw_tr = fb.collect_raw_features(train_rows)
    fb.fit_scalers(raw_tr)                   # fit scalers on TRAIN only
    X_tr = fb.transform(raw_tr)
    for row, x in zip(raw_tr["fight"].tolist(), X_tr):
        row.x = x

    # VAL (state continues: fight_count/last_date/last_wc)
    raw_val = fb.collect_raw_features(val_rows)
    X_val = fb.transform(raw_val)            # uses TRAIN scalers
    for row, x in zip(raw_val["fight"].tolist(), X_val):
        row.x = x

    # TEST
    raw_te = fb.collect_raw_features(test_rows)
    X_te = fb.transform(raw_te)              # uses TRAIN scalers
    for row, x in zip(raw_te["fight"].tolist(), X_te):
        row.x = x

    return train_rows, val_rows, test_rows

def replay_and_metrics(trainer: EloBTDTrainer, rows: List[FightRow], params: np.ndarray) -> Dict[str, float]:
    """Replay fights with learned params and compute metrics."""
    a, Kmin, Kmax, nu = params[:4]; beta = params[4:]
    # States
    R_class: Dict[Tuple[str,str], float] = {}
    R_global: Dict[str, float] = {}
    class_means: Dict[str, float] = {}
    class_counts: Dict[str, int] = {}

    records = []
    for f in rows:
        key_i = (f.i, f.wc); key_j = (f.j, f.wc)
        Ri_g = R_global.get(f.i, 1500.0); Rj_g = R_global.get(f.j, 1500.0)

        if key_i not in R_class:
            mu_c = class_means.get(f.wc, 1500.0)
            R_class[key_i] = a * Ri_g + (1 - a) * mu_c
            class_means[f.wc] = (class_means.get(f.wc, 0.0) * class_counts.get(f.wc, 0) + R_class[key_i]) / (class_counts.get(f.wc, 0) + 1)
            class_counts[f.wc] = class_counts.get(f.wc, 0) + 1
        if key_j not in R_class:
            mu_c = class_means.get(f.wc, 1500.0)
            R_class[key_j] = a * Rj_g + (1 - a) * mu_c
            class_means[f.wc] = (class_means.get(f.wc, 0.0) * class_counts.get(f.wc, 0) + R_class[key_j]) / (class_counts.get(f.wc, 0) + 1)
            class_counts[f.wc] = class_counts.get(f.wc, 0) + 1

        Ri = R_class[key_i]; Rj = R_class[key_j]
        p_i, p_d, p_j = btd_probs(Ri, Rj, nu)

        if f.outcome == "i": y = np.array([1.,0.,0.]); y_i = 1.0
        elif f.outcome == "j": y = np.array([0.,0.,1.]); y_i = 0.0
        else: y = np.array([0.,1.,0.]); y_i = 0.5

        probs = np.array([p_i, p_d, p_j], dtype=float)
        eps = 1e-12
        logloss = -float(np.sum(y * np.log(np.maximum(probs, eps))))
        brier  = float(np.sum((y - probs)**2))

        Kt = min(K_value(Kmin, Kmax, beta, f.x), trainer.K_cap)
        records.append({"p_i":p_i,"p_d":p_d,"p_j":p_j,"logloss":logloss,"brier":brier,"K":Kt,"outcome":f.outcome})

        # update ratings
        S_hat_i = p_i + 0.5*p_d
        R_class[key_i] = Ri + Kt * (y_i - S_hat_i)
        R_class[key_j] = Rj + Kt * ((1.0 - y_i) - (p_j + 0.5*p_d))

    df = pd.DataFrame(records)
    # accuracy: pick argmax among (i,d,j)
    pred = np.argmax(df[["p_i","p_d","p_j"]].values, axis=1)
    true = df["outcome"].map({"i":0,"d":1,"j":2}).values
    acc = float(np.mean(pred == true))
    return {
        "logloss": float(df["logloss"].mean()),
        "brier": float(df["brier"].mean()),
        "accuracy": acc,
        "mean_p_d": float(df["p_d"].mean()),
        "emp_draw": float(np.mean(df["outcome"]=="d")),
        "K_median": float(df["K"].median()),
        "K_p90": float(np.percentile(df["K"], 90)),
        "K_p99": float(np.percentile(df["K"], 99)),
    }

# ---------- configurable trainer (override constraints) ----------

class ConfiguredTrainer(EloBTDTrainer):
    def __init__(self, K_cap: float = 128.0, constraint_config: Dict[str, Tuple[float,float]] | None = None):
        super().__init__(K_cap=K_cap)
        self._override_constraints = constraint_config

    def _constraint_config(self):
        if self._override_constraints is not None:
            return self._override_constraints
        return super()._constraint_config()

# ---------- utility: split ----------

def time_split(rows: List[FightRow], train_frac=0.7, val_frac=0.15):
    # rows are already sorted by date in load_data; keep order
    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = rows[:n_train]
    val   = rows[n_train:n_train+n_val]
    test  = rows[n_train+n_val:]
    return train, val, test

# ---------- fit on preloaded rows ----------

def fit_on_rows(trainer: EloBTDTrainer, rows: List[FightRow], ridge_lambda=1e-3, starts=8, seed=42) -> Dict:
    # build features on these rows only
    fights = trainer.build_features(rows)
    d = len(fights[0].x)

    bounds = trainer._bounds(d)
    rng = np.random.default_rng(seed)
    def random_start():
        c = trainer._constraint_config()
        a0    = rng.uniform(*c["a"])
        Kmin0 = rng.uniform(*c["Kmin"])
        Kmax0 = rng.uniform(max(Kmin0 + 8.0, c["Kmax"][0]), c["Kmax"][1])
        nu0   = rng.uniform(*c["nu"])
        beta0 = rng.normal(0.0, 0.2, size=d)
        return trainer._pack(a0, Kmin0, Kmax0, nu0, beta0)

    best = None
    for _ in range(max(1, starts)):
        vec0 = random_start()
        res = minimize(lambda v: trainer.neg_loglik_natural(v, fights, ridge_lambda=ridge_lambda),
                       vec0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-6, "gtol": 1e-6})
        rec = {"x": res.x, "fun": float(res.fun), "nit": res.nit, "success": bool(res.success), "message": res.message}
        if (best is None) or (rec["fun"] < best["fun"]):
            best = rec
    return best

# ---------- tuning runner ----------

def run_tuning(
    fights_csv: str,
    events_csv: str,
    outdir: str = "outputs",
    ridge_grid = (1e-3, 3e-3, 1e-2),
    K_bounds = ((4.0,16.0,28.0,72.0), (6.0,20.0,32.0,96.0), (8.0,24.0,40.0,128.0)),
    beta_bounds = (2.0, 2.5),
    starts: int = 8,
    seed: int = 42,
):
    """
    Time-split tuner with **sequential feature construction**:
      - Fit z-scales on TRAIN only, apply to VAL/TEST.
      - Carry time-dependent state (fight_count, last_date, last_wc) across TRAIN→VAL→TEST.
      - Select by VAL log-loss; refit on TRAIN+VAL; evaluate on TEST.
    """
    os.makedirs(outdir, exist_ok=True)

    # ---------- load & split chronologically ----------
    base_trainer = EloBTDTrainer()
    all_rows = base_trainer.load_data(fights_csv, events_csv)
    train_rows, val_rows, test_rows = time_split(all_rows, train_frac=0.7, val_frac=0.15)

    # ---------- helper: build features sequentially with train-only scalers ----------
    from .feature_builder import FeatureBuilder
    def build_features_seq(train_rows, val_rows, test_rows):
        fb = FeatureBuilder()

        # TRAIN: collect → fit scalers (TRAIN only) → transform
        raw_tr = fb.collect_raw_features(train_rows)
        fb.fit_scalers(raw_tr)
        X_tr = fb.transform(raw_tr)
        for row, x in zip(raw_tr["fight"].tolist(), X_tr):
            row.x = x

        # VAL: continue collection (state carries), transform with TRAIN scalers
        raw_va = fb.collect_raw_features(val_rows)
        X_va = fb.transform(raw_va)
        for row, x in zip(raw_va["fight"].tolist(), X_va):
            row.x = x

        # TEST: continue collection (state carries), transform with TRAIN scalers
        raw_te = fb.collect_raw_features(test_rows)
        X_te = fb.transform(raw_te)
        for row, x in zip(raw_te["fight"].tolist(), X_te):
            row.x = x

        return train_rows, val_rows, test_rows

    # Build features ONCE (independent of parameter constraints)
    tr_b, va_b, te_b = build_features_seq(train_rows[:], val_rows[:], test_rows[:])

    # ---------- local optimizer that assumes .x already set ----------
    def fit_on_rows_with_x(trainer: EloBTDTrainer, rows: List[FightRow],
                           ridge_lambda=1e-3, starts=8, seed=42) -> Dict:
        d = len(rows[0].x)
        bounds = trainer._bounds(d)
        rng = np.random.default_rng(seed)

        def random_start():
            c = trainer._constraint_config()
            a0    = rng.uniform(*c["a"])
            Kmin0 = rng.uniform(*c["Kmin"])
            Kmax0 = rng.uniform(max(Kmin0 + 8.0, c["Kmax"][0]), c["Kmax"][1])
            nu0   = rng.uniform(*c["nu"])
            beta0 = rng.normal(0.0, 0.2, size=d)
            return trainer._pack(a0, Kmin0, Kmax0, nu0, beta0)

        best = None
        for _ in range(max(1, starts)):
            vec0 = random_start()
            res = minimize(lambda v: trainer.neg_loglik_natural(v, rows, ridge_lambda=ridge_lambda),
                           vec0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000, "ftol": 1e-6, "gtol": 1e-6})
            rec = {"x": res.x, "fun": float(res.fun), "nit": res.nit, "success": bool(res.success), "message": res.message}
            if (best is None) or (rec["fun"] < best["fun"]):
                best = rec
        return best

    results = []

    # ---------- grid over constraint ranges ----------
    for (kmin_lo, kmin_hi, kmax_lo, kmax_hi) in K_bounds:
        for bmag in beta_bounds:
            constraint = {
                "a":    (0.05, 0.95),
                "Kmin": (kmin_lo, kmin_hi),
                "Kmax": (kmax_lo, kmax_hi),
                "nu":   (1e-4, 0.20),
                "beta": (-bmag, bmag),
            }
            for ridge in ridge_grid:
                t = ConfiguredTrainer(K_cap=500, constraint_config=constraint)

                # --- fit on TRAIN (rows already have .x from TRAIN-only scaling) ---
                fit_train = fit_on_rows_with_x(t, tr_b, ridge_lambda=ridge, starts=starts, seed=seed)

                # --- eval on VAL with the SAME features (no rebuilding) ---
                params_val = fit_train["x"]
                metrics_val = replay_and_metrics(t, va_b, params_val)

                row = {
                    "ridge": ridge,
                    "Kmin_lo": kmin_lo, "Kmin_hi": kmin_hi,
                    "Kmax_lo": kmax_lo, "Kmax_hi": kmax_hi,
                    "beta_bound": bmag,
                    "val_logloss": metrics_val["logloss"],
                    "val_brier": metrics_val["brier"],
                    "val_acc": metrics_val["accuracy"],
                    "val_mean_p_d": metrics_val["mean_p_d"],
                    "val_emp_draw": metrics_val["emp_draw"],
                    "val_K_med": metrics_val["K_median"],
                    "val_K_p90": metrics_val["K_p90"],
                    "val_K_p99": metrics_val["K_p99"],
                    "params": params_val.tolist(),
                    "success": fit_train["success"],
                }
                results.append(row)
                print("TUNE:", {k: row[k] for k in ["ridge","Kmin_lo","Kmin_hi","Kmax_lo","Kmax_hi","beta_bound","val_logloss","val_acc"]})

    # ---------- select best by validation log-loss ----------
    df = pd.DataFrame(results).sort_values("val_logloss")
    df.to_csv(os.path.join(outdir, "tuning_results.csv"), index=False)

    best = df.iloc[0].to_dict()
    best_params_val = np.array(best["params"], dtype=float)

    # ---------- final refit on TRAIN+VAL, then TEST (all already have .x) ----------
    best_constraint = {
        "a":    (0.05, 0.95),
        "Kmin": (best["Kmin_lo"], best["Kmin_hi"]),
        "Kmax": (best["Kmax_lo"], best["Kmax_hi"]),
        "nu":   (1e-4, 0.20),
        "beta": (-best["beta_bound"], best["beta_bound"]),
    }
    t_final = ConfiguredTrainer(K_cap=500, constraint_config=best_constraint)

    # Refit on TRAIN+VAL with the same feature representation
    fit_trval = fit_on_rows_with_x(t_final, tr_b + va_b, ridge_lambda=best["ridge"], starts=starts, seed=seed)
    params_trval = fit_trval["x"]

    metrics_test = replay_and_metrics(t_final, te_b, params_trval)

    # ---------- save artifacts ----------
    np.save(os.path.join(outdir, "best_params_trval.npy"), params_trval)
    with open(os.path.join(outdir, "best_config.json"), "w") as f:
        json.dump(best, f, indent=2)
    with open(os.path.join(outdir, "test_metrics.json"), "w") as f:
        json.dump(metrics_test, f, indent=2)

    print("\nBEST (by val logloss):")
    print(best)
    print("\nTEST METRICS:")
    print(metrics_test)

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Time-split tuning for UFC Elo-BTD.")
    parser.add_argument("--fights", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_tuning(args.fights, args.events, outdir=args.outdir, starts=args.starts, seed=args.seed)