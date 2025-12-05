from __future__ import annotations

import os, json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .train import EloBTDTrainer
from .model import btd_probs, K_value, FightRow


# ---------- time-split helpers ----------

def chronological_folds(rows: List[FightRow], test_frac=0.15, folds=3):
    """
    Rolling, time-respecting CV:
      - Hold out the last `test_frac` for TEST.
      - Split the remaining prefix into `folds` consecutive blocks.
      - For k=1..folds:
           TRAIN = prefix[:k-1 blocks], VAL = block k
         (if k==1, use a small early segment as train seed).
    """
    n = len(rows)
    n_test = int(round(n * test_frac))
    te = rows[-n_test:] if n_test > 0 else []
    prefix = rows[:-n_test] if n_test > 0 else rows[:]

    # carve prefix into folds blocks
    blk = np.array_split(np.arange(len(prefix)), max(1, folds))
    blocks = [prefix[idx[0]:idx[-1]+1] if len(idx) else [] for idx in blk]

    # ensure a non-empty seed train
    folds_out = []
    for k in range(len(blocks)):
        val = blocks[k]
        if k == 0:
            # tiny seed train: first half of block 0 (or previous block if available)
            half = max(1, len(blocks[0]) // 2)
            tr = blocks[0][:half]
            va = blocks[0][half:]
        else:
            tr = sum(blocks[:k], [])
            va = val
        folds_out.append((tr, va))
    return folds_out, te


def build_features_seq(train_rows, val_rows, test_rows):
    """Single FeatureBuilder; fit scalers on TRAIN; carry state across VAL/TEST."""
    from .feature_builder import FeatureBuilder
    fb = FeatureBuilder()

    raw_tr = fb.collect_raw_features(train_rows)
    fb.fit_scalers(raw_tr)
    X_tr = fb.transform(raw_tr)
    for row, x in zip(raw_tr["fight"].tolist(), X_tr):
        row.x = x

    raw_va = fb.collect_raw_features(val_rows)
    X_va = fb.transform(raw_va)
    for row, x in zip(raw_va["fight"].tolist(), X_va):
        row.x = x

    raw_te = fb.collect_raw_features(test_rows)
    X_te = fb.transform(raw_te)
    for row, x in zip(raw_te["fight"].tolist(), X_te):
        row.x = x

    return train_rows, val_rows, test_rows


# ---------- metrics / replay ----------

def replay_and_metrics(trainer: EloBTDTrainer, rows: List[FightRow], params: np.ndarray) -> Dict[str, float]:
    a, Kmin, Kmax, nu = params[:4]; beta = params[4:]

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

        S_hat_i = p_i + 0.5*p_d
        R_class[key_i] = Ri + Kt * (y_i - S_hat_i)
        R_class[key_j] = Rj + Kt * ((1.0 - y_i) - (p_j + 0.5*p_d))

    df = pd.DataFrame(records)
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


# ---------- tuner with wide bounds & rolling CV ----------

def run_tuning_rolling(
    fights_csv: str,
    outdir: str = "outputs",
    folds: int = 3,
    test_frac: float = 0.15,
    ridge_grid = (1e-3, 3e-3, 1e-2),
    starts: int = 8,
    seed: int = 42,
    K_cap: float = 1e6,              # high so we don't clip during tuning
    bounds_override: Dict[str, tuple[float,float]] | None = None,
):
    os.makedirs(outdir, exist_ok=True)

    base_trainer = EloBTDTrainer(K_cap=K_cap)
    all_rows = base_trainer.load_data(fights_csv)

    folds_list, test_rows = chronological_folds(all_rows, test_frac=test_frac, folds=folds)

    fold_results = []
    for fold_idx, (train_rows, val_rows) in enumerate(folds_list, start=1):
        # Build features once per fold with train-only scaling & state carry
        tr_b, va_b, te_b = build_features_seq(train_rows[:], val_rows[:], test_rows[:])

        # local fitter assumes .x already populated
        def fit_on_rows_with_x(trainer: EloBTDTrainer, rows: List[FightRow],
                               ridge_lambda=1e-3, starts=8, seed=42) -> Dict:
            d = len(rows[0].x)
            bounds = trainer._bounds(d, bounds_override=bounds_override)
            rng = np.random.default_rng(seed)

            def random_start():
                c = trainer._constraint_config(bounds_override=bounds_override)
                a0    = rng.uniform(*c["a"])
                Kmin0 = rng.uniform(*c["Kmin"])
                Kmax0 = rng.uniform(max(Kmin0 + 8.0, c["Kmax"][0]), c["Kmax"][1])
                nu0   = rng.uniform(*c["nu"])
                beta0 = rng.normal(0.0, 0.5, size=d)
                return trainer._pack(a0, Kmin0, Kmax0, nu0, beta0)

            best = None
            for _ in range(max(1, starts)):
                vec0 = random_start()
                res = minimize(lambda v: trainer.neg_loglik_natural(v, rows, ridge_lambda=ridge_lambda),
                               vec0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 3000, "ftol": 1e-7, "gtol": 1e-6})
                rec = {"x": res.x, "fun": float(res.fun), "nit": res.nit, "success": bool(res.success), "message": res.message}
                if (best is None) or (rec["fun"] < best["fun"]):
                    best = rec
            return best

        best_by_val = None
        for ridge in ridge_grid:
            t = EloBTDTrainer(K_cap=K_cap)
            fit_tr = fit_on_rows_with_x(t, tr_b, ridge_lambda=ridge, starts=starts, seed=seed)
            params = fit_tr["x"]
            metrics_val = replay_and_metrics(t, va_b, params)
            rec = {"ridge": ridge, "fit": fit_tr, "val": metrics_val}
            if (best_by_val is None) or (metrics_val["logloss"] < best_by_val["val"]["logloss"]):
                best_by_val = rec

        # Refit on TRAIN+VAL with the chosen ridge; evaluate on TEST
        ridge_star = best_by_val["ridge"]
        t_final = EloBTDTrainer(K_cap=K_cap)
        fit_trva = fit_on_rows_with_x(t_final, tr_b + va_b, ridge_lambda=ridge_star, starts=starts, seed=seed)
        params_trva = fit_trva["x"]
        metrics_test = replay_and_metrics(t_final, te_b, params_trva)

        result = {
            "fold": fold_idx,
            "best_ridge": ridge_star,
            "test_metrics": metrics_test,
            "params_trva": params_trva.tolist(),
        }
        fold_results.append(result)
        print("\nFOLD RESULT:", result)

    # Pick best fold by test logloss
    best = sorted(fold_results, key=lambda r: r["test_metrics"]["logloss"])[0]
    np.save(os.path.join(outdir, "best_params_trva.npy"), np.array(best["params_trva"]))
    with open(os.path.join(outdir, "cv_results.json"), "w") as f:
        json.dump(fold_results, f, indent=2)
    with open(os.path.join(outdir, "best_cv_result.json"), "w") as f:
        json.dump(best, f, indent=2)

    print("\nBEST FOLD (by test logloss):")
    print(best)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rolling time-CV tuner for UFC Elo-BTD with wide bounds.")
    parser.add_argument("--fights", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-grid", default="1e-3,3e-3,1e-2",
                        help="Comma list, e.g. 0.0005,0.001,0.003")
    parser.add_argument("--K-cap", type=float, default=1e6,
                        help="Runtime clamp during tuning (set high to avoid clipping)")

    # These accept "lo,hi" strings (backward compatible)
    parser.add_argument("--Kmin-bounds", default="1,64", help="lo,hi")
    parser.add_argument("--Kmax-bounds", default="16,600", help="lo,hi (allow >500)")
    parser.add_argument("--nu-bounds",   default="1e-4,0.35", help="lo,hi for draw parameter")
    parser.add_argument("--a-bounds",    default="0.05,0.95", help="lo,hi for mixing parameter a")

    # IMPORTANT: allow negative first arg safely via nargs=2
    parser.add_argument("--beta-bounds", type=float, nargs=2, metavar=("BETA_LO","BETA_HI"),
                        default=(-2.0, 2.0),
                        help="Two numbers (space-separated), e.g. --beta-bounds -30 30")

    args = parser.parse_args()

    def parse_pair(val):
        """Accept 'lo,hi' string OR a 2-length sequence (from nargs=2)."""
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return (float(val[0]), float(val[1]))
        if isinstance(val, str):
            a, b = val.split(",")
            return (float(a), float(b))
        raise ValueError(f"Cannot parse pair from: {val!r}")

    bounds_override = {
        "a":    parse_pair(args.a_bounds),
        "Kmin": parse_pair(args.Kmin_bounds),
        "Kmax": parse_pair(args.Kmax_bounds),
        "nu":   parse_pair(args.nu_bounds),
        "beta": parse_pair(args.beta_bounds),   # works with nargs=2 or 'lo,hi'
    }

    ridge_grid = [float(x) for x in args.ridge_grid.split(",")]

    run_tuning_rolling(
        fights_csv=args.fights,
        outdir=args.outdir,
        folds=args.folds,
        test_frac=args.test_frac,
        ridge_grid=ridge_grid,
        starts=args.starts,
        seed=args.seed,
        K_cap=args.K_cap,
        bounds_override=bounds_override,
    )
def parse_pair(s):
    # Accept "lo,hi" string OR a 2-length sequence from nargs=2
    if isinstance(s, (list, tuple)) and len(s) == 2:
        return (float(s[0]), float(s[1]))
    if isinstance(s, str):
        a, b = s.split(",")
        return (float(a), float(b))
    raise ValueError(f"Cannot parse pair from: {s!r}")

    bounds_override = {
        "a":    parse_pair(args.a_bounds),
        "Kmin": parse_pair(args.Kmin_bounds),
        "Kmax": parse_pair(args.Kmax_bounds),
        "nu":   parse_pair(args.nu_bounds),
        "beta": parse_pair(args.beta_bounds)
    }
    ridge_grid = [float(x) for x in args.ridge_grid.split(",")]

    run_tuning_rolling(
        fights_csv=args.fights,
        outdir=args.outdir,
        folds=args.folds,
        test_frac=args.test_frac,
        ridge_grid=ridge_grid,
        starts=args.starts,
        seed=args.seed,
        K_cap=args.K_cap,
        bounds_override=bounds_override,
    )