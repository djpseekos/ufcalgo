# src/elo_model/train.py
from __future__ import annotations

import math
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model import FightRow, btd_probs, K_value, safe_log
from .feature_builder import FeatureBuilder


class EloBTDTrainer:
    def __init__(self, K_cap: float = 500):
        self.K_cap = K_cap

    # ------------------------------
    # Data loading / preprocessing
    # ------------------------------
    def load_data(self, fights_csv: str, events_csv: str) -> List[FightRow]:
        fights = pd.read_csv(fights_csv)
        events = pd.read_csv(events_csv, parse_dates=["date"])
        df = fights.merge(events[["event_id", "date"]], on="event_id", how="left")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "event_id"]).reset_index(drop=True)

        rows: List[FightRow] = []

        for _, r in df.iterrows():
            # outcome from winner_corner; skip NC/missing
            winner = (str(r["winner_corner"]).strip() if pd.notna(r["winner_corner"]) else "")
            if winner in ("", "NC"):
                continue
            if winner == "R":
                outcome = "i"
            elif winner == "B":
                outcome = "j"
            elif winner == "D":
                outcome = "d"
            else:
                continue

            # safe ints with defaults
            def safe_int(x, default):
                return int(x) if pd.notna(x) else int(default)

            # scheduled rounds: default to 3 if missing or weird
            sr_raw = r.get("scheduled_rounds", 3)
            sr = safe_int(sr_raw, 3)
            if sr not in (3, 5):
                sr = 3

            end_round = safe_int(r.get("end_round", 0), 0)
            end_time = safe_int(r.get("end_time_sec", 0), 0)
            is_title = safe_int(r.get("is_title", 0), 0)
            judge_scores = str(r.get("judge_scores", "")) if pd.notna(r.get("judge_scores", "")) else ""

            rows.append(
                FightRow(
                    i=str(r["r_fighter_id"]),
                    j=str(r["b_fighter_id"]),
                    outcome=outcome,
                    wc=str(r["weight_class"]),
                    date=r["date"],
                    scheduled_rounds=sr,
                    method=str(r.get("method", "")) if pd.notna(r.get("method", "")) else "",
                    end_round=end_round,
                    end_time_sec=end_time,
                    is_title=is_title,
                    judge_scores=judge_scores,
                )
            )

        return rows

    def build_features(self, fights: List[FightRow]) -> List[FightRow]:
        fb = FeatureBuilder()
        raw = fb.collect_raw_features(fights)
        fb.fit_scalers(raw)
        X = fb.transform(raw)
        for row, x in zip(raw["fight"].tolist(), X):
            row.x = x
        return raw["fight"].tolist()

    # ------------------------------
    # Parameter packing helpers
    # ------------------------------
    def _pack(self, a: float, Kmin: float, Kmax: float, nu: float, beta: np.ndarray) -> np.ndarray:
        return np.concatenate(([a, Kmin, Kmax, nu], beta))

    def _unpack(self, vec: np.ndarray) -> Tuple[float, float, float, float, np.ndarray]:
        a, Kmin, Kmax, nu = vec[:4]
        beta = vec[4:]
        return a, Kmin, Kmax, nu, beta

    # ------------------------------
    # Objective in natural parameters
    # ------------------------------
    def neg_loglik_natural(self, vec: np.ndarray, fights: List[FightRow], ridge_lambda: float = 0.0) -> float:
        a, Kmin, Kmax, nu, beta = self._unpack(vec)
        a, Kmin, Kmax, nu, beta = self._clamp(a, Kmin, Kmax, nu, beta)

        # Rating states
        R_class: Dict[Tuple[str, str], float] = {}
        R_global: Dict[str, float] = {}  # only for class-init prior
        class_means: Dict[str, float] = {}
        class_counts: Dict[str, int] = {}

        ll = 0.0

        for f in fights:
            key_i = (f.i, f.wc)
            key_j = (f.j, f.wc)

            Ri_g = R_global.get(f.i, 1500.0)
            Rj_g = R_global.get(f.j, 1500.0)

            # Initialize class ratings on first appearance in this WC
            if key_i not in R_class:
                mu_c = class_means.get(f.wc, 1500.0)
                R_class[key_i] = a * Ri_g + (1.0 - a) * mu_c
                class_means[f.wc] = (class_means.get(f.wc, 0.0) * class_counts.get(f.wc, 0) + R_class[key_i]) / (
                    class_counts.get(f.wc, 0) + 1
                )
                class_counts[f.wc] = class_counts.get(f.wc, 0) + 1

            if key_j not in R_class:
                mu_c = class_means.get(f.wc, 1500.0)
                R_class[key_j] = a * Rj_g + (1.0 - a) * mu_c
                class_means[f.wc] = (class_means.get(f.wc, 0.0) * class_counts.get(f.wc, 0) + R_class[key_j]) / (
                    class_counts.get(f.wc, 0) + 1
                )
                class_counts[f.wc] = class_counts.get(f.wc, 0) + 1

            Ri = R_class[key_i]
            Rj = R_class[key_j]

            # Outcome probabilities
            p_i, p_d, p_j = btd_probs(Ri, Rj, nu)

            # Log-likelihood and observed score for i
            if f.outcome == "i":
                ll += safe_log(p_i)
                S_i = 1.0
            elif f.outcome == "j":
                ll += safe_log(p_j)
                S_i = 0.0
            else:
                ll += safe_log(p_d)
                S_i = 0.5

            # Predicted score and K
            S_hat_i = p_i + 0.5 * p_d
            Kt = min(K_value(Kmin, Kmax, beta, f.x), self.K_cap)

            # Updates (class Elo only)
            R_class[key_i] = Ri + Kt * (S_i - S_hat_i)
            R_class[key_j] = Rj + Kt * ((1.0 - S_i) - (p_j + 0.5 * p_d))

            # (Optional) update R_global slowly if you want; kept fixed here

        # Ridge on beta only
        if ridge_lambda > 0.0:
            ll -= ridge_lambda * float(np.sum(beta * beta))

        return -ll  # minimize

    # ------------------------------
    # Fit with multi-start L-BFGS-B
    # ------------------------------
    def fit(
        self,
        fights_csv: str,
        events_csv: str,
        ridge_lambda: float = 1e-3,
        head: int | None = None,
        starts: int = 8,
        seed: int = 42,
    ):
        fights = self.load_data(fights_csv, events_csv)
        if head is not None:
            fights = fights[:head]
        fights = self.build_features(fights)

        d = len(fights[0].x)  # feature dimension

        # central bounds (match _constraint_config)
        bounds = self._bounds(d)

        rng = np.random.default_rng(seed)

        def random_start() -> np.ndarray:
            c = self._constraint_config()
            a0    = rng.uniform(*c["a"])
            Kmin0 = rng.uniform(*c["Kmin"])
            Kmax0 = rng.uniform(max(Kmin0 + 8.0, c["Kmax"][0]), c["Kmax"][1])  # ensure separation
            nu0   = rng.uniform(*c["nu"])
            beta0 = rng.normal(0.0, 0.2, size=d)
            return self._pack(a0, Kmin0, Kmax0, nu0, beta0)

        best = None
        for _ in range(max(1, starts)):
            vec0 = random_start()
            res = minimize(
                lambda v: self.neg_loglik_natural(v, fights, ridge_lambda=ridge_lambda),
                vec0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-6, "gtol": 1e-6},
            )
            rec = {
                "x": res.x, "fun": float(res.fun), "nit": res.nit,
                "success": bool(res.success), "message": res.message
            }
            if (best is None) or (rec["fun"] < best["fun"]):
                best = rec

        return best

    # --- central constraint config ---
    def _constraint_config(self):
        """All parameter constraints in one place."""
        return {
            "a":      (0.05, 0.95),
            "Kmin":   (4.0, 16.0),
            "Kmax":   (40.0, 500),
            "nu":     (1e-4, 0.20),
            "beta":   (-2.0, 2.0),
        }

    def _bounds(self, d: int):
        c = self._constraint_config()
        return [
            c["a"],          # a
            c["Kmin"],       # Kmin
            c["Kmax"],       # Kmax
            c["nu"],         # nu
            *([c["beta"]] * d)  # betas
        ]

    def _clamp(self, a, Kmin, Kmax, nu, beta):
        """Numerical safety clamp that mirrors _bounds()."""
        c = self._constraint_config()
        lo, hi = c["a"];    a = max(lo, min(a, hi))
        lo, hi = c["Kmin"]; Kmin = max(lo, min(Kmin, hi))
        lo, hi = c["Kmax"]; Kmax = max(lo, min(Kmax, hi))
        if Kmax <= Kmin:    Kmax = Kmin + 1.0
        lo, hi = c["nu"];   nu = max(lo, min(nu, hi))
        b_lo, b_hi = c["beta"]
        beta = np.clip(beta, b_lo, b_hi)
        return a, Kmin, Kmax, nu, beta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fights", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--head", type=int, default=None, help="Use only first N fights")
    parser.add_argument("--starts", type=int, default=8, help="Number of random restarts")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for restarts")
    args = parser.parse_args()

    trainer = EloBTDTrainer(K_cap=500)
    out = trainer.fit(
        args.fights,
        args.events,
        ridge_lambda=args.ridge,
        head=args.head,
        starts=args.starts,
        seed=args.seed,
    )
    print(out)