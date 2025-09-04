# src/elo_model/feature_builder.py
import math
import numpy as np
import pandas as pd
from typing import Dict, List
from .helpers import is_decision, is_ko_tko, parse_judge_margin
from .time_utils import actual_elapsed_seconds

class FeatureBuilder:
    def __init__(self, epsilon_sec: int = 30, flash_ko_sec: int = 30):
        self.epsilon_sec = int(epsilon_sec)
        self.flash_ko_sec = int(flash_ko_sec)
        self.last_date: Dict[str, pd.Timestamp] = {}
        self.fight_count: Dict[str, int] = {}
        self.last_wc: Dict[str, str] = {}
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def collect_raw_features(self, fights: List) -> pd.DataFrame:
        rows = []
        for f in fights:
            ni = self.fight_count.get(f.i, 0); nj = self.fight_count.get(f.j, 0)
            experience = 1.0 / math.sqrt(1.0 + ni + nj)
            di = self.last_date.get(f.i); dj = self.last_date.get(f.j)
            vals = []
            if di is not None: vals.append((f.date - di).days)
            if dj is not None: vals.append((f.date - dj).days)
            layoff_days = float(np.median(vals)) if vals else 365.0
            layoff = math.log1p(layoff_days / 180.0)
            newWC = int(self.last_wc.get(f.i) != f.wc or self.last_wc.get(f.j) != f.wc)
            r5 = int(f.scheduled_rounds == 5); title = int(f.is_title)

            method_clean = (f.method or "").strip()
            finish = int(not is_decision(method_clean))
            round_early = ((6.0 - float(f.end_round)) / 5.0) if finish else 0.0

            T_act = actual_elapsed_seconds(f.method, f.scheduled_rounds, f.end_round, f.end_time_sec)
            T_star = max(T_act, self.epsilon_sec)
            flash_ko = int(is_ko_tko(method_clean) and f.end_round == 1 and f.end_time_sec <= self.flash_ko_sec)

            S, J = parse_judge_margin(f.judge_scores)
            try: sr = int(f.scheduled_rounds)
            except Exception: sr = 3
            Rmax = sr if sr in (3, 5) else 3
            x_cards_raw = (S / Rmax) if (S > 0 and Rmax > 0) else 0.0

            rows.append({
                "fight": f, "finish": float(finish), "round_early": float(round_early),
                "flash_ko": float(flash_ko), "experience": float(experience),
                "layoff": float(layoff), "newWC": float(newWC), "r5": float(r5),
                "title": float(title), "x_cards_raw": float(x_cards_raw),
                "T_act": float(T_act), "T_star": float(T_star),
            })

            self.fight_count[f.i] = ni + 1; self.fight_count[f.j] = nj + 1
            self.last_date[f.i] = f.date;   self.last_date[f.j] = f.date
            self.last_wc[f.i] = f.wc;       self.last_wc[f.j] = f.wc

        return pd.DataFrame(rows)

    def fit_scalers(self, df: pd.DataFrame) -> None:
        to_z = ["round_early", "experience", "layoff", "x_cards_raw"]
        self.means = {k: float(df[k].mean()) for k in to_z}
        self.stds  = {k: (float(df[k].std(ddof=0)) if float(df[k].std(ddof=0)) > 1e-8 else 1.0) for k in to_z}

    def transform(self, df: pd.DataFrame) -> List[np.ndarray]:
        feats: List[np.ndarray] = []
        m, s = self.means, self.stds
        for _, row in df.iterrows():
            round_early_z = (row["round_early"] - m["round_early"]) / s["round_early"]
            experience_z  = (row["experience"]  - m["experience"])  / s["experience"]
            layoff_z      = (row["layoff"]      - m["layoff"])      / s["layoff"]
            x_cards_z     = (row["x_cards_raw"] - m["x_cards_raw"]) / s["x_cards_raw"]
            x_cards       = math.tanh(x_cards_z)
            x = np.array([1.0, row["finish"], round_early_z, x_cards,
                          experience_z, layoff_z, row["newWC"], row["r5"],
                          row["title"], row["flash_ko"]], dtype=float)
            feats.append(x)
        return feats