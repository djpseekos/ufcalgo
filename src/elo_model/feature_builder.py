# src/elo_model/feature_builder.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .helpers import is_decision, is_ko_tko, parse_judge_margin
from .time_utils import actual_elapsed_seconds


class FeatureBuilder:
    """
    Extended FeatureBuilder that optionally incorporates per-fight stats
    aggregated from stats_round.csv (KD, SIG, TD, CTRL) as per the paper.

    Backwards-compatible:
      - If no stats are provided, features remain identical to the prior 10-dim vector.
      - If stats are provided (plus fights meta), we append 4 features:
        kd_diff_z, sig_diff_z, td_diff_z, ctrl_diff_tanh
    """

    def __init__(
        self,
        epsilon_sec: int = 30,
        flash_ko_sec: int = 30,
        # Optional enriched inputs for K-engineering signals:
        fights_meta_df: Optional[pd.DataFrame] = None,
        stats_round_df: Optional[pd.DataFrame] = None,
    ):
        self.epsilon_sec = int(epsilon_sec)
        self.flash_ko_sec = int(flash_ko_sec)

        # Stateful trackers for time-dependent features (unchanged)
        self.last_date: Dict[str, pd.Timestamp] = {}
        self.fight_count: Dict[str, int] = {}
        self.last_wc: Dict[str, str] = {}

        # Z-scaling dictionaries
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

        # --- Optional: precomputed per-fight stats map keyed by (fighter_id, date) ---
        self._use_stats: bool = False
        self._stats_by_fd: Dict[Tuple[str, pd.Timestamp], Dict[str, float]] = {}

        if fights_meta_df is not None and stats_round_df is not None:
            self._prepare_stats_maps(fights_meta_df, stats_round_df)
            self._use_stats = True

    # ------------------------------------------------------------------
    # Round stats ingestion → per-fight per-fighter totals → keyed by (fighter_id, date)
    # ------------------------------------------------------------------
    def _prepare_stats_maps(self, fights_meta_df: pd.DataFrame, stats_round_df: pd.DataFrame) -> None:
        """
        Build self._stats_by_fd[(fighter_id, date)] = { 'kd':..., 'sig':..., 'td':..., 'ctrl':... }
        Requirements:
          fights_meta_df: columns [fight_id, date, r_fighter_id, b_fighter_id]
          stats_round_df: columns [fight_id, fighter_corner, kd, sig_landed, td_landed, ctrl_time_sec, ...]
        """
        # Ensure date is Timestamp
        fm = fights_meta_df.copy()
        if "date" not in fm.columns:
            raise ValueError("fights_meta_df must include a 'date' column")
        fm["date"] = pd.to_datetime(fm["date"])

        # Aggregate round stats to fight-level per corner
        needed_cols = ["fight_id", "fighter_corner", "kd", "sig_landed", "td_landed", "ctrl_time_sec"]
        for c in needed_cols:
            if c not in stats_round_df.columns:
                raise ValueError(f"stats_round_df missing required column '{c}'")

        agg = (
            stats_round_df[needed_cols]
            .groupby(["fight_id", "fighter_corner"], as_index=False)
            .sum()
        )

        # Map corner → fighter_id and date
        # We need to merge once per corner
        # Build a slim mapping: fight_id -> date, r_fighter_id, b_fighter_id
        slim = fm[["fight_id", "date", "r_fighter_id", "b_fighter_id"]].copy()

        # Join to get date and fighter ids
        merged = agg.merge(slim, on="fight_id", how="left")

        # Turn corner into fighter_id
        def corner_to_fid(row) -> Optional[str]:
            c = str(row["fighter_corner"]).strip().upper()
            if c == "R":
                return str(row["r_fighter_id"])
            if c == "B":
                return str(row["b_fighter_id"])
            return None

        merged["fighter_id"] = merged.apply(corner_to_fid, axis=1)
        merged = merged.dropna(subset=["fighter_id", "date"])

        # Build map
        stats_map: Dict[Tuple[str, pd.Timestamp], Dict[str, float]] = {}
        for _, r in merged.iterrows():
            key = (str(r["fighter_id"]), pd.to_datetime(r["date"]))
            stats_map[key] = {
                "kd": float(r.get("kd", 0.0) or 0.0),
                "sig": float(r.get("sig_landed", 0.0) or 0.0),
                "td": float(r.get("td_landed", 0.0) or 0.0),
                "ctrl": float(r.get("ctrl_time_sec", 0.0) or 0.0),  # seconds
            }

        self._stats_by_fd = stats_map

    # ------------------------------------------------------------------
    # Core feature collection (unchanged + new stats-based diffs)
    # ------------------------------------------------------------------
    def collect_raw_features(self, fights: List) -> pd.DataFrame:
        """
        Convert a sequence of FightRow into a DataFrame with raw features (pre-scaling),
        and carry temporal state (fight_count, last_date, last_wc).
        If stats were provided, append KD/SIG/TD/control differentials (red - blue).
        """
        rows = []
        for f in fights:
            # --- existing temporal features ---
            ni = self.fight_count.get(f.i, 0)
            nj = self.fight_count.get(f.j, 0)
            experience = 1.0 / math.sqrt(1.0 + ni + nj)

            di = self.last_date.get(f.i)
            dj = self.last_date.get(f.j)
            vals = []
            if di is not None:
                vals.append((f.date - di).days)
            if dj is not None:
                vals.append((f.date - dj).days)
            layoff_days = float(np.median(vals)) if vals else 365.0
            layoff = math.log1p(layoff_days / 180.0)

            newWC = int(self.last_wc.get(f.i) != f.wc or self.last_wc.get(f.j) != f.wc)
            r5 = int(f.scheduled_rounds == 5)
            title = int(f.is_title)

            method_clean = (f.method or "").strip()
            finish = int(not is_decision(method_clean))
            round_early = ((6.0 - float(f.end_round)) / 5.0) if finish else 0.0

            # actual elapsed seconds in fight; epsilon floor
            T_act = actual_elapsed_seconds(f.method, f.scheduled_rounds, f.end_round, f.end_time_sec)
            T_star = max(T_act, self.epsilon_sec)

            flash_ko = int(is_ko_tko(method_clean) and f.end_round == 1 and f.end_time_sec <= self.flash_ko_sec)

            # judges card margin proxy
            S, J = parse_judge_margin(f.judge_scores)
            try:
                sr = int(f.scheduled_rounds)
            except Exception:
                sr = 3
            Rmax = sr if sr in (3, 5) else 3
            x_cards_raw = (S / Rmax) if (S > 0 and Rmax > 0) else 0.0

            # --- NEW: per-minute differentials from round stats (red - blue) ---
            kd_diff = 0.0
            sig_diff = 0.0
            td_diff = 0.0
            ctrl_diff = 0.0
            if self._use_stats:
                # look up totals for each fighter on this date
                key_i = (f.i, f.date)
                key_j = (f.j, f.date)
                si = self._stats_by_fd.get(key_i, {"kd": 0.0, "sig": 0.0, "td": 0.0, "ctrl": 0.0})
                sj = self._stats_by_fd.get(key_j, {"kd": 0.0, "sig": 0.0, "td": 0.0, "ctrl": 0.0})

                # per-minute rates using T_star seconds
                scale = 60.0 / float(T_star)
                kd_i = si["kd"] * scale
                kd_j = sj["kd"] * scale
                sig_i = si["sig"] * scale
                sig_j = sj["sig"] * scale
                td_i = si["td"] * scale
                td_j = sj["td"] * scale

                # control share (seconds over T_star)
                ctrl_i = si["ctrl"] / float(T_star)
                ctrl_j = sj["ctrl"] / float(T_star)

                kd_diff = kd_i - kd_j
                sig_diff = sig_i - sig_j
                td_diff = td_i - td_j
                ctrl_diff = ctrl_i - ctrl_j  # share difference in [-1,1] ideally

            rows.append({
                "fight": f,
                # legacy raw features
                "finish": float(finish),
                "round_early": float(round_early),
                "flash_ko": float(flash_ko),
                "experience": float(experience),
                "layoff": float(layoff),
                "newWC": float(newWC),
                "r5": float(r5),
                "title": float(title),
                "x_cards_raw": float(x_cards_raw),
                "T_act": float(T_act),
                "T_star": float(T_star),
                # new raw diffs (may be zeros if no stats)
                "kd_diff": float(kd_diff),
                "sig_diff": float(sig_diff),
                "td_diff": float(td_diff),
                "ctrl_diff": float(ctrl_diff),
            })

            # advance temporal state
            self.fight_count[f.i] = ni + 1
            self.fight_count[f.j] = nj + 1
            self.last_date[f.i] = f.date
            self.last_date[f.j] = f.date
            self.last_wc[f.i] = f.wc
            self.last_wc[f.j] = f.wc

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Scaling on TRAIN only
    # ------------------------------------------------------------------
    def fit_scalers(self, df: pd.DataFrame) -> None:
        """
        Fit means/stds for z-scoring on TRAIN (called once).
        Always includes legacy z-features; conditionally includes new diffs if present.
        """
        to_z = ["round_early", "experience", "layoff", "x_cards_raw"]

        # If we used stats in collection, also scale these diffs
        if "kd_diff" in df.columns:
            to_z += ["kd_diff", "sig_diff", "td_diff", "ctrl_diff"]

        self.means = {k: float(df[k].mean()) for k in to_z}
        self.stds = {}
        for k in to_z:
            stdv = float(df[k].std(ddof=0))
            self.stds[k] = stdv if stdv > 1e-8 else 1.0

    # ------------------------------------------------------------------
    # Transform to final feature vectors
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> List[np.ndarray]:
        feats: List[np.ndarray] = []
        m, s = self.means, self.stds

        use_stats = all(k in df.columns for k in ["kd_diff", "sig_diff", "td_diff", "ctrl_diff"]) and \
                    all(k in m for k in ["kd_diff", "sig_diff", "td_diff", "ctrl_diff"])

        for _, row in df.iterrows():
            # legacy z’s
            round_early_z = (row["round_early"] - m["round_early"]) / s["round_early"]
            experience_z  = (row["experience"]  - m["experience"])  / s["experience"]
            layoff_z      = (row["layoff"]      - m["layoff"])      / s["layoff"]
            x_cards_z     = (row["x_cards_raw"] - m["x_cards_raw"]) / s["x_cards_raw"]
            x_cards       = math.tanh(x_cards_z)

            base = [
                1.0,
                float(row["finish"]),
                float(round_early_z),
                float(x_cards),
                float(experience_z),
                float(layoff_z),
                float(row["newWC"]),
                float(row["r5"]),
                float(row["title"]),
                float(row["flash_ko"]),
            ]

            if use_stats:
                kd_z   = (row["kd_diff"]  - m["kd_diff"])  / s["kd_diff"]
                sig_z  = (row["sig_diff"] - m["sig_diff"]) / s["sig_diff"]
                td_z   = (row["td_diff"]  - m["td_diff"])  / s["td_diff"]
                ctrl_z = (row["ctrl_diff"]- m["ctrl_diff"])/ s["ctrl_diff"]
                ctrl_t = math.tanh(ctrl_z)

                x = np.array(base + [float(kd_z), float(sig_z), float(td_z), float(ctrl_t)], dtype=float)
            else:
                x = np.array(base, dtype=float)

            feats.append(x)

        return feats