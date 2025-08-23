from __future__ import annotations
import argparse, re
from typing import List, Dict, Optional
import pandas as pd

from .common.http import get_html
from .common.parse import soup
from .common.io import load_csv, upsert_csv, update_manifest

FIGHTS_CSV = "data/curated/fights.csv"
ROUNDSTAT_CSV = "data/curated/stats_round.csv"

def _to_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except:
        return None

def _parse_round_rows(fight_id: str, fight_url: str) -> List[Dict]:
    """
    Parse per-round totals & significant strikes. Structure varies; be robust.
    Output one row per round, with red/blue sides.
    """
    html = get_html(fight_url, cache_key=f"fight_{fight_id}")
    doc = soup(html)

    # We’ll build a dict per round and fill fields as we find tables
    rounds: dict[int, Dict] = {}

    # Totals table often has KD, TD, Ctrl. Find any table with headers that include "KD" or "Ctrl".
    for tbl in doc.select("table"):
        headers = [th.get_text(" ", strip=True).lower() for th in tbl.select("thead th")]
        if not headers:
            continue
        hdr_join = " ".join(headers)
        is_totals_like = any(k in hdr_join for k in ["kd", "ctrl", "td"])
        is_sig_like = any(k in hdr_join for k in ["sig. str", "head", "body", "leg", "distance", "clinch", "ground"])
        if not (is_totals_like or is_sig_like):
            continue

        # Rows: usually one per round, plus "Totals"
        for tr in tbl.select("tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
            if not cells:
                continue

            # Round number usually first cell
            # Accept "1", "2", ..., or "Round 1"
            m_round = re.search(r"(\d+)", cells[0])
            if not m_round:
                continue
            rnd = int(m_round.group(1))
            if rnd not in rounds:
                rounds[rnd] = {
                    "fight_id": fight_id,
                    "round": rnd,
                    "r_kd": 0, "r_td": 0, "r_ctrl_sec": 0,
                    "r_sig_landed": 0, "r_sig_attempts": 0,
                    "r_head_landed": 0, "r_body_landed": 0, "r_leg_landed": 0,
                    "r_distance_landed": 0, "r_clinch_landed": 0, "r_ground_landed": 0,
                    "b_kd": 0, "b_td": 0, "b_ctrl_sec": 0,
                    "b_sig_landed": 0, "b_sig_attempts": 0,
                    "b_head_landed": 0, "b_body_landed": 0, "b_leg_landed": 0,
                    "b_distance_landed": 0, "b_clinch_landed": 0, "b_ground_landed": 0,
                }

            row = rounds[rnd]

            # Detect a pattern like: "KD 1 0", "TD 2 0", or "Sig. Str. 10 of 25 8 of 20"
            txt = " | ".join(cells).lower()

            # KD
            m = re.search(r"\bkd\b.*?(\d+)\D+(\d+)", txt)
            if m:
                row["r_kd"], row["b_kd"] = int(m.group(1)), int(m.group(2))

            # TD
            m = re.search(r"\btd\b.*?(\d+)\D+(\d+)", txt)
            if m:
                row["r_td"], row["b_td"] = int(m.group(1)), int(m.group(2))

            # Ctrl time "Ctrl 2:30 0:12"
            m = re.search(r"\bctrl\b.*?(\d+):(\d+)\D+(\d+):(\d+)", txt)
            if m:
                row["r_ctrl_sec"] = int(m.group(1)) * 60 + int(m.group(2))
                row["b_ctrl_sec"] = int(m.group(3)) * 60 + int(m.group(4))

            # Sig. Str. "10 of 25 8 of 20"
            m = re.search(r"sig[.\s]*str.*?(\d+)\s*of\s*(\d+)\D+(\d+)\s*of\s*(\d+)", txt)
            if m:
                row["r_sig_landed"], row["r_sig_attempts"] = int(m.group(1)), int(m.group(2))
                row["b_sig_landed"], row["b_sig_attempts"] = int(m.group(3)), int(m.group(4))

            # By target/position — try each, tolerant to missing
            for key, pat in [
                ("head", r"\bhead\b.*?(\d+)\D+(\d+)"),
                ("body", r"\bbody\b.*?(\d+)\D+(\d+)"),
                ("leg",  r"\bleg\b.*?(\d+)\D+(\d+)"),
                ("distance", r"\bdistance\b.*?(\d+)\D+(\d+)"),
                ("clinch",   r"\bclinch\b.*?(\d+)\D+(\d+)"),
                ("ground",   r"\bground\b.*?(\d+)\D+(\d+)"),
            ]:
                m = re.search(pat, txt)
                if m:
                    row[f"r_{key}_landed"], row[f"b_{key}_landed"] = int(m.group(1)), int(m.group(2))

    return list(sorted(rounds.values(), key=lambda r: r["round"]))

def scrape_round_stats(limit_fights: Optional[int] = None) -> pd.DataFrame:
    fights = load_csv(FIGHTS_CSV)
    if fights.empty:
        raise SystemExit("fights.csv not found or empty — run event_fights first.")
    if limit_fights:
        fights = fights.head(limit_fights)

    all_rows: list[dict] = []
    for _, row in fights.iterrows():
        rows = _parse_round_rows(row["fight_id"], row["fight_url"])
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    # Stable column order
    cols = [
        "fight_id","round",
        "r_kd","r_sig_landed","r_sig_attempts","r_td","r_ctrl_sec",
        "r_head_landed","r_body_landed","r_leg_landed",
        "r_distance_landed","r_clinch_landed","r_ground_landed",
        "b_kd","b_sig_landed","b_sig_attempts","b_td","b_ctrl_sec",
        "b_head_landed","b_body_landed","b_leg_landed",
        "b_distance_landed","b_clinch_landed","b_ground_landed",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]
    return df

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scrape per-round stats → stats_round.csv")
    ap.add_argument("--limit-fights", type=int, default=None)
    ap.add_argument("--out", default=ROUNDSTAT_CSV)
    args = ap.parse_args(argv)

    df = scrape_round_stats(limit_fights=args.limit_fights)
    if df.empty:
        print("[rounds] parsed 0 rows; site structure may differ for some fights.")
        return 0

    upsert_csv(df, args.out, keys=["fight_id","round"])
    update_manifest("stats_round.csv", rows=len(df))
    print(f"[rounds] wrote {len(df)} rows → {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())