# src/scraper/ufcstats_fight_rounds.py
from __future__ import annotations

import argparse
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .common.http import get_html
from .common.parse import soup
from .common.io import load_csv, upsert_csv, update_manifest

FIGHTS_CSV = "data/curated/fights.csv"
ROUNDSTAT_CSV = "data/curated/stats_round.csv"

# --- Helpers -----------------------------------------------------------------

_INT = re.compile(r"\d+")
_MMSS = re.compile(r"^(\d{1,2}):(\d{2})$")

def _mmss_to_seconds(x: str | None) -> Optional[int]:
    if not x:
        return None
    m = _MMSS.match(x.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))

def _clean(t: str | None) -> str | None:
    if t is None:
        return None
    return re.sub(r"\s+", " ", t.strip())

def _num(x: str | None) -> Optional[int]:
    try:
        return int(x) if x is not None and x != "" else None
    except Exception:
        return None

def _of_pair(cell_text: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse strings like '10 of 25' → (10, 25)."""
    if not cell_text:
        return (None, None)
    m = re.search(r"(\d+)\s*of\s*(\d+)", cell_text.replace("\xa0", " "), flags=re.I)
    if not m:
        return (None, None)
    return int(m.group(1)), int(m.group(2))

# --- Top section parsers (names, method, round/time, title) ------------------

def _top_names(doc) -> Tuple[Optional[str], Optional[str]]:
    """Extract Red/Blue fighter display names from the header panel."""
    people = doc.select("div.b-fight-details__person")
    if len(people) >= 2:
        r = people[0].select_one(".b-fight-details__person-name a, .b-fight-details__person__name a")
        b = people[1].select_one(".b-fight-details__person-name a, .b-fight-details__person__name a")
        r_name = _clean(r.get_text(" ", strip=True)) if r else None
        b_name = _clean(b.get_text(" ", strip=True)) if b else None
        return r_name, b_name
    return None, None

def _top_meta(doc) -> Dict[str, Optional[str]]:
    """Parse Method, Round, Time, Time format, Details/Judges, Title Bout."""
    out: Dict[str, Optional[str]] = {
        "method": None,
        "end_round": None,
        "end_time": None,
        "scheduled_rounds": None,
        "is_title": None,
        "judge_scores": None,
    }
    for li in doc.select("ul.b-fight-details__list li"):
        raw = li.get_text(" ", strip=True)
        if ":" not in raw:
            continue
        label, val = [t.strip() for t in raw.split(":", 1)]
        L = label.lower()
        if L == "method":
            out["method"] = val
        elif L == "round":
            out["end_round"] = val
        elif L == "time":
            out["end_time"] = val
        elif L in ("time format", "format"):
            m = re.search(r"(\d+)\s*Rnd", val, re.I)
            if m:
                out["scheduled_rounds"] = m.group(1)
        elif L in ("details", "judges"):
            out["judge_scores"] = val
        elif "title bout" in raw.lower():
            out["is_title"] = "1"

    if out["is_title"] is None:
        if doc.find(string=re.compile(r"\btitle bout\b", re.I)):
            out["is_title"] = "1"

    return out

# --- Round tables parsers -----------------------------------------------------

def _iter_round_blocks(table) -> List[Tuple[int, List]]:
    """
    Round tables often have a row 'Round N' followed by two rows (red, blue).
    Returns [(round, [tr_red, tr_blue]), ...]. Falls back to row pairs.
    """
    tbody = table.find("tbody") or table
    rows = tbody.find_all("tr", recursive=False)
    out = []
    i = 0
    while i < len(rows):
        txt = rows[i].get_text(" ", strip=True)
        m = re.search(r"\bround\s*(\d+)\b", txt, re.I)
        if m:
            rnd = int(m.group(1))
            r_row = rows[i + 1] if i + 1 < len(rows) else None
            b_row = rows[i + 2] if i + 2 < len(rows) else None
            if r_row and b_row:
                out.append((rnd, [r_row, b_row]))
            i += 3
        else:
            if i + 1 < len(rows):
                out.append((len(out) + 1, [rows[i], rows[i + 1]]))
                i += 2
            else:
                break
    return out

def _sig_table(doc):
    """Find the Significant Strikes table by headers (Head/Body/Leg, Distance/Clinch/Ground)."""
    for tbl in doc.select("table"):
        hdrs = [th.get_text(" ", strip=True).lower() for th in tbl.select("thead th")]
        joined = " | ".join(hdrs)
        if ("sig" in joined and "head" in joined and "distance" in joined):
            return tbl
    return None

def _totals_table(doc):
    """Find the Totals table by headers (KD, Sig. Str., TD, Ctrl)."""
    for tbl in doc.select("table"):
        hdrs = [th.get_text(" ", strip=True).lower() for th in tbl.select("thead th")]
        joined = " | ".join(hdrs)
        if ("kd" in joined and "sig" in joined and ("ctrl" in joined or "control" in joined)):
            return tbl
    return None

def _row_name(tr) -> str:
    tds = tr.find_all("td")
    if not tds:
        return ""
    a = tds[0].find("a")
    if a:
        return _clean(a.get_text(" ", strip=True)) or ""
    return _clean(tds[0].get_text(" ", strip=True)) or ""

def _parse_rounds(doc, red_name: Optional[str], blue_name: Optional[str]) -> List[Dict]:
    """Parse per-round stats from Totals and Significant Strikes tables."""
    rounds: Dict[int, Dict] = {}

    totals = _totals_table(doc)
    if totals:
        for rnd, pair in _iter_round_blocks(totals):
            rounds[rnd] = {
                "fight_id": None,
                "round": rnd,
                "r_kd": 0, "r_sig_landed": 0, "r_sig_attempts": 0, "r_td": 0, "r_ctrl_sec": 0,
                "r_head_landed": 0, "r_body_landed": 0, "r_leg_landed": 0,
                "r_distance_landed": 0, "r_clinch_landed": 0, "r_ground_landed": 0,
                "b_kd": 0, "b_sig_landed": 0, "b_sig_attempts": 0, "b_td": 0, "b_ctrl_sec": 0,
                "b_head_landed": 0, "b_body_landed": 0, "b_leg_landed": 0,
                "b_distance_landed": 0, "b_clinch_landed": 0, "b_ground_landed": 0,
            }
            for tr in pair:
                name = _row_name(tr)
                cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
                hdrs = [th.get_text(" ", strip=True).lower() for th in totals.select("thead th")]
                def _col_contains(key):
                    for idx, h in enumerate(hdrs):
                        if key in h:
                            return idx
                    return -1
                kd_i = _col_contains("kd")
                sig_i = _col_contains("sig")
                td_i = _col_contains("td")
                ctrl_i = _col_contains("ctrl")

                side = "r" if red_name and name and name.lower() in (red_name or "").lower() else \
                       "b" if blue_name and name and name.lower() in (blue_name or "").lower() else None
                if side is None:
                    side = "r" if tr is pair[0] else "b"

                if kd_i >= 0 and kd_i < len(cells):
                    val = _num(cells[kd_i])
                    if val is not None:
                        rounds[rnd][f"{side}_kd"] = val
                if sig_i >= 0 and sig_i < len(cells):
                    l, a = _of_pair(cells[sig_i])
                    if l is not None:
                        rounds[rnd][f"{side}_sig_landed"] = l or 0
                    if a is not None:
                        rounds[rnd][f"{side}_sig_attempts"] = a or 0
                if td_i >= 0 and td_i < len(cells):
                    l, a = _of_pair(cells[td_i])
                    if l is not None:
                        rounds[rnd][f"{side}_td"] = l or 0
                if ctrl_i >= 0 and ctrl_i < len(cells):
                    sec = _mmss_to_seconds(cells[ctrl_i])
                    if sec is not None:
                        rounds[rnd][f"{side}_ctrl_sec"] = sec

    sig = _sig_table(doc)
    if sig:
        hdrs = [th.get_text(" ", strip=True).lower() for th in sig.select("thead th")]
        idx_map = {
            "head": next((i for i, h in enumerate(hdrs) if "head" in h), -1),
            "body": next((i for i, h in enumerate(hdrs) if "body" in h), -1),
            "leg": next((i for i, h in enumerate(hdrs) if "leg" in h), -1),
            "distance": next((i for i, h in enumerate(hdrs) if "distance" in h), -1),
            "clinch": next((i for i, h in enumerate(hdrs) if "clinch" in h), -1),
            "ground": next((i for i, h in enumerate(hdrs) if "ground" in h), -1),
        }
        for rnd, pair in _iter_round_blocks(sig):
            if rnd not in rounds:
                rounds[rnd] = {
                    "fight_id": None,
                    "round": rnd,
                    "r_kd": 0, "r_sig_landed": 0, "r_sig_attempts": 0, "r_td": 0, "r_ctrl_sec": 0,
                    "r_head_landed": 0, "r_body_landed": 0, "r_leg_landed": 0,
                    "r_distance_landed": 0, "r_clinch_landed": 0, "r_ground_landed": 0,
                    "b_kd": 0, "b_sig_landed": 0, "b_sig_attempts": 0, "b_td": 0, "b_ctrl_sec": 0,
                    "b_head_landed": 0, "b_body_landed": 0, "b_leg_landed": 0,
                    "b_distance_landed": 0, "b_clinch_landed": 0, "b_ground_landed": 0,
                }
            for tr in pair:
                name = _row_name(tr)
                cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
                side = "r" if red_name and name and name.lower() in (red_name or "").lower() else \
                       "b" if blue_name and name and name.lower() in (blue_name or "").lower() else None
                if side is None:
                    side = "r" if tr is pair[0] else "b"
                for key, idx in idx_map.items():
                    if idx >= 0 and idx < len(cells):
                        landed, _attempts = _of_pair(cells[idx])
                        if landed is not None:
                            rounds[rnd][f"{side}_{key}_landed"] = landed or 0

    return [rounds[k] for k in sorted(rounds.keys())]

# --- Main per-fight parser ----------------------------------------------------

def _parse_fight_page(fight_id: str, fight_url: str, referer: Optional[str]) -> Tuple[Dict, List[Dict]]:
    """
    Fetch and parse a UFCStats fight page.
    1) Force a REAL GET to the event page (even if cached) to obtain cookies.
    2) Short delay.
    3) Fetch the fight page with Referer=event (HTTPS then HTTP fallback).
    Returns (details_row, round_rows).
    """
    # 1) Prime session on event page (force network; don't use cache)
    try:
        if referer and referer.startswith("https://www.ufcstats.com/event-details/"):
            # ttl_hours=0 ensures we bypass fresh cache and actually hit the network
            _ = get_html(
                referer,
                cache_key=None,        # don't read/write cache for the priming step
                ttl_hours=0,           # force network
                timeout=15,
                headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
            )
    except Exception:
        pass

    # 2) Tiny pause so requests are not back-to-back
    time.sleep(0.35)

    # 3) Fetch fight page with Referer; try https then http
    html = None
    for proto in ("https", "http"):
        try:
            url = re.sub(r"^https?", proto, fight_url)
            html = get_html(
                url,
                cache_key=f"fight_{fight_id}",
                ttl_hours=720,  # historical pages don't change
                headers={"Referer": referer} if referer else None,
            )
            break
        except Exception:
            html = None
            time.sleep(0.2)

    if html is None:
        return (
            {
                "fight_id": fight_id,
                "method": None,
                "end_round": None,
                "end_time_sec": None,
                "scheduled_rounds": None,
                "is_title": None,
                "judge_scores": None,
            },
            []
        )

    doc = soup(html)

    r_name, b_name = _top_names(doc)
    meta = _top_meta(doc)
    details_row = {
        "fight_id": fight_id,
        "method": meta.get("method"),
        "end_round": _num(meta.get("end_round")) if meta.get("end_round") else None,
        "end_time_sec": _mmss_to_seconds(meta.get("end_time")),
        "scheduled_rounds": _num(meta.get("scheduled_rounds")) if meta.get("scheduled_rounds") else None,
        "is_title": 1 if meta.get("is_title") == "1" else 0 if meta.get("is_title") is not None else None,
        "judge_scores": meta.get("judge_scores"),
    }

    round_rows = _parse_rounds(doc, r_name, b_name)
    for rr in round_rows:
        rr["fight_id"] = fight_id

    return details_row, round_rows

# --- Orchestration ------------------------------------------------------------

def scrape_round_stats(limit_fights: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fights = load_csv(FIGHTS_CSV)
    if fights.empty:
        raise SystemExit("fights.csv not found — run event_fights first.")

    if limit_fights:
        fights = fights.head(limit_fights)

    details_updates: List[Dict] = []
    all_rounds: List[Dict] = []
    skipped: List[str] = []

    for i, f in enumerate(fights.itertuples(index=False), 1):
        fid = f.fight_id
        eid = getattr(f, "event_id", None)
        referer = f"https://www.ufcstats.com/event-details/{eid}" if eid else "https://www.ufcstats.com/"
        furl = getattr(f, "fight_url", None) or f"https://www.ufcstats.com/fight-details/{fid}"

        details, rounds = _parse_fight_page(fid, furl, referer=referer)
        if not details["method"] and not rounds:
            skipped.append(fid)
        else:
            details_updates.append(details)
            all_rounds.extend(rounds)

        if i % 25 == 0:
            print(f"[rounds] processed {i}/{len(fights)} fights…", flush=True)
        time.sleep(0.3)  # polite pacing

    df_rounds = pd.DataFrame(all_rounds)
    round_cols = [
        "fight_id","round",
        "r_kd","r_sig_landed","r_sig_attempts","r_td","r_ctrl_sec",
        "r_head_landed","r_body_landed","r_leg_landed",
        "r_distance_landed","r_clinch_landed","r_ground_landed",
        "b_kd","b_sig_landed","b_sig_attempts","b_td","b_ctrl_sec",
        "b_head_landed","b_body_landed","b_leg_landed",
        "b_distance_landed","b_clinch_landed","b_ground_landed",
    ]
    for c in round_cols:
        if c not in df_rounds.columns:
            df_rounds[c] = 0
    if not df_rounds.empty:
        df_rounds = df_rounds[round_cols]

    df_details = pd.DataFrame(details_updates, columns=[
        "fight_id","method","end_round","end_time_sec","scheduled_rounds","is_title","judge_scores"
    ])

    if skipped:
        pd.Series(skipped, name="fight_id").to_csv("data/curated/_rounds_skipped.csv", index=False)
        print(f"[rounds] skipped {len(skipped)} fights due to fetch issues → data/curated/_rounds_skipped.csv")

    return df_details, df_rounds

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scrape per-fight details + per-round stats → stats_round.csv and update fights.csv")
    ap.add_argument("--limit-fights", type=int, default=None, help="Process only the first N fights (by current CSV order)")
    ap.add_argument("--rounds-out", default=ROUNDSTAT_CSV, help="Output CSV for per-round stats")
    ap.add_argument("--fights-out", default=FIGHTS_CSV, help="Upsert back into fights.csv with method/round/time/etc.")
    args = ap.parse_args(argv)

    df_details, df_rounds = scrape_round_stats(limit_fights=args.limit_fights)

    if df_details.empty and df_rounds.empty:
        print("[rounds] parsed 0 rows (check fight pages).")
        return 0

    if not df_details.empty:
        upsert_csv(df_details, args.fights_out, keys=["fight_id"])
        update_manifest("fights.csv", rows=len(df_details))
        print(f"[rounds] updated fights.csv details for {len(df_details)} fights")

    if not df_rounds.empty:
        upsert_csv(df_rounds, args.rounds_out, keys=["fight_id","round"])
        update_manifest("stats_round.csv", rows=len(df_rounds))
        print(f"[rounds] wrote {len(df_rounds)} per-round rows → {args.rounds_out}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())