# src/scraper/ufcstats_fight_rounds.py
from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd

from .common import http as http_common
from .common.parse import soup
from .common.io import load_csv, upsert_csv, update_manifest


FIGHTS_CSV = "data/curated/fights.csv"
ROUNDSTAT_CSV = "data/curated/stats_round.csv"
DEBUG_DIR = "data/debug"

# Slightly faster HTTP profile; still polite due to per-host throttle in http_common
http_common._cfg.rate_per_sec = 4.0
http_common._cfg.retries = 2
http_common._cfg.backoff = 0.3

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_MMSS = re.compile(r"^(\d{1,2}):(\d{2})$")
_NON_NAME = re.compile(r"[^a-z .'\-]")
_SPACEY = re.compile(r"\s+")
_RD_LAB = re.compile(r"^rd\s*([1-5])\b", re.I)
_PAIR = re.compile(r"(\d+)\s*of\s*(\d+)", re.I)
_ID_RX = re.compile(r"/fighter-details/([a-f0-9]+)", re.I)
_JUDGE_RX = re.compile(r"\b\d{2}\s*-\s*\d{2}\b")
_TITLE_RX = re.compile(r"\b(title\s+bout|title\s+fight|interim\s+title)\b", re.I)

ROUND_COLS = [
    "fight_id", "round",
    # Totals / base
    "r_kd", "r_sig_landed", "r_sig_attempts",
    "r_total_landed", "r_total_attempts",
    "r_td", "r_td_attempts", "r_sub_att", "r_rev", "r_ctrl_sec",
    "b_kd", "b_sig_landed", "b_sig_attempts",
    "b_total_landed", "b_total_attempts",
    "b_td", "b_td_attempts", "b_sub_att", "b_rev", "b_ctrl_sec",
    # Target
    "r_head_landed", "r_head_attempts", "r_body_landed", "r_body_attempts", "r_leg_landed", "r_leg_attempts",
    "b_head_landed", "b_head_attempts", "b_body_landed", "b_body_attempts", "b_leg_landed", "b_leg_attempts",
    # Position
    "r_distance_landed", "r_distance_attempts", "r_clinch_landed", "r_clinch_attempts", "r_ground_landed", "r_ground_attempts",
    "b_distance_landed", "b_distance_attempts", "b_clinch_landed", "b_clinch_attempts", "b_ground_landed", "b_ground_attempts",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_missing_text(s: Optional[str]) -> bool:
    if s is None:
        return True
    t = s.strip().lower()
    return t == "" or t == "---" or t == "—" or t == "–"


def _mmss_to_seconds(x: Optional[str]) -> Optional[int]:
    if _is_missing_text(x):
        return None
    m = _MMSS.match(x.strip())
    if not m:   
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def _clean(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return _SPACEY.sub(" ", x.strip())


def _num(x: Optional[str]) -> Optional[int]:
    if _is_missing_text(x):
        return None
    s = x.strip()
    try:
        return int(s)
    except Exception:
        return None


def _is_str(x: object) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _pair(cell_text: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse '10 of 25' → (10, 25)."""
    if _is_missing_text(cell_text):
        return (None, None)
    m = _PAIR.search(cell_text.replace("\xa0", " "))
    if not m:
        return (None, None)
    return int(m.group(1)), int(m.group(2))


def _pair_from_texts(txt: str) -> Tuple[Optional[int], Optional[int]]:
    l, a = _pair(txt)
    return (l if l is not None else None, a if a is not None else None)


def _norm_name(n: Optional[str]) -> str:
    if not n:
        return ""
    n = n.replace("\xa0", " ")
    n = _SPACEY.sub(" ", n).strip().lower()
    return _NON_NAME.sub("", n)


def _table_headers(tbl) -> List[str]:
    """Robust header extractor."""
    hdrs = [th.get_text(" ", strip=True).lower() for th in tbl.select("thead th")]
    if not hdrs:
        hdrs = [th.get_text(" ", strip=True).lower() for th in tbl.select("th")]
    if not hdrs:
        first = tbl.find("tr")
        if first:
            hdrs = [td.get_text(" ", strip=True).lower() for td in first.find_all(["th", "td"])]
    return hdrs


def _find_col(hdrs: List[str], key: str) -> int:
    """Find a column index by whole word; else fallback to substring; else -1."""
    pat = re.compile(rf"\b{re.escape(key)}\b")
    for i, h in enumerate(hdrs):
        if pat.search(h):
            return i
    for i, h in enumerate(hdrs):
        if key in h:
            return i
    return -1


def _row_name(tr) -> str:
    tds = tr.find_all("td")
    if not tds:
        return ""
    a = tds[0].find("a")
    if a:
        return _clean(a.get_text(" ", strip=True)) or ""
    return _clean(tds[0].get_text(" ", strip=True)) or ""


def _guess_side(row_idx: int, tr, rnorm: str, bnorm: str) -> str:
    """Pick 'r' or 'b' using row name match, else positional fallback."""
    name = _row_name(tr)
    tnorm = _norm_name(name)
    if tnorm and rnorm and tnorm in rnorm:
        return "r"
    if tnorm and bnorm and tnorm in bnorm:
        return "b"
    return "r" if row_idx == 0 else "b"


def _td_texts(td) -> List[str]:
    """Return [red_text, blue_text] for stacked cells, or [] if not stacked."""
    ps = td.find_all("p", recursive=False)
    if len(ps) >= 2:
        return [
            ps[0].get_text(" ", strip=True),
            ps[1].get_text(" ", strip=True),
        ]
    return []


def _iter_round_blocks(table) -> List[Tuple[int, object]]:
    """
    Return [(round_number, data_tr)] where data_tr is the first <tr> after the
    'Round N' header row. Works for:
      - stacked layout: one data_tr with two <p> per <td>
      - legacy layout: two consecutive data <tr> (red, blue)
    """
    rows = table.find_all("tr")
    out: List[Tuple[int, object]] = []
    i = 0
    while i < len(rows):
        txt = rows[i].get_text(" ", strip=True) if rows[i] else ""
        m = re.search(r"\bround\s*(\d+)\b", txt, re.I)
        if m:
            rnd = int(m.group(1))
            data_tr = rows[i + 1] if i + 1 < len(rows) else None
            if data_tr:
                out.append((rnd, data_tr))
                i += 2
            else:
                i += 1
        else:
            i += 1
    return out


def _looks_like_fight_page(txt: str) -> bool:
    if not txt:
        return False
    low = txt.lower()
    return (
        "b-fight-details__person" in low
        or "b-fight-details__table" in low
        or "b-fight-details__list" in low
        or "fight details" in low
    )

def _cell_part_text(td, part_idx: int) -> Optional[str]:
    """
    Returns the text of the Nth <p class="b-fight-details__table-text"> in this <td>.
    If there are no <p>s, returns the whole cell text (legacy two-row layout).
    part_idx: 0 for top/first (winner), 1 for bottom/second (loser).
    """
    if td is None:
        return None
    ps = td.select("p.b-fight-details__table-text")
    if ps:
        if 0 <= part_idx < len(ps):
            return ps[part_idx].get_text(" ", strip=True)
        return None
    # legacy row: entire cell belongs to one fighter
    return td.get_text(" ", strip=True) or None


def _iter_round_blocks(table) -> List[Tuple[int, List, bool]]:
    """
    Return a list of (round_number, rows, is_stacked):
      - stacked: rows == [single_tr] with two <p> per <td> (top=winner, bottom=loser)
      - legacy : rows == [tr_red, tr_blue]
    Works across multiple <thead>/<tbody> segments.
    """
    rows = table.find_all("tr")
    out: List[Tuple[int, List, bool]] = []
    i = 0
    while i < len(rows):
        txt = rows[i].get_text(" ", strip=True) if rows[i] else ""
        if not txt:
            i += 1
            continue

        m = re.search(r"\bround\s*(\d+)\b", txt, re.I)
        if m:
            rnd = int(m.group(1))
            # advance to first data row after this header
            j = i + 1
            # skip any empty/non-data rows
            while j < len(rows) and not rows[j].find_all("td"):
                j += 1
            if j >= len(rows):
                i = j
                continue

            tr1 = rows[j]
            # Detect "stacked" by checking if most numeric columns have 2 <p> blocks
            tds1 = tr1.find_all("td")
            stacked = False
            if tds1:
                # Count how many cells have ≥2 <p> elements
                cells_with_two_p = sum(1 for td in tds1 if len(td.select("p.b-fight-details__table-text")) >= 2)
                # Heuristic: if the fighter column + at least one numeric column are stacked → stacked
                stacked = cells_with_two_p >= 2

            if stacked:
                out.append((rnd, [tr1], True))
                i = j + 1
                continue

            # Legacy: expect the next row to be opponent
            k = j + 1
            # skip non-data rows (e.g., stray thead between)
            while k < len(rows) and not rows[k].find_all("td"):
                k += 1
            if k < len(rows):
                out.append((rnd, [tr1, rows[k]], False))
                i = k + 1
            else:
                i = k
        else:
            # Not a round header; try to pair current + next as a best-effort legacy block
            if i + 1 < len(rows) and rows[i].find_all("td") and rows[i+1].find_all("td"):
                out.append((len(out) + 1, [rows[i], rows[i + 1]], False))
                i += 2
            else:
                i += 1
    return out

# --------------------------------------------------------------------------- #
# Charts (modern layout) helpers
# --------------------------------------------------------------------------- #

def _parse_chart_panel(panel) -> dict[int, dict[str, tuple[tuple[int, int], tuple[int, int]]]]:
    """
    Parse one chart panel ('Landed by target' or 'Landed by position').
    Returns: { round: { metric: ((r_landed,r_attempts), (b_landed,b_attempts)) } }
    metric ∈ {'head','body','leg'} or {'distance','clinch','ground'}.
    """
    out: dict[int, dict[str, tuple[tuple[int, int], tuple[int, int]]]] = {}
    if not panel:
        return out

    for rd_block in panel.select(".b-fight-details__charts-col-row, .b-fight-details__charts-col-row.clearfix"):
        txt = rd_block.get_text(" ", strip=True)
        m = _RD_LAB.search(txt)
        if not m:
            continue
        rnd = int(m.group(1))
        out.setdefault(rnd, {})

        for table in rd_block.select(".b-fight-details__charts-table"):
            metric = None
            sib = table.previous_sibling
            hops = 0
            while sib and hops < 6 and not metric:
                if getattr(sib, "get_text", None):
                    t = sib.get_text(" ", strip=True).lower()
                    for k in ("head", "body", "leg", "distance", "clinch", "ground"):
                        if k in t:
                            metric = k
                            break
                sib = sib.previous_sibling
                hops += 1
            if not metric:
                continue

            nums = []
            for num_el in table.select(".b-fight-details__charts-num"):
                s = num_el.get_text(" ", strip=True)
                mm = _PAIR.search(s)
                if mm:
                    nums.append((int(mm.group(1)), int(mm.group(2))))
            if len(nums) >= 2:
                out[rnd][metric] = (nums[0], nums[1])  # (red, blue)
    return out


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class FightDetails:
    fight_id: str
    r_fighter_id: Optional[str] = None
    b_fighter_id: Optional[str] = None
    r_fighter_name: Optional[str] = None
    b_fighter_name: Optional[str] = None

    method: Optional[str] = None
    end_round: Optional[int] = None
    end_time_sec: Optional[int] = None
    scheduled_rounds: Optional[int] = None
    is_title: Optional[int] = None
    judge_scores: Optional[str] = None
    debug_li: str = ""


@dataclass(slots=True)
class RoundRow:
    fight_id: str
    round: int
    # Totals
    r_kd: Optional[int] = None
    r_sig_landed: Optional[int] = None
    r_sig_attempts: Optional[int] = None
    r_total_landed: Optional[int] = None
    r_total_attempts: Optional[int] = None
    r_td: Optional[int] = None
    r_td_attempts: Optional[int] = None
    r_sub_att: Optional[int] = None
    r_rev: Optional[int] = None
    r_ctrl_sec: Optional[int] = None
    b_kd: Optional[int] = None
    b_sig_landed: Optional[int] = None
    b_sig_attempts: Optional[int] = None
    b_total_landed: Optional[int] = None
    b_total_attempts: Optional[int] = None
    b_td: Optional[int] = None
    b_td_attempts: Optional[int] = None
    b_sub_att: Optional[int] = None
    b_rev: Optional[int] = None
    b_ctrl_sec: Optional[int] = None
    # Target
    r_head_landed: Optional[int] = None
    r_head_attempts: Optional[int] = None
    r_body_landed: Optional[int] = None
    r_body_attempts: Optional[int] = None
    r_leg_landed: Optional[int] = None
    r_leg_attempts: Optional[int] = None
    b_head_landed: Optional[int] = None
    b_head_attempts: Optional[int] = None
    b_body_landed: Optional[int] = None
    b_body_attempts: Optional[int] = None
    b_leg_landed: Optional[int] = None
    b_leg_attempts: Optional[int] = None
    # Position
    r_distance_landed: Optional[int] = None
    r_distance_attempts: Optional[int] = None
    r_clinch_landed: Optional[int] = None
    r_clinch_attempts: Optional[int] = None
    r_ground_landed: Optional[int] = None
    r_ground_attempts: Optional[int] = None
    b_distance_landed: Optional[int] = None
    b_distance_attempts: Optional[int] = None
    b_clinch_landed: Optional[int] = None
    b_clinch_attempts: Optional[int] = None
    b_ground_landed: Optional[int] = None
    b_ground_attempts: Optional[int] = None


# --------------------------------------------------------------------------- #
# Fight page parser
# --------------------------------------------------------------------------- #

class FightPageParser:
    def __init__(self, doc) -> None:
        self.doc = doc

    def top_people(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Return (r_id, r_name, b_id, b_name) from the header cards."""
        people = self.doc.select("div.b-fight-details__person")
        r_id = r_name = b_id = b_name = None
        if len(people) >= 2:
            def _extract(block):
                a = block.select_one(".b-fight-details__person-name a, .b-fight-details__person__name a")
                if not a:
                    return (None, None)
                name = _clean(a.get_text(" ", strip=True))
                href = a.get("href", "")
                m = _ID_RX.search(href)
                fid = m.group(1) if m else None
                return (fid, name)
            r_id, r_name = _extract(people[0])
            b_id, b_name = _extract(people[1])
        return r_id, r_name, b_id, b_name

    def top_meta(self) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {
            "method": None,
            "end_round": None,
            "end_time": None,
            "scheduled_rounds": None,
            "is_title": None,
            "judge_scores": None,
            "debug_li": "",
        }

        lines: List[str] = []

        def assign(label: str, value: str) -> None:
            L = (label or "").strip().rstrip(":").lower()
            v = value.strip()
            if not L:
                return
            if L == "method":
                out["method"] = v
            elif L == "round":
                out["end_round"] = v
            elif L == "time":
                out["end_time"] = v
            elif L in ("time format", "format"):
                m = re.search(r"(\d+)\s*Rnd", v, re.I)
                if m:
                    out["scheduled_rounds"] = m.group(1)
            elif L in ("details", "judges"):
                out["judge_scores"] = v
            if "title bout" in (label + " " + value).lower():
                out["is_title"] = "1"

        content = self.doc.select_one("div.b-fight-details__content")
        if content:
            for block in content.select(
                ".b-fight-details__text, .b-fight-details__text-item, .b-fight-details__text-item_first"
            ):
                label_el = block.select_one(".b-fight-details__label")
                txt = block.get_text(" ", strip=True)
                lines.append(txt)
                if "title bout" in txt.lower():
                    out["is_title"] = "1"
                if label_el:
                    label_txt = label_el.get_text(" ", strip=True)
                    val = re.sub(r"^\s*" + re.escape(label_txt) + r"\s*:?\s*", "", txt, flags=re.I)
                    assign(label_txt, val)

        if not out["method"] and not out["end_round"] and not out["end_time"]:
            for li in self.doc.select("ul.b-fight-details__list li"):
                raw_all = li.get_text(" ", strip=True)
                lines.append(raw_all)
                label_el = li.select_one("i, .b-fight-details__label")
                label_txt = (label_el.get_text(" ", strip=True) if label_el else "")
                if label_el:
                    val = re.sub(r"^\s*" + re.escape(label_txt) + r"\s*:?\s*", "", raw_all, flags=re.I).strip()
                else:
                    low = raw_all.lower()
                    val = raw_all
                    for c in ("method", "round", "time format", "format", "time", "details", "judges"):
                        if low.startswith(c):
                            label_txt = c
                            val = raw_all.split(":", 1)[-1].strip() if ":" in raw_all else raw_all[len(c):].strip(" :-\u00a0")
                            break
                assign(label_txt, val)

        out["debug_li"] = "\n".join(lines)

        if out["is_title"] is None:
            banner_txt = ""
            head = self.doc.select_one(".b-fight-details__fight, .b-fight-details")
            if head:
                banner_txt = head.get_text(" ", strip=True)
            if _TITLE_RX.search(banner_txt or ""):
                out["is_title"] = "1"

        if not out.get("judge_scores"):
            blob = " | ".join(lines)
            scores = _JUDGE_RX.findall(blob)
            if scores:
                out["judge_scores"] = ", ".join(scores)

        return out

    def _find_totals_table(self):
        """
        Pick the *per-round* totals table:
        - has headers with KD/Sig/Total/TD/Ctrl
        - and contains 'Round 1' section headers somewhere in the same table
        """
        candidates = []
        for tbl in self.doc.select("table"):
            hdrs = _table_headers(tbl)
            joined = " | ".join(hdrs)
            if not hdrs:
                continue
            if ("kd" in joined and "sig" in joined and ("total" in joined or "total str" in joined)):
                txt = tbl.get_text(" ", strip=True).lower()
                if "round 1" in txt:
                    return tbl
                candidates.append(tbl)
        # fallback: choose the one whose text includes any 'Round N' marker
        for tbl in candidates:
            if re.search(r"\bround\s*[1-5]\b", tbl.get_text(" ", strip=True), re.I):
                return tbl
        return candidates[0] if candidates else None

    def _find_sig_table(self):
        """
        Pick the *per-round* significant strikes by target/position table:
        - headers include Sig/Head/Body/Leg and/or Distance/Clinch/Ground
        - table text includes 'Round 1'
        """
        candidates = []
        for tbl in self.doc.select("table"):
            hdrs = _table_headers(tbl)
            if not hdrs:
                continue
            has_sig = any("sig" in h for h in hdrs)
            has_target = any(k in h for h in ("head", "body", "leg"))
            has_pos = any(k in h for h in ("distance", "clinch", "ground"))
            if has_sig and (has_target or has_pos):
                txt = tbl.get_text(" ", strip=True).lower()
                if "round 1" in txt:
                    return tbl
                candidates.append(tbl)
        # fallback: the one with explicit round headers in its text
        for tbl in candidates:
            if re.search(r"\bround\s*[1-5]\b", tbl.get_text(" ", strip=True), re.I):
                return tbl
        return candidates[0] if candidates else None

    # Charts fallback (used rarely now)
    def _parse_charts_rounds(self, fight_id: str) -> List[RoundRow]:
        def _panel_with(title_sub: str):
            title_sub = title_sub.lower()
            for head in self.doc.select(".b-fight-details__charts-head"):
                if title_sub in head.get_text(" ", strip=True).lower():
                    pnl = head.find_next("div", class_="b-fight-details__charts-body")
                    if not pnl:
                        pnl = head.find_next("div", class_="b-fight-details__charts")
                    return pnl
            return None

        pnl_target = _panel_with("landed by target")
        pnl_pos = _panel_with("landed by position")

        by_target = _parse_chart_panel(pnl_target) if pnl_target else {}
        by_pos = _parse_chart_panel(pnl_pos) if pnl_pos else {}

        rows: dict[int, RoundRow] = {}
        for rnd in sorted(set(by_target) | set(by_pos)):
            rr = rows.setdefault(rnd, RoundRow(fight_id=fight_id, round=rnd))

            tgt = by_target.get(rnd, {})
            for metric in ("head", "body", "leg"):
                pair = tgt.get(metric)
                if pair:
                    (r_la, r_at), (b_la, b_at) = pair
                    setattr(rr, f"r_{metric}_landed", r_la)
                    setattr(rr, f"r_{metric}_attempts", r_at)
                    setattr(rr, f"b_{metric}_landed", b_la)
                    setattr(rr, f"b_{metric}_attempts", b_at)

            pos = by_pos.get(rnd, {})
            for metric in ("distance", "clinch", "ground"):
                pair = pos.get(metric)
                if pair:
                    (r_la, r_at), (b_la, b_at) = pair
                    setattr(rr, f"r_{metric}_landed", r_la)
                    setattr(rr, f"r_{metric}_attempts", r_at)
                    setattr(rr, f"b_{metric}_landed", b_la)
                    setattr(rr, f"b_{metric}_attempts", b_at)

        return [rows[k] for k in sorted(rows.keys())]

    # ---- per-round parsing ------------------------------------------------ #
    def parse_rounds(
        self, red_name: Optional[str], blue_name: Optional[str], fight_id: str
    ) -> Tuple[List[RoundRow], Dict[str, List[str]]]:
        rounds: Dict[int, RoundRow] = {}
        dbg: Dict[str, List[str]] = {"totals_hdrs": [], "sig_hdrs": []}

        rnorm = _norm_name(red_name)
        bnorm = _norm_name(blue_name)

        # ---- TOTLAS (KD / Sig / Total / TD / Sub / Rev / Ctrl) ----
        totals = self._find_totals_table()
        if totals:
            hdrs = _table_headers(totals)
            dbg["totals_hdrs"] = hdrs
            kd_i   = _find_col(hdrs, "kd")
            sig_i  = _find_col(hdrs, "sig")
            tot_i  = _find_col(hdrs, "total str") if _find_col(hdrs, "total str") != -1 else _find_col(hdrs, "total")
            # TD columns are duplicated as "Td %" twice on some pages. Pick the one that has "X of Y".
            # First, get all columns that look like TD-ish
            td_candidates = [i for i, h in enumerate(hdrs) if re.search(r"\btd\b", h)]
            td_i = -1
            td_pct_i = -1

            # Peek first data block to decide which TD column is counts vs percent
            blocks = _iter_round_blocks(totals)
            first_cells = None
            first_stacked = False
            if blocks:
                _, rows, stacked = blocks[0]
                first_stacked = stacked
                tr = rows[0]
                first_cells = tr.find_all("td") if tr else []

            def _looks_counts(col_idx: int) -> bool:
                if first_cells is None or col_idx < 0 or col_idx >= len(first_cells):
                    return False
                td = first_cells[col_idx]
                # Look for "X of Y" in either top or bottom part (stacked) or whole cell (legacy)
                tops = []
                if first_stacked:
                    tops.extend([_cell_part_text(td, 0) or "", _cell_part_text(td, 1) or ""])
                else:
                    tops.append(_cell_part_text(td, 0) or "")
                return any(_PAIR.search(t) for t in tops)

            for i_cand in td_candidates:
                if _looks_counts(i_cand):
                    td_i = i_cand
                else:
                    td_pct_i = i_cand if td_pct_i == -1 else td_pct_i

            sub_i  = _find_col(hdrs, "sub. att")
            rev_i  = _find_col(hdrs, "rev.")
            ctrl_i = _find_col(hdrs, "ctrl")

            for rnd, pair, stacked in _iter_round_blocks(totals):
                rr = rounds.setdefault(rnd, RoundRow(fight_id=fight_id, round=rnd))
                if stacked:
                    tr = pair[0]
                    tds = tr.find_all("td")
                    # side 0 = top <p>, side 1 = bottom <p>
                    for side_idx, side_key in ((0, "r"), (1, "b")):
                        def cell(i):
                            return _cell_part_text(tds[i], side_idx) if 0 <= i < len(tds) else None

                        if kd_i >= 0: 
                            v = _num(cell(kd_i))
                            if v is not None: setattr(rr, f"{side_key}_kd", v)

                        if sig_i >= 0:
                            l, a = _pair(cell(sig_i))
                            if l is not None: setattr(rr, f"{side_key}_sig_landed", l)
                            if a is not None: setattr(rr, f"{side_key}_sig_attempts", a)

                        if tot_i >= 0:
                            l, a = _pair(cell(tot_i))
                            if l is not None: setattr(rr, f"{side_key}_total_landed", l)
                            if a is not None: setattr(rr, f"{side_key}_total_attempts", a)

                        if td_i >= 0:
                            l, a = _pair(cell(td_i))
                            if l is not None: setattr(rr, f"{side_key}_td", l)
                            if a is not None: setattr(rr, f"{side_key}_td_attempts", a)

                        if sub_i >= 0:
                            v = _num(cell(sub_i))
                            if v is not None: setattr(rr, f"{side_key}_sub_att", v)

                        if rev_i >= 0:
                            v = _num(cell(rev_i))
                            if v is not None: setattr(rr, f"{side_key}_rev", v)

                        if ctrl_i >= 0:
                            sec = _mmss_to_seconds(cell(ctrl_i))
                            if sec is not None: setattr(rr, f"{side_key}_ctrl_sec", sec)
                else:
                    # legacy two-row (one fighter per <tr>)
                    for idx, tr in enumerate(pair):
                        cells = tr.find_all("td")
                        side = _guess_side(idx, tr, rnorm, bnorm)

                        def cell(i):
                            return _cell_part_text(cells[i], 0) if 0 <= i < len(cells) else None

                        if kd_i >= 0:
                            v = _num(cell(kd_i))
                            if v is not None: setattr(rr, f"{side}_kd", v)

                        if sig_i >= 0:
                            l, a = _pair(cell(sig_i))
                            if l is not None: setattr(rr, f"{side}_sig_landed", l)
                            if a is not None: setattr(rr, f"{side}_sig_attempts", a)

                        if tot_i >= 0:
                            l, a = _pair(cell(tot_i))
                            if l is not None: setattr(rr, f"{side}_total_landed", l)
                            if a is not None: setattr(rr, f"{side}_total_attempts", a)

                        if td_i >= 0:
                            l, a = _pair(cell(td_i))
                            if l is not None: setattr(rr, f"{side}_td", l)
                            if a is not None: setattr(rr, f"{side}_td_attempts", a)

                        if sub_i >= 0:
                            v = _num(cell(sub_i))
                            if v is not None: setattr(rr, f"{side}_sub_att", v)

                        if rev_i >= 0:
                            v = _num(cell(rev_i))
                            if v is not None: setattr(rr, f"{side}_rev", v)

                        if ctrl_i >= 0:
                            sec = _mmss_to_seconds(cell(ctrl_i))
                            if sec is not None: setattr(rr, f"{side}_ctrl_sec", sec)

        # ---- SIG by target/position (per-round) ----
        sig = self._find_sig_table()
        if sig:
            hdrs = _table_headers(sig)
            dbg["sig_hdrs"] = hdrs
            idx_map = {
                "head":     next((i for i, h in enumerate(hdrs) if "head" in h), -1),
                "body":     next((i for i, h in enumerate(hdrs) if "body" in h), -1),
                "leg":      next((i for i, h in enumerate(hdrs) if "leg" in h), -1),
                "distance": next((i for i, h in enumerate(hdrs) if "distance" in h), -1),
                "clinch":   next((i for i, h in enumerate(hdrs) if "clinch" in h), -1),
                "ground":   next((i for i, h in enumerate(hdrs) if "ground" in h), -1),
            }

            for rnd, pair, stacked in _iter_round_blocks(sig):
                rr = rounds.setdefault(rnd, RoundRow(fight_id=fight_id, round=rnd))
                if stacked:
                    tr = pair[0]
                    tds = tr.find_all("td")
                    for side_idx, side_key in ((0, "r"), (1, "b")):
                        def cell(i):
                            return _cell_part_text(tds[i], side_idx) if 0 <= i < len(tds) else None
                        for key, ci in idx_map.items():
                            if ci >= 0:
                                landed, attempts = _pair(cell(ci))
                                if landed is not None:   setattr(rr, f"{side_key}_{key}_landed", landed)
                                if attempts is not None: setattr(rr, f"{side_key}_{key}_attempts", attempts)
                else:
                    for idx, tr in enumerate(pair):
                        cells = tr.find_all("td")
                        side = _guess_side(idx, tr, rnorm, bnorm)
                        def cell(i):
                            return _cell_part_text(cells[i], 0) if 0 <= i < len(cells) else None
                        for key, ci in idx_map.items():
                            if ci >= 0:
                                landed, attempts = _pair(cell(ci))
                                if landed is not None:   setattr(rr, f"{side}_{key}_landed", landed)
                                if attempts is not None: setattr(rr, f"{side}_{key}_attempts", attempts)

        # Charts fallback (modern pages without per-round tables)
        if not rounds:
            chart_rows = self._parse_charts_rounds(fight_id)
            return chart_rows, dbg

        return [rounds[k] for k in sorted(rounds.keys())], dbg


# --------------------------------------------------------------------------- #
# Networking + orchestration
# --------------------------------------------------------------------------- #

def _fetch_fight_html(fight_id: str, fight_url: str, referer: Optional[str]) -> Optional[str]:
    """Try HTTP first, then HTTPS. Cache key includes proto."""
    def _fetch(proto: str) -> Optional[str]:
        if not isinstance(fight_url, str) or not fight_url:
            return None
        url = re.sub(r"^https?", proto, fight_url)
        try:
            return http_common.get_html(
                url,
                cache_key=f"{proto}_fight_{fight_id}",
                ttl_hours=720,
                timeout=6,
                headers={"Referer": referer or f"{proto}://www.ufcstats.com/"},
            )
        except Exception:
            return None

    html = _fetch("http")
    if not _looks_like_fight_page(html or ""):
        html = _fetch("https")
    if not _looks_like_fight_page(html or ""):
        return None
    return html


def _dump_debug(fid: str, dbg: Dict[str, List[str]]) -> None:
    """Write out whatever we have so we can inspect failures easily."""
    _ensure_dir(DEBUG_DIR)
    for proto in ("http", "https"):
        src = f"data/raw/html/{proto}_fight_{fid}.html"
        if os.path.exists(src) and os.path.getsize(src) > 0:
            dst = os.path.join(DEBUG_DIR, f"debug_{proto}_fight_{fid}.html")
            try:
                shutil.copyfile(src, dst)
            except Exception:
                pass
    info_path = os.path.join(DEBUG_DIR, f"debug_fight_{fid}.txt")
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            f.write("[totals headers]\n")
            f.write(", ".join(dbg.get("totals_hdrs", [])) + "\n\n")
            f.write("[sig headers]\n")
            f.write(", ".join(dbg.get("sig_hdrs", [])) + "\n")
    except Exception:
        pass


def _normalize_fights_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise SystemExit("fights.csv not found — run event_fights first.")
    out = df.dropna(subset=["fight_id"]).copy()
    out["fight_id"] = out["fight_id"].astype(str).str.strip()

    def _clean_str_col(col: str) -> None:
        if col in out.columns:
            out[col] = out[col].where(out[col].apply(_is_str), other=pd.NA)

    _clean_str_col("fight_url")
    _clean_str_col("event_id")
    return out


def _prime_event_once(eid: str, seen: set[str], lock: Lock) -> None:
    if not eid:
        return
    with lock:
        if eid in seen:
            return
        try:
            _ = http_common.get_html(
                f"http://www.ufcstats.com/event-details/{eid}",
                cache_key=None, ttl_hours=0, timeout=5,
                headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
            )
        except Exception:
            pass
        seen.add(eid)


def _parse_fight_page(
    fight_id: str, fight_url: str, referer: Optional[str]
) -> Tuple[FightDetails, List[RoundRow], Dict[str, List[str]]]:
    html = _fetch_fight_html(fight_id, fight_url, referer)
    if not html:
        return (
            FightDetails(fight_id=fight_id),
            [],
            {"totals_hdrs": [], "sig_hdrs": []},
        )

    doc = soup(html)
    parser = FightPageParser(doc)
    r_id, r_name, b_id, b_name = parser.top_people()
    meta = parser.top_meta()
    details = FightDetails(
        fight_id=fight_id,
        r_fighter_id=r_id, b_fighter_id=b_id,
        r_fighter_name=r_name, b_fighter_name=b_name,
        method=meta.get("method"),
        end_round=_num(meta.get("end_round")) if meta.get("end_round") else None,
        end_time_sec=_mmss_to_seconds(meta.get("end_time")),
        scheduled_rounds=_num(meta.get("scheduled_rounds")) if meta.get("scheduled_rounds") else None,
        is_title=1 if meta.get("is_title") == "1" else 0 if meta.get("is_title") is not None else None,
        judge_scores=meta.get("judge_scores"),
        debug_li=meta.get("debug_li", ""),
    )

    round_rows, dbg = parser.parse_rounds(r_name, b_name, fight_id=fight_id)
    return details, round_rows, dbg

def scrape_round_stats(limit_fights: Optional[int] = None, workers: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fights = _normalize_fights_df(load_csv(FIGHTS_CSV))
    if limit_fights:
        fights = fights.head(limit_fights)

    details_acc: List[FightDetails] = []
    rounds_acc: List[RoundRow] = []
    skipped: List[str] = []

    primed_events: set[str] = set()
    prime_lock = Lock()

    def _job(frow):
        fid = str(getattr(frow, "fight_id"))
        raw_url = getattr(frow, "fight_url", None)
        furl = raw_url if _is_str(raw_url) else f"https://www.ufcstats.com/fight-details/{fid}"
        eid = getattr(frow, "event_id", None)
        eid = eid if _is_str(eid) else None
        referer = f"http://www.ufcstats.com/event-details/{eid}" if eid else "http://www.ufcstats.com/"
        if eid:
            _prime_event_once(eid, primed_events, prime_lock)
        return fid, _parse_fight_page(fid, furl, referer)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_job, frow): idx for idx, frow in enumerate(fights.itertuples(index=False), 1)}
        total = len(futs)
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                fid, (details, rounds, dbg) = fut.result()
                if not details.method and not rounds:
                    skipped.append(fid)
                    _dump_debug(fid, dbg)
                else:
                    details_acc.append(details)
                    rounds_acc.extend(rounds)
            except Exception:
                skipped.append(f"ERR:{i}")
            if i % 25 == 0 or i == total:
                print(f"[rounds] processed {i}/{total} fights…", flush=True)

    df_rounds = pd.DataFrame([asdict(r) for r in rounds_acc])
    # Ensure all expected columns exist, but DO NOT fill with zeros.
    for c in ROUND_COLS:
        if c not in df_rounds.columns:
            df_rounds[c] = pd.NA
    if not df_rounds.empty:
        df_rounds = df_rounds[ROUND_COLS]

    df_details = pd.DataFrame([asdict(d) for d in details_acc], columns=[
        "fight_id",
        "r_fighter_id","b_fighter_id","r_fighter_name","b_fighter_name",
        "method","end_round","end_time_sec","scheduled_rounds","is_title","judge_scores","debug_li"
    ])

    if skipped:
        try:
            pd.Series(skipped, name="fight_id").to_csv("data/curated/_rounds_skipped.csv", index=False)
            print(f"[rounds] skipped {len(skipped)} fights due to fetch/parsing issues → data/curated/_rounds_skipped.csv")
            print(f"[rounds] wrote debug artifacts → {DEBUG_DIR}/debug_*_fight_<id>.html (+ .txt)")
        except Exception:
            pass

    return df_details, df_rounds

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Scrape per-fight details + per-round stats → stats_round.csv and update fights.csv"
    )
    ap.add_argument("--limit-fights", type=int, default=None, help="Process only the first N fights (by current CSV order)")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent workers to parse fights")
    ap.add_argument("--rounds-out", default=ROUNDSTAT_CSV, help="Output CSV for per-round stats")
    ap.add_argument("--fights-out", default=FIGHTS_CSV, help="Upsert back into fights.csv with method/round/time/etc.")
    args = ap.parse_args(argv)

    df_details, df_rounds = scrape_round_stats(limit_fights=args.limit_fights, workers=args.workers)

    if df_details.empty and df_rounds.empty:
        print("[rounds] parsed 0 rows (check fight pages).")
        return 0

    if not df_details.empty:
        upsert_csv(df_details, args.fights_out, keys=["fight_id"])
        update_manifest("fights.csv", rows=len(df_details))
        print(f"[rounds] updated fights.csv details for {len(df_details)} fights")

    if not df_rounds.empty:
        upsert_csv(df_rounds, args.rounds_out, keys=["fight_id", "round"])
        update_manifest("stats_round.csv", rows=len(df_rounds))
        print(f"[rounds] wrote {len(df_rounds)} per-round rows → {args.rounds_out}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())