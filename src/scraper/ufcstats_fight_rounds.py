# src/scraper/ufcstats_fight_rounds.py
from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .common import http as http_common
from .common.parse import soup
from .common.io import load_csv, upsert_csv, update_manifest

FIGHTS_CSV = "data/curated/fights.csv"
ROUNDSTAT_CSV = "data/curated/stats_round.csv"
DEBUG_DIR = "data/debug"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_MMSS = re.compile(r"^(\d{1,2}):(\d{2})$")
_NON_NAME = re.compile(r"[^a-z .'\-]")
_SPACEY = re.compile(r"\s+")
_RD_LAB = re.compile(r"^rd\s*([1-5])\b", re.I)
_PAIR = re.compile(r"(\d+)\s*of\s*(\d+)", re.I)

ROUND_COLS = [
    "fight_id", "round",
    "r_kd", "r_sig_landed", "r_sig_attempts", "r_td", "r_ctrl_sec",
    "r_head_landed", "r_body_landed", "r_leg_landed",
    "r_distance_landed", "r_clinch_landed", "r_ground_landed",
    "b_kd", "b_sig_landed", "b_sig_attempts", "b_td", "b_ctrl_sec",
    "b_head_landed", "b_body_landed", "b_leg_landed",
    "b_distance_landed", "b_clinch_landed", "b_ground_landed",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _mmss_to_seconds(x: Optional[str]) -> Optional[int]:
    if not x:
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
    if x is None:
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _is_str(x: object) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _pair(cell_text: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse '10 of 25' → (10, 25)."""
    if not cell_text:
        return (None, None)
    m = _PAIR.search(cell_text.replace("\xa0", " "))
    if not m:
        return (None, None)
    return int(m.group(1)), int(m.group(2))


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


def _iter_round_blocks(table) -> List[Tuple[int, List]]:
    """Return [(round, [tr_red, tr_blue]), ...] for legacy tables with 'Round N' row separators."""
    tbody = table.find("tbody") or table
    rows = tbody.find_all("tr")
    out: List[Tuple[int, List]] = []
    i = 0
    while i < len(rows):
        txt = rows[i].get_text(" ", strip=True) if rows[i] else ""
        if not txt:
            i += 1
            continue
        m = re.search(r"\bround\s*(\d+)\b", txt, re.I)
        if m:
            rnd = int(m.group(1))
            r_row = rows[i + 1] if i + 1 < len(rows) else None
            b_row = rows[i + 2] if i + 2 < len(rows) else None
            if r_row and b_row:
                out.append((rnd, [r_row, b_row]))
                i += 3
            else:
                i += 1
        else:
            if i + 1 < len(rows):
                out.append((len(out) + 1, [rows[i], rows[i + 1]]))
                i += 2
            else:
                break
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

# --- wide-table cell parsers -------------------------------------------------

def _two_ints(cell: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse 'A B' (e.g., '1 0') into (A, B)."""
    if not cell:
        return None, None
    nums = re.findall(r"\d+", cell)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    if len(nums) == 1:
        return int(nums[0]), None
    return None, None


def _two_times(cell: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse 'mm:ss mm:ss' into (secA, secB)."""
    if not cell:
        return None, None
    parts = re.findall(r"\d{1,2}:\d{2}", cell)
    if len(parts) >= 2:
        return _mmss_to_seconds(parts[0]), _mmss_to_seconds(parts[1])
    if len(parts) == 1:
        s = _mmss_to_seconds(parts[0])
        return s, None
    return None, None


def _two_pairs(cell: str) -> Tuple[Tuple[Optional[int], Optional[int]], Tuple[Optional[int], Optional[int]]]:
    """Parse 'A of B C of D' into ((A,B),(C,D))."""
    if not cell:
        return (None, None), (None, None)
    m = re.findall(r"(\d+)\s*of\s*(\d+)", cell.replace("\xa0", " "), flags=re.I)
    if len(m) >= 2:
        (a, b), (c, d) = m[0], m[1]
        return (int(a), int(b)), (int(c), int(d))
    if len(m) == 1:
        (a, b) = m[0]
        return (int(a), int(b)), (None, None)
    return (None, None), (None, None)


def _parse_chart_panel(panel) -> dict[int, dict[str, tuple[tuple[int, int], tuple[int, int]]]]:
    """
    Parse one chart panel ('Landed by target' or 'Landed by position').
    Returns: { round: { metric: ((r_landed,r_attempts), (b_landed,b_attempts)) } }
    metric ∈ {'head','body','leg'} or {'distance','clinch','ground'}.
    """
    out: dict[int, dict[str, tuple[tuple[int, int], tuple[int, int]]]] = {}
    if not panel:
        return out

    # Each round appears as a ".b-fight-details__charts-col-row"
    for rd_block in panel.select(".b-fight-details__charts-col-row, .b-fight-details__charts-col-row.clearfix"):
        txt = rd_block.get_text(" ", strip=True)
        m = _RD_LAB.search(txt)
        if not m:
            continue
        rnd = int(m.group(1))
        out.setdefault(rnd, {})

        # Within the round, metrics are laid out as small “tables”.
        for table in rd_block.select(".b-fight-details__charts-table"):
            # Find a nearby label indicating metric name
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
    method: Optional[str]
    end_round: Optional[int]
    end_time_sec: Optional[int]
    scheduled_rounds: Optional[int]
    is_title: Optional[int]
    judge_scores: Optional[str]
    debug_li: str = ""


@dataclass(slots=True)
class RoundRow:
    fight_id: str
    round: int
    r_kd: int = 0
    r_sig_landed: int = 0
    r_sig_attempts: int = 0
    r_td: int = 0
    r_ctrl_sec: int = 0
    r_head_landed: int = 0
    r_body_landed: int = 0
    r_leg_landed: int = 0
    r_distance_landed: int = 0
    r_clinch_landed: int = 0
    r_ground_landed: int = 0
    b_kd: int = 0
    b_sig_landed: int = 0
    b_sig_attempts: int = 0
    b_td: int = 0
    b_ctrl_sec: int = 0
    b_head_landed: int = 0
    b_body_landed: int = 0
    b_leg_landed: int = 0
    b_distance_landed: int = 0
    b_clinch_landed: int = 0
    b_ground_landed: int = 0


# --------------------------------------------------------------------------- #
# Fight page parser
# --------------------------------------------------------------------------- #

class FightPageParser:
    def __init__(self, doc) -> None:
        self.doc = doc

    # ---- top section ----------------------------------------------------- #

    def top_names(self) -> Tuple[Optional[str], Optional[str]]:
        people = self.doc.select("div.b-fight-details__person")
        if len(people) >= 2:
            r = people[0].select_one(".b-fight-details__person-name a, .b-fight-details__person__name a")
            b = people[1].select_one(".b-fight-details__person-name a, .b-fight-details__person__name a")
            return _clean(r.get_text(" ", strip=True)) if r else None, _clean(b.get_text(" ", strip=True)) if b else None
        return None, None

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

        # New inline format
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

        # Legacy list format (fallback)
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

        if out["is_title"] is None and self.doc.find(string=re.compile(r"\btitle bout\b", re.I)):
            out["is_title"] = "1"

        return out

    # ---- tables ----------------------------------------------------------- #

    def _find_totals_table(self):
        for tbl in self.doc.select("table"):
            hdrs = _table_headers(tbl)
            joined = " | ".join(hdrs)
            if ("kd" in joined and "sig" in joined):
                return tbl
        for tbl in self.doc.select("table"):
            txt = tbl.get_text(" ", strip=True).lower()
            if "round 1" in txt and ("sig." in txt or "sig str" in txt or "total str" in txt or "kd" in txt):
                return tbl
        return None

    def _find_sig_table(self):
        for tbl in self.doc.select("table"):
            hdrs = _table_headers(tbl)
            has_sig = any("sig" in h for h in hdrs)
            has_target = any(k in hdrs for k in ("head", "body", "leg"))
            has_pos = any(k in hdrs for k in ("distance", "clinch", "ground"))
            if has_sig and (has_target or has_pos):
                return tbl
        for tbl in self.doc.select("table"):
            txt = tbl.get_text(" ", strip=True).lower()
            if "round 1" in txt and ("head" in txt or "distance" in txt):
                return tbl
        return None

    # ---- wide per-round tables (modern layout) ---------------------------- #

    def _is_wide_round_table(self, tbl) -> Optional[int]:
        """Return round number if this table is a per-round wide table (headers include 'round N')."""
        hdrs = _table_headers(tbl)
        joined = " ".join(hdrs)
        m = re.search(r"\bround\s*(\d+)\b", joined, re.I)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _parse_wide_totals_table(self, tbl, rnd: int, rr: RoundRow) -> None:
        """Fill KD, SIG, TD, CTRL from a wide per-round 'TOTALS' table."""
        hdrs = _table_headers(tbl)
        col_idx = {
            "kd": _find_col(hdrs, "kd"),
            "sig": _find_col(hdrs, "sig"),
            "td": _find_col(hdrs, "td"),
            "ctrl": _find_col(hdrs, "ctrl"),
        }
        rows = tbl.select("tbody tr") or tbl.select("tr")
        if not rows:
            return
        tds = [td.get_text(" ", strip=True) for td in rows[0].select("td")]
        # KD
        i = col_idx["kd"]
        if 0 <= i < len(tds):
            ra, ba = _two_ints(tds[i])
            if ra is not None: rr.r_kd = ra
            if ba is not None: rr.b_kd = ba
        # SIG. STR.
        i = col_idx["sig"]
        if 0 <= i < len(tds):
            (rl, ra), (bl, ba) = _two_pairs(tds[i])
            if rl is not None: rr.r_sig_landed = rl
            if ra is not None: rr.r_sig_attempts = ra
            if bl is not None: rr.b_sig_landed = bl
            if ba is not None: rr.b_sig_attempts = ba
        # TD (landed)
        i = col_idx["td"]
        if 0 <= i < len(tds):
            (rl, _ra), (bl, _ba) = _two_pairs(tds[i])
            if rl is not None: rr.r_td = rl
            if bl is not None: rr.b_td = bl
        # CTRL
        i = col_idx["ctrl"]
        if 0 <= i < len(tds):
            rs, bs = _two_times(tds[i])
            if rs is not None: rr.r_ctrl_sec = rs
            if bs is not None: rr.b_ctrl_sec = bs

    def _parse_wide_sig_table(self, tbl, rnd: int, rr: RoundRow) -> None:
        """Fill head/body/leg/distance/clinch/ground landed from a wide per-round SIG table."""
        hdrs = _table_headers(tbl)
        idx_map = {
            "head": next((i for i, h in enumerate(hdrs) if "head" in h), -1),
            "body": next((i for i, h in enumerate(hdrs) if "body" in h), -1),
            "leg": next((i for i, h in enumerate(hdrs) if "leg" in h), -1),
            "distance": next((i for i, h in enumerate(hdrs) if "distance" in h), -1),
            "clinch": next((i for i, h in enumerate(hdrs) if "clinch" in h), -1),
            "ground": next((i for i, h in enumerate(hdrs) if "ground" in h), -1),
            "sig": next((i for i, h in enumerate(hdrs) if "sig" in h), -1),
        }
        rows = tbl.select("tbody tr") or tbl.select("tr")
        if not rows:
            return
        tds = [td.get_text(" ", strip=True) for td in rows[0].select("td")]
        for key, i in idx_map.items():
            if i < 0 or i >= len(tds):
                continue
            (rl, ra), (bl, ba) = _two_pairs(tds[i])
            if key == "sig":
                if rl is not None: rr.r_sig_landed = rl
                if ra is not None: rr.r_sig_attempts = ra
                if bl is not None: rr.b_sig_landed = bl
                if ba is not None: rr.b_sig_attempts = ba
            else:
                if rl is not None: setattr(rr, f"r_{key}_landed", rl)
                if bl is not None: setattr(rr, f"b_{key}_landed", bl)

    # ---- charts (new layout) --------------------------------------------- #

    def _parse_charts_rounds(self, rnorm: str, bnorm: str, fight_id: str) -> List[RoundRow]:
        """
        Fallback parser for the new UFCStats chart layout (no per-round tables).
        Fills head/body/leg and distance/clinch/ground landed per round.
        """
        def _panel_with(title_sub: str):
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
                    (r_la, _r_at), (b_la, _b_at) = pair
                    setattr(rr, f"r_{metric}_landed", r_la)
                    setattr(rr, f"b_{metric}_landed", b_la)

            pos = by_pos.get(rnd, {})
            for metric in ("distance", "clinch", "ground"):
                pair = pos.get(metric)
                if pair:
                    (r_la, _r_at), (b_la, _b_at) = pair
                    setattr(rr, f"r_{metric}_landed", r_la)
                    setattr(rr, f"b_{metric}_landed", b_la)

        return [rows[k] for k in sorted(rows.keys())]

    # ---- per-round parsing ------------------------------------------------ #

    def parse_rounds(
        self, red_name: Optional[str], blue_name: Optional[str], fight_id: str
    ) -> Tuple[List[RoundRow], Dict[str, List[str]]]:
        rounds: Dict[int, RoundRow] = {}
        dbg: Dict[str, List[str]] = {"totals_hdrs": [], "sig_hdrs": []}

        # Normalized names (kept for legacy side-detection; not used by wide tables)
        rnorm = _norm_name(red_name)
        bnorm = _norm_name(blue_name)

        # (A) Prefer WIDE per-round tables if present
        wide_totals = []
        wide_sig = []
        for tbl in self.doc.select("table"):
            rnd = self._is_wide_round_table(tbl)
            if not rnd:
                continue
            hdrs = _table_headers(tbl)
            j = " ".join(hdrs)
            if ("kd" in j and "sig" in j):
                wide_totals.append((rnd, tbl, hdrs))
            elif any(k in j for k in ("head", "body", "leg", "distance", "clinch", "ground")):
                wide_sig.append((rnd, tbl, hdrs))

        if wide_totals or wide_sig:
            if wide_totals:
                dbg["totals_hdrs"] = wide_totals[0][2]
            if wide_sig:
                dbg["sig_hdrs"] = wide_sig[0][2]

            for rnd, tbl, _ in sorted(wide_totals, key=lambda x: x[0]):
                rr = rounds.setdefault(rnd, RoundRow(fight_id=fight_id, round=rnd))
                self._parse_wide_totals_table(tbl, rnd, rr)
            for rnd, tbl, _ in sorted(wide_sig, key=lambda x: x[0]):
                rr = rounds.setdefault(rnd, RoundRow(fight_id=fight_id, round=rnd))
                self._parse_wide_sig_table(tbl, rnd, rr)

            return [rounds[k] for k in sorted(rounds.keys())], dbg

        # (B) Legacy row-pair tables (older pages)
        totals = self._find_totals_table()
        if totals:
            hdrs = _table_headers(totals)
            dbg["totals_hdrs"] = hdrs
            kd_i = _find_col(hdrs, "kd")
            sig_i = _find_col(hdrs, "sig")
            td_i = _find_col(hdrs, "td")
            ctrl_i = _find_col(hdrs, "ctrl")

            for rnd, pair in _iter_round_blocks(totals):
                rr = rounds.setdefault(rnd, RoundRow(fight_id=fight_id, round=rnd))
                for idx, tr in enumerate(pair):
                    cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
                    side = _guess_side(idx, tr, rnorm, bnorm)

                    if 0 <= kd_i < len(cells):
                        v = _num(cells[kd_i])
                        if v is not None:
                            setattr(rr, f"{side}_kd", v)

                    if 0 <= sig_i < len(cells):
                        l, a = _pair(cells[sig_i])
                        if l is not None:
                            setattr(rr, f"{side}_sig_landed", l or 0)
                        if a is not None:
                            setattr(rr, f"{side}_sig_attempts", a or 0)

                    if 0 <= td_i < len(cells):
                        l, _ = _pair(cells[td_i])
                        if l is not None:
                            setattr(rr, f"{side}_td", l or 0)

                    if 0 <= ctrl_i < len(cells):
                        sec = _mmss_to_seconds(cells[ctrl_i])
                        if sec is not None:
                            setattr(rr, f"{side}_ctrl_sec", sec)

        sig = self._find_sig_table()
        if sig:
            hdrs = _table_headers(sig)
            dbg["sig_hdrs"] = hdrs
            idx_map = {
                "head": next((i for i, h in enumerate(hdrs) if "head" in h), -1),
                "body": next((i for i, h in enumerate(hdrs) if "body" in h), -1),
                "leg": next((i for i, h in enumerate(hdrs) if "leg" in h), -1),
                "distance": next((i for i, h in enumerate(hdrs) if "distance" in h), -1),
                "clinch": next((i for i, h in enumerate(hdrs) if "clinch" in h), -1),
                "ground": next((i for i, h in enumerate(hdrs) if "ground" in h), -1),
            }

            for rnd, pair in _iter_round_blocks(sig):
                rr = rounds.setdefault(rnd, RoundRow(fight_id=fight_id, round=rnd))
                for idx, tr in enumerate(pair):
                    cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
                    side = _guess_side(idx, tr, rnorm, bnorm)
                    for key, ci in idx_map.items():
                        if 0 <= ci < len(cells):
                            landed, _ = _pair(cells[ci])
                            if landed is not None:
                                setattr(rr, f"{side}_{key}_landed", landed or 0)

        # (C) Charts fallback (when there are no per-round tables at all)
        if not rounds:
            chart_rows = self._parse_charts_rounds(rnorm, bnorm, fight_id)
            return chart_rows, dbg

        return [rounds[k] for k in sorted(rounds.keys())], dbg


# --------------------------------------------------------------------------- #
# Networking + orchestration
# --------------------------------------------------------------------------- #

def _fetch_fight_html(fight_id: str, fight_url: str, referer: Optional[str]) -> Optional[str]:
    """Try HTTP first, then HTTPS. Cache key includes proto."""
    time.sleep(0.15)  # courtesy delay

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


def _prime_event_once(eid: str, seen: set[str]) -> None:
    if not eid or eid in seen:
        return
    try:
        _ = http_common.get_html(
            f"http://www.ufcstats.com/event-details/{eid}",
            cache_key=None, ttl_hours=0, timeout=6,
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
        )
        seen.add(eid)
        time.sleep(0.15)
    except Exception:
        pass


def _parse_fight_page(
    fight_id: str, fight_url: str, referer: Optional[str]
) -> Tuple[FightDetails, List[RoundRow], Dict[str, List[str]]]:
    html = _fetch_fight_html(fight_id, fight_url, referer)
    if not html:
        return (
            FightDetails(
                fight_id=fight_id,
                method=None,
                end_round=None,
                end_time_sec=None,
                scheduled_rounds=None,
                is_title=None,
                judge_scores=None,
                debug_li="",
            ),
            [],
            {"totals_hdrs": [], "sig_hdrs": []},
        )

    doc = soup(html)
    parser = FightPageParser(doc)
    r_name, b_name = parser.top_names()
    meta = parser.top_meta()
    details = FightDetails(
        fight_id=fight_id,
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


def scrape_round_stats(limit_fights: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fights = _normalize_fights_df(load_csv(FIGHTS_CSV))
    if limit_fights:
        fights = fights.head(limit_fights)

    details_acc: List[FightDetails] = []
    rounds_acc: List[RoundRow] = []
    skipped: List[str] = []
    primed_events: set[str] = set()

    for i, f in enumerate(fights.itertuples(index=False), 1):
        fid = str(getattr(f, "fight_id"))
        raw_url = getattr(f, "fight_url", None)
        furl = raw_url if _is_str(raw_url) else f"https://www.ufcstats.com/fight-details/{fid}"

        eid = getattr(f, "event_id", None)
        eid = eid if _is_str(eid) else None
        referer = f"http://www.ufcstats.com/event-details/{eid}" if eid else "http://www.ufcstats.com/"

        _prime_event_once(eid or "", primed_events)

        details, rounds, dbg = _parse_fight_page(fid, furl, referer)
        if not details.method and not rounds:
            skipped.append(fid)
            _dump_debug(fid, dbg)
        else:
            details_acc.append(details)
            rounds_acc.extend(rounds)

        if i % 25 == 0:
            print(f"[rounds] processed {i}/{len(fights)} fights…", flush=True)
        time.sleep(0.1)  # polite pacing

    # DataFrames
    df_rounds = pd.DataFrame([asdict(r) for r in rounds_acc])
    for c in ROUND_COLS:
        if c not in df_rounds.columns:
            df_rounds[c] = 0
    if not df_rounds.empty:
        df_rounds = df_rounds[ROUND_COLS]

    df_details = pd.DataFrame([asdict(d) for d in details_acc], columns=[
        "fight_id", "method", "end_round", "end_time_sec", "scheduled_rounds", "is_title", "judge_scores", "debug_li"
    ])

    if skipped:
        pd.Series(skipped, name="fight_id").to_csv("data/curated/_rounds_skipped.csv", index=False)
        print(f"[rounds] skipped {len(skipped)} fights due to fetch/parsing issues → data/curated/_rounds_skipped.csv")
        print(f"[rounds] wrote debug artifacts → {DEBUG_DIR}/debug_*_fight_<id>.html (+ .txt)")

    return df_details, df_rounds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Scrape per-fight details + per-round stats → stats_round.csv and update fights.csv"
    )
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
        upsert_csv(df_rounds, args.rounds_out, keys=["fight_id", "round"])
        update_manifest("stats_round.csv", rows=len(df_rounds))
        print(f"[rounds] wrote {len(df_rounds)} per-round rows → {args.rounds_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())