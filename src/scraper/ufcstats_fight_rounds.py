# src/scraper/ufcstats_fight_rounds.py
from __future__ import annotations

import argparse
import inspect
import os
import re
import time, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer

from .common import http as http_common
from .common.io import upsert_csv as _io_upsert_csv
try:
    from .common.io import update_manifest as _io_update_manifest
except Exception:
    # Some repos don't expose update_manifest; noop fallback keeps scraper from crashing
    def _io_update_manifest(*args, **kwargs):
        return None

# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #

DEFAULT_FIGHTS_CSV = "data/curated/fights.csv"
DEFAULT_ROUNDSTAT_CSV = "data/curated/stats_round.csv"
DEFAULT_EVENTS_CSV = "data/curated/events.csv"
DEFAULT_MANIFEST_PATH = "data/manifest.json"

# polite baseline (can be overridden via CLI)
http_common._cfg.rate_per_sec = 4.0

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

PAIR_RE = re.compile(r"(\d+)\s*of\s*(\d+)")
ROUND_RE = re.compile(r"Round\s+(\d+)", re.I)
EVENT_ID_RE = re.compile(r"/event-details/([a-f0-9]{16})", re.I)
FIGHT_ID_RE = re.compile(r"/fight-details/([a-f0-9]{16})", re.I)
FIGHTER_ID_RE = re.compile(r"/fighter-details/([a-f0-9]{16})", re.I)
MMSS_RE = re.compile(r"^(\d{1,2}):(\d{2})$")

def _retry_get_html(cache_key: str, url: str, *, max_attempts=6, base_sleep=0.6):
    """
    Retry helper that backs off aggressively on HTTP 429 or 'Too Many Requests'.
    base_sleep ~0.6s → worst case ~0.6 * (2^5) ≈ 19s on final attempt.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return http_common.get_html(url, cache_key=cache_key)
        except Exception as e:
            msg = str(e)
            # treat 429 or 'Too Many Requests' as backoff-able
            is_429 = ("429" in msg) or ("Too Many Requests" in msg)
            if attempt < max_attempts and is_429:
                sleep = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, base_sleep)
                time.sleep(sleep)
                continue
            # one soft retry for other transient errors
            if attempt < min(3, max_attempts) and any(s in msg for s in ("timed out", "timeout", "RST", "Temporary")):
                sleep = 0.4 + random.uniform(0, 0.4)
                time.sleep(sleep)
                continue
            raise
def _local_upsert_csv(path: str, rows: list, keys: list, field_order: list | None = None):
    """Local pandas-based upsert used as last resort and for very old upsert_csv APIs."""
    df_new = pd.DataFrame(rows)
    # enforce column order if requested
    if field_order:
        for c in field_order:
            if c not in df_new.columns:
                df_new[c] = pd.NA
        df_new = df_new[field_order]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        df_new.to_csv(path, index=False)
        return

    df_old = pd.read_csv(path)

    # make column union
    for c in df_new.columns:
        if c not in df_old.columns:
            df_old[c] = pd.NA
    for c in df_old.columns:
        if c not in df_new.columns:
            df_new[c] = pd.NA

    # upsert on composite key
    def _key_tuple(df: pd.DataFrame) -> pd.Series:
        return df[keys].astype(str).agg("||".join, axis=1)

    df_old["_k"] = _key_tuple(df_old)
    df_new["_k"] = _key_tuple(df_new)

    old_keep = df_old[~df_old["_k"].isin(df_new["_k"])].drop(columns=["_k"])
    merged = pd.concat([old_keep, df_new.drop(columns=["_k"])], ignore_index=True)
    merged.to_csv(path, index=False)


def _read_existing_keys(path: str, keys: list[str]) -> set[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    try:
        df = pd.read_csv(path, usecols=keys)
    except Exception:
        df = pd.read_csv(path)
        df = df[keys]
    return set(df[keys].astype(str).agg("||".join, axis=1).tolist())


def _key_tuple_from_rows(rows: list[dict], keys: list[str]) -> set[str]:
    if not rows:
        return set()
    df = pd.DataFrame(rows)
    for k in keys:
        if k not in df.columns:
            df[k] = pd.NA
    return set(df[keys].astype(str).agg("||".join, axis=1).tolist())


def _upsert_csv_compat(path: str, rows: list, keys: list, field_order: list | None = None):
    """
    Try repo upsert_csv with various historical signatures.
    Fallback to a local pandas-based upsert if none match.

    Supported shapes seen in the wild:
      1) upsert_csv(path, rows, key_fields=[...], field_order=[...])
      2) upsert_csv(path, rows, keys=[...], field_order=[...])
      3) upsert_csv(path, rows, keys)
      4) upsert_csv(path, rows)  → then we must do local to respect keys/order
    """
    # try keyword forms
    try:
        return _io_upsert_csv(path, rows, key_fields=keys, field_order=field_order)  # new
    except TypeError:
        pass
    try:
        return _io_upsert_csv(path, rows, keys=keys, field_order=field_order)       # mid
    except TypeError:
        pass

    # inspect positional arity
    try:
        sig = inspect.signature(_io_upsert_csv)
        n = len(sig.parameters)
        if n >= 3:
            # assume old: (path, rows, keys)
            return _io_upsert_csv(path, rows, keys)
        elif n == 2:
            # very old: no keys support → do local upsert to respect keys & order
            return _local_upsert_csv(path, rows, keys, field_order)
    except Exception:
        pass

    # last resort
    return _local_upsert_csv(path, rows, keys, field_order)


def _update_manifest_compat(payload: dict, default_path: str = DEFAULT_MANIFEST_PATH):
    """
    Handle both:
      - update_manifest(payload_dict)
      - update_manifest(path, payload_dict)
    If neither signature matches, silently no-op.
    """
    try:
        return _io_update_manifest(payload)  # newer style
    except TypeError:
        pass
    try:
        return _io_update_manifest(default_path, payload)  # older style
    except TypeError:
        return None


def _parse_pair(text: str) -> Tuple[Optional[int], Optional[int]]:
    m = PAIR_RE.search(text or "")
    if not m:
        return (None, None)
    return (int(m.group(1)), int(m.group(2)))


def _parse_pct(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.strip().replace("%", "")
    if not t or t.startswith("-"):
        return None
    try:
        return float(t)
    except Exception:
        return None


def _time_to_sec(text: str) -> int:
    if not text:
        return 0
    t = text.strip()
    if not t or t.startswith("-"):
        return 0
    if ":" not in t:
        try:
            return int(t)
        except Exception:
            return 0
    mm, ss = t.split(":", 1)
    try:
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0


def _mmss_to_seconds(x: Optional[str]) -> Optional[int]:
    if not x:
        return None
    t = x.strip()
    if not t or t in ("---", "—", "–"):
        return None
    m = MMSS_RE.match(t)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def _id_from_href(href: str, kind: str) -> Optional[str]:
    if not href:
        return None
    rx = {"event": EVENT_ID_RE, "fight": FIGHT_ID_RE, "fighter": FIGHTER_ID_RE}.get(kind)
    if not rx:
        return None
    m = rx.search(href)
    return m.group(1) if m else None


def _norm_text(x) -> str:
    return " ".join((x or "").split())


# --------- FAST SOUP ----------
# Parse only what we need to cut parse time dramatically.
_STRAINER_EVENT = SoupStrainer(["table", "tr", "td", "a"])
_STRAINER_FIGHT = SoupStrainer([
    "h2", "section", "table", "thead", "tbody", "tr", "td", "i", "p", "a", "div", "span"
])


def _safe_soup(html: str, fight_page: bool = False):
    """Prefer lxml (fast). Fall back to builtin if lxml missing."""
    try:
        return BeautifulSoup(html, "lxml", parse_only=_STRAINER_FIGHT if fight_page else _STRAINER_EVENT)
    except Exception:
        return BeautifulSoup(html, "html.parser", parse_only=_STRAINER_FIGHT if fight_page else _STRAINER_EVENT)


# --------- Legacy-inspired fast round iterator ----------
def _iter_round_blocks(table) -> List[Tuple[int, List, bool]]:
    """
    Yield (round_number, rows, is_stacked) where 'rows' is either [tr] (stacked two-<p> style)
    or [tr_red, tr_blue] legacy two-row style.
    """
    nodes = [n for n in table.descendants if getattr(n, "name", None) in ("thead", "tr")]
    out: List[Tuple[int, List, bool]] = []

    def _first_data_tr(start_idx: int):
        j = start_idx + 1
        while j < len(nodes):
            nd = nodes[j]
            if getattr(nd, "name", None) == "tr" and nd.find_all("td"):
                return j, nd
            j += 1
        return None, None

    i = 0
    while i < len(nodes):
        node = nodes[i]
        txt = node.get_text(" ", strip=True) if node else ""
        m = re.search(r"\bround\s*(\d+)\b", txt, re.I)
        if not m:
            i += 1
            continue

        rnd = int(m.group(1))
        j, tr1 = _first_data_tr(i)
        if tr1 is None:
            i += 1
            continue

        tds1 = tr1.find_all("td")
        cells_with_two_p = sum(1 for td in tds1 if len(td.select("p.b-fight-details__table-text")) >= 2)
        is_stacked = cells_with_two_p >= 2

        if is_stacked:
            out.append((rnd, [tr1], True))
            i = j + 1
            continue

        k, tr2 = _first_data_tr(j)
        if tr2 is not None:
            out.append((rnd, [tr1, tr2], False))
            i = (k or j) + 1
        else:
            out.append((rnd, [tr1], False))
            i = j + 1

    return out


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class FightMeta:
    fight_id: str
    event_id: str
    r_fighter_id: Optional[str]
    b_fighter_id: Optional[str]
    r_fighter_name: Optional[str]
    b_fighter_name: Optional[str]
    weight_class: Optional[str]
    winner_corner: Optional[str]  # "R","B","D","NC"
    fight_url: str
    scheduled_rounds: Optional[int]
    method: Optional[str]
    end_round: Optional[int]
    end_time_sec: Optional[int]
    is_title: int
    judge_scores: str
    debug_li: str  # stash referee here for lightweight debug


@dataclass
class RoundRow:
    fight_id: str
    event_id: str
    round: int
    fighter_id: str
    fighter_corner: Optional[str]  # R/B if mapped, else None
    kd: Optional[int]
    sig_landed: Optional[int]
    sig_attempted: Optional[int]
    sig_pct: Optional[float]
    tot_landed: Optional[int]
    tot_attempted: Optional[int]
    td_landed: Optional[int]
    td_attempted: Optional[int]
    td_pct: Optional[float]
    sub_att: Optional[int]
    rev: Optional[int]
    ctrl_time_sec: Optional[int]
    head_landed: Optional[int]
    head_attempted: Optional[int]
    body_landed: Optional[int]
    body_attempted: Optional[int]
    leg_landed: Optional[int]
    leg_attempted: Optional[int]
    distance_landed: Optional[int]
    distance_attempted: Optional[int]
    clinch_landed: Optional[int]
    clinch_attempted: Optional[int]
    ground_landed: Optional[int]
    ground_attempted: Optional[int]


# --------------------------------------------------------------------------- #
# Event listing                                                               #
# --------------------------------------------------------------------------- #

def fetch_event_fights(event_id: str) -> List[Dict]:
    """
    Return list of fights for an event with:
      dict(fight_url, fight_id, event_id, r_fighter_id, b_fighter_id, r_fighter_name, b_fighter_name)
    Uses event-details page which *does* label red/blue by column order.
    """
    url = f"http://www.ufcstats.com/event-details/{event_id}"
    html = _retry_get_html(cache_key=f"event_{event_id}", url=url)
    doc = _safe_soup(html, fight_page=False)

    fights: List[Dict] = []
    for tr in doc.find_all("tr", class_="b-fight-details__table-row"):
        fight_link = tr.select_one('a[href*="/fight-details/"]')
        if not fight_link:
            continue
        fight_url = (fight_link.get("href") or "").strip()
        fight_id = _id_from_href(fight_url, "fight")

        fighter_links = tr.select('a[href*="/fighter-details/"]')
        if len(fighter_links) < 2:
            # fallback: any anchors in row
            fighter_links = [a for a in tr.find_all("a") if "/fighter-details/" in (a.get("href") or "")]
        if len(fighter_links) >= 2:
            # event listing: first = RED, second = BLUE
            r_a, b_a = fighter_links[0], fighter_links[1]
            r_id = _id_from_href(r_a.get("href"), "fighter")
            b_id = _id_from_href(b_a.get("href"), "fighter")
            r_name = _norm_text(r_a.get_text(" ", strip=True))
            b_name = _norm_text(b_a.get_text(" ", strip=True))
        else:
            r_id = b_id = r_name = b_name = None

        fights.append(
            dict(
                fight_url=fight_url,
                fight_id=fight_id,
                event_id=event_id,
                r_fighter_id=r_id,
                b_fighter_id=b_id,
                r_fighter_name=r_name,
                b_fighter_name=b_name,
            )
        )

    # dedup while preserving order
    seen = set()
    out: List[Dict] = []
    for f in fights:
        fid = f.get("fight_id")
        if fid and fid not in seen:
            seen.add(fid)
            out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Fight-details parsing                                                        #
# --------------------------------------------------------------------------- #

def parse_fight_details(fight_url: str, seed: Optional[Dict] = None) -> Tuple[FightMeta, List[RoundRow]]:
    fight_id = _id_from_href(fight_url, "fight") or ""
    html = _retry_get_html(cache_key=f"fight_{fight_id}", url=fight_url)
    doc = _safe_soup(html, fight_page=True)

    # ---------------- Basic page context ----------------
    # Event ID
    event_a = doc.select_one('h2.b-content__title a[href*="/event-details/"]')
    event_id = _id_from_href(event_a.get("href") if event_a else "", "event") or (seed.get("event_id") if seed else "")

    # Header persons (collect ids + detect outcome badges)
    persons = doc.select(".b-fight-details__persons .b-fight-details__person")
    winner_fighter_id: Optional[str] = None
    saw_draw_badge = False
    saw_nc_badge = False
    header_fighter_ids: List[str] = []
    header_names: List[str] = []
    for p in persons:
        a = p.select_one("a.b-fight-details__person-link")
        fid = _id_from_href(a.get("href") if a else "", "fighter")
        name = _norm_text(a.get_text(" ", strip=True) if a else "")
        if fid:
            header_fighter_ids.append(fid)
            header_names.append(name)

        # Status badge text can be "W", "L", "D", "NC" (gray)
        status_txt = _norm_text((p.select_one(".b-fight-details__person-status") or p).get_text(" ", strip=True)).upper()
        if status_txt.startswith("W"):
            winner_fighter_id = fid
        elif status_txt == "D":
            saw_draw_badge = True
        elif status_txt == "NC":
            saw_nc_badge = True

    # Bout title / weight class
    title_i = doc.select_one(".b-fight-details__fight-head .b-fight-details__fight-title")
    weight_class = None
    is_title = 0
    if title_i:
        t = _norm_text(title_i.get_text(" ", strip=True))
        weight_class = t.replace("Bout", "").strip()
        is_title = 1 if "Title" in t else 0

    # Meta paragraph (method / round / time / time format / referee)
    method = end_round = scheduled_rounds = None
    end_time_sec = None
    referee = None
    meta_p = doc.select_one(".b-fight-details__content p.b-fight-details__text")
    if meta_p:
        for lab in meta_p.select("i.b-fight-details__label"):
            key = _norm_text(lab.get_text(" ", strip=True)).rstrip(":").lower()
            val = lab.find_next_sibling()
            text_val = _norm_text(val.get_text(" ", strip=True) if val else lab.next_sibling or "")
            if key == "method":
                method = text_val
            elif key == "round":
                try:
                    end_round = int(text_val)
                except Exception:
                    end_round = None
            elif key == "time":
                end_time_sec = _time_to_sec(text_val)
            elif key in ("time format", "format"):
                m = re.search(r"(\d+)\s*Rnd", text_val)
                scheduled_rounds = int(m.group(1)) if m else None
            elif key == "referee":
                referee = text_val

    # Details paragraph (judges or finish notes)
    judge_scores = ""
    details_p = meta_p.find_next_sibling("p", class_="b-fight-details__text") if meta_p else None
    if details_p:
        detail_text = _norm_text(details_p.get_text(" ", strip=True))
        judges = re.findall(r"([A-Za-z .’'`-]+?\s+\d+\s*-\s*\d+)", detail_text)
        judge_scores = " ".join(j.strip().rstrip(".") for j in judges) if judges else detail_text

    # ---------------- Determine RED/BLUE from the FIGHT PAGE ----------------
    r_id = seed.get("r_fighter_id") if seed else None
    b_id = seed.get("b_fighter_id") if seed else None
    r_name = seed.get("r_fighter_name") if seed else None
    b_name = seed.get("b_fighter_name") if seed else None

    corner_map: Dict[str, str] = {}

    totals_table = None
    for t in doc.find_all("table", class_="b-fight-details__table"):
        head_txt = " ".join(th.get_text(" ", strip=True) for th in t.select("thead tr th"))
        if ("Total str." in head_txt or "Total" in head_txt) and "KD" in head_txt and "Ctrl" in head_txt:
            low = t.get_text(" ", strip=True).lower()
            if "round 1" in low:
                totals_table = t
                break

    def _derive_corner_map_from_totals(tbl) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        if not tbl:
            return None, None, None, None
        for current_round, rows_pack, is_stacked in _iter_round_blocks(tbl):
            if not rows_pack:
                continue
            tr = rows_pack[0]
            tds = tr.find_all("td")
            if not tds:
                continue
            f_as = tds[0].select('a[href*="/fighter-details/"]')
            two_ids = [_id_from_href(a.get("href"), "fighter") for a in f_as][:2]
            two_names = [_norm_text(a.get_text(" ", strip=True)) for a in f_as][:2]
            if len(two_ids) == 2:
                # Top stack = RED, bottom = BLUE
                return two_ids[0], two_ids[1], (two_names[0] if two_names else None), (two_names[1] if len(two_names) > 1 else None)
        return None, None, None, None

    r_id_fp, b_id_fp, r_name_fp, b_name_fp = _derive_corner_map_from_totals(totals_table)
    if r_id_fp:
        r_id = r_id_fp; corner_map[r_id] = "R"
    if b_id_fp:
        b_id = b_id_fp; corner_map[b_id] = "B"
    if r_name_fp: r_name = r_name_fp
    if b_name_fp: b_name = b_name_fp

    # Fallback: header order (left then right)
    if not corner_map and len(header_fighter_ids) >= 2:
        corner_map[header_fighter_ids[0]] = "R"
        corner_map[header_fighter_ids[1]] = "B"
        if not r_id: r_id = header_fighter_ids[0]
        if not b_id: b_id = header_fighter_ids[1]
        if len(header_names) >= 2:
            if not r_name: r_name = header_names[0]
            if not b_name: b_name = header_names[1]

    # ---------------- Winner / Draw / NC ------------------------------------
    # Prefer badge: W / D / NC. If no W badge, infer from text.
    winner_corner: Optional[str] = None
    if winner_fighter_id:
        winner_corner = corner_map.get(winner_fighter_id)
    else:
        text_blob = " ".join([s for s in [method or "", judge_scores or ""] if s]).lower()
        # Normalize common wordings
        is_draw_text = any(
            kw in text_blob
            for kw in (
                "split draw", "majority draw", "unanimous draw", "draw",
            )
        )
        is_nc_text = any(
            kw in text_blob
            for kw in (
                "no contest", " nc", "nc ", "(nc)", "overturned", "overturn", "over ruled", "over-ruled",
            )
        )
        if saw_draw_badge or is_draw_text:
            winner_corner = "D"
        elif saw_nc_badge or is_nc_text:
            winner_corner = "NC"
        else:
            winner_corner = None  # unknown (rare)

    # ---------------- Assemble meta ----------------
    meta = FightMeta(
        fight_id=fight_id,
        event_id=event_id,
        r_fighter_id=r_id,
        b_fighter_id=b_id,
        r_fighter_name=r_name,
        b_fighter_name=b_name,
        weight_class=weight_class,
        winner_corner=winner_corner,
        fight_url=fight_url,
        scheduled_rounds=scheduled_rounds,
        method=method,
        end_round=end_round,
        end_time_sec=end_time_sec,
        is_title=is_title,
        judge_scores=judge_scores,
        debug_li=referee or "",
    )

    # ---------------- Per-round parsing (unchanged) ----------------
    # A) Totals per round
    totals_rounds: Dict[int, Dict[str, Dict[str, Optional[int]]]] = {}
    if totals_table:
        for current_round, rows_pack, is_stacked in _iter_round_blocks(totals_table):
            if not rows_pack:
                continue
            if is_stacked:
                el = rows_pack[0]
                columns = el.find_all("td")

                def cell_pairs(idx) -> Tuple[str, str]:
                    ps = columns[idx].select("p")
                    return (_norm_text(ps[0].get_text()) if len(ps) > 0 else "",
                            _norm_text(ps[1].get_text()) if len(ps) > 1 else "")

                f_anchors = columns[0].select('a[href*="/fighter-details/"]')
                two_ids = [_id_from_href(a.get("href"), "fighter") for a in f_anchors][:2]
                fighter_ids = two_ids if len(two_ids) == 2 else header_fighter_ids[:2]

                kd_a, kd_b = cell_pairs(1)
                sig_a, sig_b = cell_pairs(2)
                sigp_a, sigp_b = cell_pairs(3)
                tot_a, tot_b = cell_pairs(4)
                td_a, td_b = cell_pairs(5)
                tdp_a, tdp_b = cell_pairs(6)
                sub_a, sub_b = cell_pairs(7)
                rev_a, rev_b = cell_pairs(8)
                ctrl_a, ctrl_b = cell_pairs(9)

                pairs = [
                    (fighter_ids[0], kd_a, sig_a, sigp_a, tot_a, td_a, tdp_a, sub_a, rev_a, ctrl_a),
                    (fighter_ids[1], kd_b, sig_b, sigp_b, tot_b, td_b, tdp_b, sub_b, rev_b, ctrl_b),
                ]
                for fid, kd_t, sig_t, sigp_t, tot_t, td_t, tdp_t, sub_t, rev_t, ctrl_t in pairs:
                    if not fid:
                        continue
                    landed_sig, att_sig = _parse_pair(sig_t)
                    landed_tot, att_tot = _parse_pair(tot_t)
                    td_l, td_a2 = _parse_pair(td_t)
                    totals_rounds.setdefault(current_round, {})[fid] = dict(
                        kd=int(kd_t) if kd_t and kd_t.isdigit() else 0,
                        sig_landed=landed_sig, sig_attempted=att_sig, sig_pct=_parse_pct(sigp_t),
                        tot_landed=landed_tot, tot_attempted=att_tot,
                        td_landed=td_l, td_attempted=td_a2, td_pct=_parse_pct(tdp_t),
                        sub_att=int(sub_t) if sub_t and sub_t.isdigit() else 0,
                        rev=int(rev_t) if rev_t and rev_t.isdigit() else 0,
                        ctrl_time_sec=_time_to_sec(ctrl_t),
                    )
            else:
                el = rows_pack[0]
                columns = el.find_all("td")

                def cell_pairs(idx) -> Tuple[str, str]:
                    ps = columns[idx].select("p")
                    if len(ps) >= 2:
                        return (_norm_text(ps[0].get_text()), _norm_text(ps[1].get_text()))
                    t = _norm_text(columns[idx].get_text(" ", strip=True))
                    parts = [p.strip() for p in t.split() if p.strip()]
                    return (parts[0] if parts else "", parts[1] if len(parts) > 1 else "")

                f_anchors = columns[0].select('a[href*="/fighter-details/"]')
                two_ids = [_id_from_href(a.get("href"), "fighter") for a in f_anchors][:2]
                fighter_ids = two_ids if len(two_ids) == 2 else header_fighter_ids[:2]

                kd_a, kd_b = cell_pairs(1)
                sig_a, sig_b = cell_pairs(2)
                sigp_a, sigp_b = cell_pairs(3)
                tot_a, tot_b = cell_pairs(4)
                td_a, td_b = cell_pairs(5)
                tdp_a, tdp_b = cell_pairs(6)
                sub_a, sub_b = cell_pairs(7)
                rev_a, rev_b = cell_pairs(8)
                ctrl_a, ctrl_b = cell_pairs(9)

                pairs = [
                    (fighter_ids[0], kd_a, sig_a, sigp_a, tot_a, td_a, tdp_a, sub_a, rev_a, ctrl_a),
                    (fighter_ids[1], kd_b, sig_b, sigp_b, tot_b, td_b, tdp_b, sub_b, rev_b, ctrl_b),
                ]
                for fid, kd_t, sig_t, sigp_t, tot_t, td_t, tdp_t, sub_t, rev_t, ctrl_t in pairs:
                    if not fid:
                        continue
                    landed_sig, att_sig = _parse_pair(sig_t)
                    landed_tot, att_tot = _parse_pair(tot_t)
                    td_l, td_a2 = _parse_pair(td_t)
                    totals_rounds.setdefault(current_round, {})[fid] = dict(
                        kd=int(kd_t) if kd_t and kd_t.isdigit() else 0,
                        sig_landed=landed_sig, sig_attempted=att_sig, sig_pct=_parse_pct(sigp_t),
                        tot_landed=landed_tot, tot_attempted=att_tot,
                        td_landed=td_l, td_attempted=td_a2, td_pct=_parse_pct(tdp_t),
                        sub_att=int(sub_t) if sub_t and sub_t.isdigit() else 0,
                        rev=int(rev_t) if rev_t and rev_t.isdigit() else 0,
                        ctrl_time_sec=_time_to_sec(ctrl_t),
                    )

    # B) Significant strikes breakdown table (unchanged)
    sig_rounds: Dict[int, Dict[str, Dict[str, Optional[int]]]] = {}
    sig_table = None
    for t in doc.find_all("table", class_="b-fight-details__table"):
        header_txt = " ".join(th.get_text(" ", strip=True) for th in t.select("thead tr th"))
        if "Head" in header_txt and "Distance" in header_txt and "KD" not in header_txt:
            low = t.get_text(" ", strip=True).lower()
            if "round 1" in low:
                sig_table = t
                break
    if sig_table:
        for current_round, rows_pack, is_stacked in _iter_round_blocks(sig_table):
            if not rows_pack:
                continue
            el = rows_pack[0]
            columns = el.find_all("td")
            f_anchors = columns[0].select('a[href*="/fighter-details/"]')
            two_ids = [_id_from_href(a.get("href"), "fighter") for a in f_anchors][:2]
            fighter_ids = two_ids if len(two_ids) == 2 else header_fighter_ids[:2]

            def cell_pairs(idx) -> Tuple[str, str]:
                ps = columns[idx].select("p")
                if len(ps) >= 2:
                    return (_norm_text(ps[0].get_text()), _norm_text(ps[1].get_text()))
                t = _norm_text(columns[idx].get_text(" ", strip=True))
                parts = [p.strip() for p in t.split() if p.strip()]
                return (parts[0] if parts else "", parts[1] if len(parts) > 1 else "")

            head_a, head_b = cell_pairs(3)
            body_a, body_b = cell_pairs(4)
            leg_a, leg_b = cell_pairs(5)
            dist_a, dist_b = cell_pairs(6)
            clin_a, clin_b = cell_pairs(7)
            grnd_a, grnd_b = cell_pairs(8)

            for fid, h_t, b_t, l_t, d_t, c_t, g_t in [
                (fighter_ids[0], head_a, body_a, leg_a, dist_a, clin_a, grnd_a),
                (fighter_ids[1], head_b, body_b, leg_b, dist_b, clin_b, grnd_b),
            ]:
                if not fid:
                    continue
                hL, hA = _parse_pair(h_t)
                bL, bA = _parse_pair(b_t)
                lL, lA = _parse_pair(l_t)
                dL, dA = _parse_pair(d_t)
                cL, cA = _parse_pair(c_t)
                gL, gA = _parse_pair(g_t)
                sig_rounds.setdefault(current_round, {})[fid] = dict(
                    head_landed=hL, head_attempted=hA,
                    body_landed=bL, body_attempted=bA,
                    leg_landed=lL, leg_attempted=lA,
                    distance_landed=dL, distance_attempted=dA,
                    clinch_landed=cL, clinch_attempted=cA,
                    ground_landed=gL, ground_attempted=gA,
                )

    # ---------------- Merge per-round dicts into RoundRow[] ----------------
    rows: List[RoundRow] = []
    all_rounds = sorted(set(totals_rounds.keys()) | set(sig_rounds.keys()))
    for rnd in all_rounds:
        fids = set(totals_rounds.get(rnd, {}).keys()) | set(sig_rounds.get(rnd, {}).keys())
        for fid in fids:
            t = totals_rounds.get(rnd, {}).get(fid, {})
            g = sig_rounds.get(rnd, {}).get(fid, {})
            rows.append(RoundRow(
                fight_id=fight_id,
                event_id=event_id,
                round=rnd,
                fighter_id=fid,
                fighter_corner=corner_map.get(fid),
                kd=t.get("kd"),
                sig_landed=t.get("sig_landed"),
                sig_attempted=t.get("sig_attempted"),
                sig_pct=t.get("sig_pct"),
                tot_landed=t.get("tot_landed"),
                tot_attempted=t.get("tot_attempted"),
                td_landed=t.get("td_landed"),
                td_attempted=t.get("td_attempted"),
                td_pct=t.get("td_pct"),
                sub_att=t.get("sub_att"),
                rev=t.get("rev"),
                ctrl_time_sec=t.get("ctrl_time_sec"),
                head_landed=g.get("head_landed"),
                head_attempted=g.get("head_attempted"),
                body_landed=g.get("body_landed"),
                body_attempted=g.get("body_attempted"),
                leg_landed=g.get("leg_landed"),
                leg_attempted=g.get("leg_attempted"),
                distance_landed=g.get("distance_landed"),
                distance_attempted=g.get("distance_attempted"),
                clinch_landed=g.get("clinch_landed"),
                clinch_attempted=g.get("clinch_attempted"),
                ground_landed=g.get("ground_landed"),
                ground_attempted=g.get("ground_attempted"),
            ))

    return meta, rows
# --------------------------------------------------------------------------- #
# I/O helpers                                                                  #
# --------------------------------------------------------------------------- #

def _load_events_from_csv(path: str) -> List[str]:
    if not os.path.exists(path):
        print(f"[events] not found: {path}")
        return []
    df = pd.read_csv(path)
    # accept either 'event_id' named column or first column as IDs
    if "event_id" in df.columns:
        col = "event_id"
    else:
        col = df.columns[0]
    ids = [str(x) for x in df[col].dropna().astype(str).tolist()]
    return ids


def _upsert_outputs(fights_out_path: str, rounds_out_path: str, fights: List[FightMeta], rounds: List[RoundRow]):
    # 1) sanitize & dedup fights
    dedup_fights: Dict[str, FightMeta] = {}
    for f in fights:
        fid = (f.fight_id or "").strip()
        if not fid:
            continue
        dedup_fights[fid] = f  # last one wins, but they should be identical
    fights_clean = list(dedup_fights.values())

    # 2) sanitize & dedup rounds (by fight_id,round,fighter_id)
    dedup_rounds: Dict[Tuple[str,int,str], RoundRow] = {}
    for r in rounds:
        fid = (r.fight_id or "").strip()
        if not fid or r.round is None or not r.fighter_id:
            continue
        key = (fid, int(r.round), r.fighter_id)
        dedup_rounds[key] = r
    rounds_clean = list(dedup_rounds.values())

    # 3) compute adds/updates for visibility
    fight_rows = [asdict(f) for f in fights_clean]
    round_rows = [asdict(r) for r in rounds_clean]

    existing_fight_keys = _read_existing_keys(fights_out_path, ["fight_id"])
    incoming_fight_keys = _key_tuple_from_rows(fight_rows, ["fight_id"])
    new_fight_keys = incoming_fight_keys - existing_fight_keys
    updated_fight_keys = incoming_fight_keys & existing_fight_keys

    existing_round_keys = _read_existing_keys(rounds_out_path, ["fight_id","round","fighter_id"])
    incoming_round_keys = _key_tuple_from_rows(round_rows, ["fight_id","round","fighter_id"])
    new_round_keys = incoming_round_keys - existing_round_keys
    updated_round_keys = incoming_round_keys & existing_round_keys

    # 4) write fights
    _upsert_csv_compat(
        fights_out_path,
        fight_rows,
        keys=["fight_id"],
        field_order=[
            "fight_id","event_id","r_fighter_id","b_fighter_id",
            "r_fighter_name","b_fighter_name","weight_class","winner_corner",
            "fight_url","scheduled_rounds","method","end_round","end_time_sec",
            "is_title","judge_scores","debug_li"
        ],
    )
    # 5) write rounds
    _upsert_csv_compat(
        rounds_out_path,
        round_rows,
        keys=["fight_id", "round", "fighter_id"],
        field_order=[
            "fight_id","event_id","round","fighter_id","fighter_corner",
            "kd","sig_landed","sig_attempted","sig_pct",
            "tot_landed","tot_attempted",
            "td_landed","td_attempted","td_pct",
            "sub_att","rev","ctrl_time_sec",
            "head_landed","head_attempted",
            "body_landed","body_attempted",
            "leg_landed","leg_attempted",
            "distance_landed","distance_attempted",
            "clinch_landed","clinch_attempted",
            "ground_landed","ground_attempted",
        ],
    )

    # 6) friendly console summary
    print(f"[fights.csv] +{len(new_fight_keys)} new, {len(updated_fight_keys)} updated, {len(fight_rows)} written in this batch")
    print(f"[stats_round.csv] +{len(new_round_keys)} new, {len(updated_round_keys)} updated, {len(round_rows)} written in this batch")

    # 7) manifest (safe/no-op compatible)
    try:
        _update_manifest_compat({
            "fights.csv_rows": len(fight_rows),
            "stats_round.csv_rows": len(round_rows),
            "fights_csv_path": fights_out_path,
            "stats_round_csv_path": rounds_out_path,
        })
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Scrape UFCStats per-round data; builds fights.csv and stats_round.csv without preexisting seeds."
    )
    ap.add_argument("--events-csv", default=DEFAULT_EVENTS_CSV, help="CSV with event_id in first column (or named 'event_id')")
    ap.add_argument("--event-ids", default="", help="Comma-separated event ids to scrape (overrides --events-csv)")
    ap.add_argument("--max-events", type=int, default=0, help="Limit number of events (0 = all)")
    ap.add_argument("--limit-fights", type=int, default=0, help="Stop after scraping N fights (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rate", type=float, default=10.0, help="Requests per second (per host)")
    ap.add_argument("--retries", type=int, default=1, help="HTTP retries")
    ap.add_argument("--timeout", type=float, default=6.0, help="HTTP timeout seconds")
    ap.add_argument("--fights-out", default=DEFAULT_FIGHTS_CSV, help="Output path for fights.csv")
    ap.add_argument("--rounds-out", default=DEFAULT_ROUNDSTAT_CSV, help="Output path for stats_round.csv")
    args = ap.parse_args()

    # Network profile
    http_common.set_network_profile(
        rate_per_sec=args.rate,
        retries=args.retries,
        backoff=0.15,
        timeout=args.timeout,
    )

    # Build event id list
    if args.event_ids.strip():
        event_ids = [e.strip() for e in args.event_ids.split(",") if e.strip()]
    else:
        event_ids = _load_events_from_csv(args.events_csv)

    if not event_ids:
        raise SystemExit("[events] No event ids provided or found.")

    if args.max_events and args.max_events > 0:
        event_ids = event_ids[: args.max_events]

    # Crawl event pages → fight seeds (STOP as soon as we’ve collected enough)
    seeds: List[Dict] = []
    for eid in event_ids:
        try:
            seeds.extend(fetch_event_fights(eid))
        except Exception as e:
            print(f"[event] failed {eid}: {e}")
        # short-circuit if we only need N fights for this run
        if args.limit_fights and args.limit_fights > 0 and len(seeds) >= args.limit_fights:
            break

    # Dedup while preserving order
    seen = set()
    ordered_seeds: List[Dict] = []
    for s in seeds:
        fid = s.get("fight_id")
        if fid and fid not in seen:
            seen.add(fid)
            ordered_seeds.append(s)

    if args.limit_fights and args.limit_fights > 0:
        ordered_seeds = ordered_seeds[: args.limit_fights]

    if not ordered_seeds:
        raise SystemExit("[seeds] No fights discovered from the given events.")

    # Scrape fight-details concurrently
    fights_meta: List[FightMeta] = []
    rounds_all: List[RoundRow] = []
    lock = Lock()


    def work(seed: Dict):
        """Fetch + parse a single fight with a small retry loop for 429s."""
        url = seed["fight_url"]
        attempts = 3
        for i in range(attempts):
            try:
                return parse_fight_details(url, seed)
            except Exception as e:
                msg = str(e)
                if i < attempts - 1 and ("429" in msg or "Too Many Requests" in msg):
                    # small stagger to avoid thundering herd
                    time.sleep(1.0 + i + random.uniform(0, 0.5))
                    continue
                raise

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, seed): seed for seed in ordered_seeds}
        done_count = 0
        total = len(futs)
        for fut in as_completed(futs):
            sd = futs[fut]
            try:
                meta, rows = fut.result()
                with lock:
                    fights_meta.append(meta)
                    rounds_all.extend(rows)
            except Exception as e:
                print(f"[fight] failed {sd.get('fight_id')} :: {sd.get('fight_url')} :: {e}")
            finally:
                done_count += 1
                if done_count % 25 == 0 or done_count == total:
                    print(f"[progress] processed {done_count}/{total}")

    _upsert_outputs(args.fights_out, args.rounds_out, fights_meta, rounds_all)
    print(f"[done] attempted_fights={len(fights_meta)} attempted_round_rows={len(rounds_all)} → {args.fights_out} / {args.rounds_out}")

if __name__ == "__main__":
    main()