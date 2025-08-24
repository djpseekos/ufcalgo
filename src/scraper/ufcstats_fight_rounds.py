from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .common import http as http_common
from .common.parse import soup
from .common.io import load_csv, upsert_csv, update_manifest

FIGHTS_CSV = "data/curated/fights.csv"
ROUNDSTAT_CSV = "data/curated/stats_round.csv"
DEBUG_DIR = "data/debug"

# --- Helpers -----------------------------------------------------------------

_MMSS = re.compile(r"^(\d{1,2}):(\d{2})$")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def _norm(n: Optional[str]) -> str:
    """Normalize fighter display names for robust side matching."""
    if not n:
        return ""
    n = n.replace("\xa0", " ")
    n = re.sub(r"\s+", " ", n).strip().lower()
    n = re.sub(r"[^a-z .'\-]", "", n)
    return n


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
    """
    Parse Method, Round, Time, Time format, Details/Judges, Title Bout.
    Supports both legacy <ul.b-fight-details__list><li>…</li></ul>
    and the newer inline <div.b-fight-details__content> with
    <i class="b-fight-details__label">Label:</i> Value blocks.
    """
    out: Dict[str, Optional[str]] = {
        "method": None,
        "end_round": None,
        "end_time": None,
        "scheduled_rounds": None,
        "is_title": None,
        "judge_scores": None,
        "debug_li": "",
    }

    def assign(label: str, value: str):
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
        # Title bout is a presence marker (can appear outside label/value)
        if "title bout" in (label + " " + value).lower():
            out["is_title"] = "1"

    lines_collected = []

    # --- New inline format: inside div.b-fight-details__content -------------
    content = doc.select_one("div.b-fight-details__content")
    if content:
        # Each info chunk is in .b-fight-details__text(_item[_first]) with an inner label
        for block in content.select(".b-fight-details__text, .b-fight-details__text-item, .b-fight-details__text-item_first"):
            label_el = block.select_one(".b-fight-details__label")
            if not label_el:
                # still scan the text for "Title Bout"
                txt = block.get_text(" ", strip=True)
                lines_collected.append(txt)
                if "title bout" in txt.lower():
                    out["is_title"] = "1"
                continue

            label_txt = label_el.get_text(" ", strip=True)
            whole = block.get_text(" ", strip=True)
            lines_collected.append(whole)

            # value = whole text minus the leading label (and any colon/space)
            val = re.sub(r"^\s*" + re.escape(label_txt) + r"\s*:?\s*", "", whole, flags=re.I)
            assign(label_txt, val)

    # --- Legacy list format: ul.b-fight-details__list li --------------------
    if not out["method"] and not out["end_round"] and not out["end_time"]:
        for li in doc.select("ul.b-fight-details__list li"):
            raw_all = li.get_text(" ", strip=True)
            lines_collected.append(raw_all)
            label_el = li.select_one("i, .b-fight-details__label")
            label_txt = (label_el.get_text(" ", strip=True) if label_el else "")
            val = raw_all
            if label_el:
                val = re.sub(r"^\s*" + re.escape(label_txt) + r"\s*:?\s*", "", raw_all, flags=re.I).strip()
            else:
                # prefix-based fallback
                low = raw_all.lower()
                for c in ("method", "round", "time format", "format", "time", "details", "judges"):
                    if low.startswith(c):
                        label_txt = c
                        val = raw_all.split(":", 1)[-1].strip() if ":" in raw_all else raw_all[len(c):].strip(" :-\u00a0")
                        break
            assign(label_txt, val)

    out["debug_li"] = "\n".join(lines_collected)

    # Final fallback for title marker anywhere on the page
    if out["is_title"] is None and doc.find(string=re.compile(r"\btitle bout\b", re.I)):
        out["is_title"] = "1"

    return out


# --- Round tables parsers -----------------------------------------------------

def _iter_round_blocks(table) -> List[Tuple[int, List]]:
    """
    Round tables often have a row 'Round N' followed by two rows (red, blue).
    Returns [(round, [tr_red, tr_blue]), ...]. Falls back to row pairs.
    """
    tbody = table.find("tbody") or table
    rows = tbody.find_all("tr")  # allow nested; UFCStats sometimes wraps rows
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
            # Fallback: assume row pairs (red, blue)
            if i + 1 < len(rows):
                out.append((len(out) + 1, [rows[i], rows[i + 1]]))
                i += 2
            else:
                break
    return out


def _sig_table(doc):
    """Find the Significant Strikes table (targets and/or positions)."""
    for tbl in doc.select("table"):
        hdrs = _table_headers(tbl)
        has_sig = any("sig" in h for h in hdrs)
        has_target = any(k in hdrs for k in ("head", "body", "leg"))
        has_pos = any(k in hdrs for k in ("distance", "clinch", "ground"))
        if has_sig and (has_target or has_pos):
            return tbl
    # Fallback: a table that contains 'Round 1' and either 'head' or 'distance'
    for tbl in doc.select("table"):
        txt = tbl.get_text(" ", strip=True).lower()
        if "round 1" in txt and ("head" in txt or "distance" in txt):
            return tbl
    return None


def _totals_table(doc):
    """Find the Totals table by headers (KD, Sig. Str., TD, Ctrl) — relaxed."""
    for tbl in doc.select("table"):
        hdrs = _table_headers(tbl)
        joined = " | ".join(hdrs)
        if ("kd" in joined and "sig" in joined):
            return tbl
    # Fallback: 'Round 1' and any of sig/total/kd words
    for tbl in doc.select("table"):
        txt = tbl.get_text(" ", strip=True).lower()
        if "round 1" in txt and ("sig." in txt or "sig str" in txt or "total str" in txt or "kd" in txt):
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


def _parse_rounds(doc, red_name: Optional[str], blue_name: Optional[str]) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """Parse per-round stats from Totals and Significant Strikes tables.
       Returns (round_rows, debug_info) where debug_info has seen headers."""
    rounds: Dict[int, Dict] = {}
    dbg: Dict[str, List[str]] = {"totals_hdrs": [], "sig_hdrs": []}

    totals = _totals_table(doc)
    if totals:
        hdrs = _table_headers(totals)
        dbg["totals_hdrs"] = hdrs

        def _col_contains(key: str) -> int:
            # strict: whole-word-ish
            for idx, h in enumerate(hdrs):
                if re.search(rf"\b{re.escape(key)}\b", h):
                    return idx
            # fallback: substring
            for idx, h in enumerate(hdrs):
                if key in h:
                    return idx
            return -1

        for rnd, pair in _iter_round_blocks(totals):
            rounds.setdefault(rnd, {
                "fight_id": None,
                "round": rnd,
                "r_kd": 0, "r_sig_landed": 0, "r_sig_attempts": 0, "r_td": 0, "r_ctrl_sec": 0,
                "r_head_landed": 0, "r_body_landed": 0, "r_leg_landed": 0,
                "r_distance_landed": 0, "r_clinch_landed": 0, "r_ground_landed": 0,
                "b_kd": 0, "b_sig_landed": 0, "b_sig_attempts": 0, "b_td": 0, "b_ctrl_sec": 0,
                "b_head_landed": 0, "b_body_landed": 0, "b_leg_landed": 0,
                "b_distance_landed": 0, "b_clinch_landed": 0, "b_ground_landed": 0,
            })

            kd_i = _col_contains("kd")
            sig_i = _col_contains("sig")
            td_i = _col_contains("td")
            ctrl_i = _col_contains("ctrl")

            rnorm = _norm(red_name)
            bnorm = _norm(blue_name)

            for tr in pair:
                name = _row_name(tr)
                tnorm = _norm(name)
                cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]

                if tnorm and rnorm and tnorm in rnorm:
                    side = "r"
                elif tnorm and bnorm and tnorm in bnorm:
                    side = "b"
                else:
                    side = "r" if tr is pair[0] else "b"

                if 0 <= kd_i < len(cells):
                    val = _num(cells[kd_i])
                    if val is not None:
                        rounds[rnd][f"{side}_kd"] = val
                if 0 <= sig_i < len(cells):
                    l, a = _of_pair(cells[sig_i])
                    if l is not None:
                        rounds[rnd][f"{side}_sig_landed"] = l or 0
                    if a is not None:
                        rounds[rnd][f"{side}_sig_attempts"] = a or 0
                if 0 <= td_i < len(cells):
                    l, _a = _of_pair(cells[td_i])
                    if l is not None:
                        rounds[rnd][f"{side}_td"] = l or 0
                if 0 <= ctrl_i < len(cells):
                    sec = _mmss_to_seconds(cells[ctrl_i])
                    if sec is not None:
                        rounds[rnd][f"{side}_ctrl_sec"] = sec

    sig = _sig_table(doc)
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
            rounds.setdefault(rnd, {
                "fight_id": None,
                "round": rnd,
                "r_kd": 0, "r_sig_landed": 0, "r_sig_attempts": 0, "r_td": 0, "r_ctrl_sec": 0,
                "r_head_landed": 0, "r_body_landed": 0, "r_leg_landed": 0,
                "r_distance_landed": 0, "r_clinch_landed": 0, "r_ground_landed": 0,
                "b_kd": 0, "b_sig_landed": 0, "b_sig_attempts": 0, "b_td": 0, "b_ctrl_sec": 0,
                "b_head_landed": 0, "b_body_landed": 0, "b_leg_landed": 0,
                "b_distance_landed": 0, "b_clinch_landed": 0, "b_ground_landed": 0,
            })

            rnorm = _norm(red_name)
            bnorm = _norm(blue_name)

            for tr in pair:
                name = _row_name(tr)
                tnorm = _norm(name)
                cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]

                if tnorm and rnorm and tnorm in rnorm:
                    side = "r"
                elif tnorm and bnorm and tnorm in bnorm:
                    side = "b"
                else:
                    side = "r" if tr is pair[0] else "b"

                for key, idx in idx_map.items():
                    if 0 <= idx < len(cells):
                        landed, _attempts = _of_pair(cells[idx])
                        if landed is not None:
                            rounds[rnd][f"{side}_{key}_landed"] = landed or 0

    rows = [rounds[k] for k in sorted(rounds.keys())]
    return rows, dbg


# --- Main per-fight parser ----------------------------------------------------

def _parse_fight_page(fight_id: str, fight_url: str, referer: Optional[str]) -> Tuple[Dict, List[Dict], Dict[str, List[str]]]:
    """
    Fetch and parse a UFCStats fight page.
      - HTTP first (443 often fails)
      - Send Referer pointing to the event (HTTP)
      - Validate we got real fight markup before parsing
    """
    time.sleep(0.15)  # courtesy delay

    def _fetch(proto: str) -> Optional[str]:
        url = re.sub(r"^https?", proto, fight_url)
        try:
            return http_common.get_html(
                url,
                # IMPORTANT: include proto to avoid poisoning cache
                cache_key=f"{proto}_fight_{fight_id}",
                ttl_hours=720,   # historical pages don't change
                timeout=6,       # fail fast so we can fallback
                headers={"Referer": referer or f"{proto}://www.ufcstats.com/"},
            )
        except Exception:
            return None

    def _looks_like_fight_page(txt: str) -> bool:
        if not txt:
            return False
        low = txt.lower()
        return (
            "b-fight-details__person" in low
            or "b-fight-details__table" in low
            or "b-fight-details__list" in low
            or "fight details" in low  # tolerate slight variations
        )

    # Try HTTP first, then HTTPS
    html = _fetch("http")
    if not _looks_like_fight_page(html or ""):
        html = _fetch("https")

    if not _looks_like_fight_page(html or ""):
        return (
            {
                "fight_id": fight_id,
                "method": None,
                "end_round": None,
                "end_time_sec": None,
                "scheduled_rounds": None,
                "is_title": None,
                "judge_scores": None,
                "debug_li": "",
            },
            [],
            {"totals_hdrs": [], "sig_hdrs": []}
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
        "debug_li": meta.get("debug_li", ""),
    }

    round_rows, dbg = _parse_rounds(doc, r_name, b_name)
    for rr in round_rows:
        rr["fight_id"] = fight_id

    return details_row, round_rows, dbg


# --- Orchestration ------------------------------------------------------------

def _dump_debug(fid: str, dbg: Dict[str, List[str]]):
    """Write out whatever we have so we can inspect failures easily."""
    _ensure_dir(DEBUG_DIR)
    # copy any cached html into debug folder
    for proto in ("http", "https"):
        src = f"data/raw/html/{proto}_fight_{fid}.html"
        if os.path.exists(src) and os.path.getsize(src) > 0:
            dst = os.path.join(DEBUG_DIR, f"debug_{proto}_fight_{fid}.html")
            try:
                shutil.copyfile(src, dst)
            except Exception:
                pass
    # write a small text file with parsed li lines and headers
    info_path = os.path.join(DEBUG_DIR, f"debug_fight_{fid}.txt")
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            f.write("[meta li lines]\n")
            # meta lines were stored in details_row['debug_li'] by caller; here we can only dump headers
            f.write("— see HTML for raw meta block —\n\n")
            f.write("[totals headers]\n")
            f.write(", ".join(dbg.get("totals_hdrs", [])) + "\n\n")
            f.write("[sig headers]\n")
            f.write(", ".join(dbg.get("sig_hdrs", [])) + "\n")
    except Exception:
        pass


def scrape_round_stats(limit_fights: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fights = load_csv(FIGHTS_CSV)
    if fights.empty:
        raise SystemExit("fights.csv not found — run event_fights first.")

    if limit_fights:
        fights = fights.head(limit_fights)

    details_updates: List[Dict] = []
    all_rounds: List[Dict] = []
    skipped: List[str] = []
    primed_events: set[str] = set()   # Prime each event page once (HTTP)

    # Newest-first order is fine; these pages are historical
    for i, f in enumerate(fights.itertuples(index=False), 1):
        fid = f.fight_id
        eid = getattr(f, "event_id", None)

        # Always use HTTP referer (port 443 is flaky in some environments)
        referer = f"http://www.ufcstats.com/event-details/{eid}" if eid else "http://www.ufcstats.com/"
        furl = getattr(f, "fight_url", None) or f"https://www.ufcstats.com/fight-details/{fid}"

        # PRIME COOKIES: hit the event page ONCE per event over HTTP, no cache, short timeout
        if eid and eid not in primed_events:
            try:
                _ = http_common.get_html(
                    f"http://www.ufcstats.com/event-details/{eid}",
                    cache_key=None,          # force network
                    ttl_hours=0,             # bypass fresh cache
                    timeout=6,
                    headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
                )
                primed_events.add(eid)
                time.sleep(0.15)  # tiny courtesy delay
            except Exception:
                pass

        details, rounds, dbg = _parse_fight_page(fid, furl, referer=referer)
        if not details["method"] and not rounds:
            skipped.append(fid)
            _dump_debug(fid, dbg)
        else:
            details_updates.append(details)
            all_rounds.extend(rounds)

        if i % 25 == 0:
            print(f"[rounds] processed {i}/{len(fights)} fights…", flush=True)
        time.sleep(0.1)  # polite pacing

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
        "fight_id","method","end_round","end_time_sec","scheduled_rounds","is_title","judge_scores","debug_li"
    ])

    if skipped:
        pd.Series(skipped, name="fight_id").to_csv("data/curated/_rounds_skipped.csv", index=False)
        print(f"[rounds] skipped {len(skipped)} fights due to fetch/parsing issues → data/curated/_rounds_skipped.csv")
        print(f"[rounds] wrote debug artifacts → {DEBUG_DIR}/debug_*_fight_<id>.html (+ .txt)")

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