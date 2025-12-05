# src/scraper/bestfightodds/scraping/backfill_odds.py
from __future__ import annotations

import argparse
import html
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from src.scraper.common.http import get_html, set_network_profile, set_identity, HttpError

# ----------------------- Defaults & headers -----------------------

DEFAULT_FIGHTS = "data/curated/fights.csv"
DEFAULT_ODDS   = "data/curated/fighters_odds_bfo_iso.csv"
DEFAULT_LINKS  = "data/curated/fighter_bfo_links.csv"
DEFAULT_OUT    = DEFAULT_ODDS

BASE_HEADERS = {
    "Referer": "https://www.bestfightodds.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

# ----------------------- Name & date helpers ----------------------

ORDINALS = re.compile(r"(st|nd|rd|th)\b", re.IGNORECASE)

def norm_name(s: str) -> str:
    s = html.unescape(str(s or ""))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def to_iso_date(s: str):
    """Parse YYYY-MM-DD → date | None."""
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None

def _normalize_spaces(text: str) -> str:
    """Replace NBSP/thin spaces and collapse to single spaces."""
    if not text:
        return ""
    t = text.replace("\xa0", " ").replace("\u202f", " ").replace("\u2009", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def parse_date_text(text: str):
    """
    Robustly parse BFO-like date strings: 'Sep 10th 2025' / 'September 10th 2025'
    Also tolerant to optional commas and weird spaces.
    Returns date | None.
    """
    if not text:
        return None
    t = _normalize_spaces(text)
    # drop ordinals; normalize optional commas before year
    t = ORDINALS.sub("", t)                        # 'Sep 10th 2025' -> 'Sep 10 2025'
    t = re.sub(r",\s*(\d{2,4})\b", r" \1", t)      # 'Sep 10, 2025' -> 'Sep 10 2025'

    # Find the month-day-year chunk anywhere
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2})\s+(\d{2,4})\b", t)
    if not m:
        return None

    mon, day, year = m.group(1), m.group(2), m.group(3)
    for fmt in ("%b %d %Y", "%B %d %Y", "%b %d %y", "%B %d %y"):
        try:
            dt = datetime.strptime(f"{mon} {day} {year}", fmt)
            if dt.year < 1993:
                dt = dt.replace(year=dt.year + 2000)
            return dt.date()
        except ValueError:
            continue
    return None

def within_tol(a, b, tol_days: int) -> bool:
    """Inclusive tolerance: abs(delta) <= tol_days."""
    if not a or not b:
        return False
    return abs((a - b).days) <= tol_days

# ----------------------- Anti-bot detection -----------------------

_CF_PATTERNS = (
    "Just a moment...",
    "cf-browser-verification",
    "cf-chl-",
    "data-cf-beacon",
)

def looks_like_cloudflare(html_text: str) -> bool:
    if not html_text:
        return False
    lt = html_text[:4096]
    low = lt.lower()
    return any(pat.lower() in low for pat in _CF_PATTERNS)

# ----------------------- Odds parsing helpers ---------------------

def read_odds_from_row(row) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (open, close_low, close_high) strings from a single fighter row.
    Uses the span IDs per your spec; mirrors single closing number.
    """
    def txt(sel):
        el = row.select_one(sel)
        return el.get_text(strip=True) if el else None

    open_str = txt("span#oID0")
    low_str  = txt("span#oID1")
    high_str = txt("span#oID2")

    if low_str and not high_str:
        high_str = low_str
    if high_str and not low_str:
        low_str = high_str
    return (open_str, low_str, high_str)

def row_date_fallback(row):
    """
    Pull a date from the row. If the usual cell isn't present, scan the whole row text.
    """
    td = (
        row.select_one("td.item-non-mobile")
        or row.select_one("td.date")
        or row.select_one("td.right")
    )
    txt = td.get_text(" ", strip=True) if td else row.get_text(" ", strip=True)
    return parse_date_text(txt)

def cache_key_for_bfo(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    return f"bfo_fighter/{slug}"

def build_headers(user_agent: Optional[str], cookie: Optional[str]) -> Dict[str, str]:
    h = dict(BASE_HEADERS)
    if user_agent:
        h["User-Agent"] = user_agent
    if cookie:
        h["Cookie"] = cookie
    return h

# ----------------------- Page scanning logic ----------------------

def iter_event_blocks(html: str):
    """
    Yield (header_text, header_date, [main_rows]) for each event block.
    - header_date may be None; we'll fallback to row date.
    - main_rows contains consecutive 'tr.main-row' until the next header (usually 2).
    """
    soup = BeautifulSoup(html, "lxml")

    # find the odds history table
    table = None
    for t in soup.select("table.team-stats-table"):
        if "odds history for" in (t.get("summary") or "").lower():
            table = t
            break
    if table is None:
        table = soup.find("table")
        if table is None:
            return  # nothing to iterate

    tbody = table.find("tbody") or table
    rows = tbody.find_all("tr", recursive=False)

    i = 0
    while i < len(rows):
        tr = rows[i]
        classes = set(tr.get("class") or [])
        if "event-header" in classes:
            header_text = _normalize_spaces(tr.get_text(" ", strip=True))
            header_date = parse_date_text(header_text)  # may be None
            # collect following main rows until next header
            mains: List = []
            j = i + 1
            while j < len(rows):
                cj = set(rows[j].get("class") or [])
                if "event-header" in cj:
                    break
                if "main-row" in cj:
                    mains.append(rows[j])
                j += 1
            yield (header_text, header_date, mains)
            i = j
        else:
            i += 1

def find_target_pair(html: str, target_date, tol_days: int):
    """
    For a given BFO fighter page HTML and a target fight date, select the event block
    whose effective date (header or row-level fallback) is within ±tol_days AND is
    the closest to target_date. Returns (row1, row2) if found, else None.
    """
    best = None  # (abs_delta, (row1,row2))
    for _hdr_text, header_date, mains in iter_event_blocks(html):
        if len(mains) < 2:
            continue

        # Decide the event date for matching:
        # 1) header date if present
        # 2) else per-row grey date (first main row)
        event_date = header_date or row_date_fallback(mains[0])
        if not event_date and len(mains) > 1:
            event_date = row_date_fallback(mains[1])
        if not event_date:
            continue

        delta = abs((event_date - target_date).days)
        if delta <= tol_days:
            if (best is None) or (delta < best[0]):
                best = (delta, (mains[0], mains[1]))

    return best[1] if best else None

# ----------------------- Data loading maps ------------------------

def build_odds_map(odds_df: pd.DataFrame) -> Dict[str, set]:
    """
    normalized name -> set of ISO dates present in crude odds
    """
    mp: Dict[str, set] = {}
    for _, r in odds_df.iterrows():
        n = norm_name(r.get("fighter_name", ""))
        d = to_iso_date(r.get("date_iso", ""))
        if n and d:
            mp.setdefault(n, set()).add(d)
    return mp

def build_link_map(links_df: pd.DataFrame) -> Dict[str, str]:
    """
    normalized name -> BFO fighter URL
    """
    mp: Dict[str, str] = {}
    for _, r in links_df.iterrows():
        n = norm_name(r.get("fighter_name", ""))
        u = r.get("fighter_link_bfo", "")
        if n and u and n not in mp:
            mp[n] = u
    return mp

# ----------------------- Backfill one fight -----------------------

def backfill_one(page_url: str, fight_date_iso: str, tol_days: int, missing_csv_name: str,
                 user_agent: Optional[str], cookie: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Open matched fighter's page, find event within ±tol_days of fight_date_iso,
    take the SECOND main-row (opponent), extract odds, and build a row using the
    missing fighter's CSV name.
    """
    headers = build_headers(user_agent, cookie)
    html = get_html(page_url, cache_key=cache_key_for_bfo(page_url), ttl_hours=24, headers=headers)
    if looks_like_cloudflare(html):
        # Surface as no-match; diagnostics script will pinpoint anti-bot
        return None

    target_date = to_iso_date(fight_date_iso)
    if not target_date:
        return None

    pair = find_target_pair(html, target_date, tol_days)
    if not pair:
        return None

    _, opponent_row = pair  # second row = opponent (per your rule)
    open_str, low_str, high_str = read_odds_from_row(opponent_row)

    # allow open-only backfills (leave close_* blank)
    if not (open_str or low_str or high_str):
        return None

    return {
        "fighter_name": missing_csv_name,  # strictly from our CSV
        "date_iso": fight_date_iso,
        "open": open_str or "",
        "close_low": low_str or "",
        "close_high": high_str or "",
    }

# ----------------------- Driver ------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Backfill missing opponent odds by matching BFO event date (± tol) and taking the opponent row."
    )
    ap.add_argument("--fights", default=DEFAULT_FIGHTS)
    ap.add_argument("--odds", default=DEFAULT_ODDS)
    ap.add_argument("--links", default=DEFAULT_LINKS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tolerance-days", type=int, default=1)  # strict per request
    ap.add_argument("--rate-per-sec", type=float, default=10.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=0.25)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--user-agent", type=str, default=None, help="Optional UA override sent to BFO")
    ap.add_argument("--cookie", type=str, default=None, help="Optional Cookie header (e.g., cf_clearance=...)")
    args = ap.parse_args()

    set_network_profile(rate_per_sec=args.rate_per_sec, retries=args.retries, backoff=args.backoff, timeout=args.timeout)
    set_identity(user_agent=args.user_agent)

    fights = pd.read_csv(args.fights, dtype=str).fillna("")
    odds   = pd.read_csv(args.odds, dtype=str).fillna("")
    links  = pd.read_csv(args.links, dtype=str).fillna("")

    odds_map  = build_odds_map(odds)
    link_map  = build_link_map(links)
    existing  = set((norm_name(r["fighter_name"]), str(r["date_iso"])) for _, r in odds.iterrows())

    # ---------- diagnostics ----------
    total_fights = len(fights)
    one_sided = 0
    with_link = 0
    headers_matched = 0
    rows_written = 0

    skip_no_link = 0
    skip_no_header_match = 0
    skip_all_odds_missing = 0
    skip_duplicate = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pending: List[Dict[str, str]] = []

    for _, fr in fights.iterrows():
        event_date_iso = str(fr.get("date", ""))
        event_date = to_iso_date(event_date_iso)
        if not event_date:
            continue

        red = str(fr.get("r_fighter_name", ""))
        blu = str(fr.get("b_fighter_name", ""))

        rn = norm_name(red)
        bn = norm_name(blu)

        # inclusive tolerance against every crude odds date we have per fighter
        r_has = any(within_tol(d, event_date, args.tolerance_days) for d in odds_map.get(rn, set()))
        b_has = any(within_tol(d, event_date, args.tolerance_days) for d in odds_map.get(bn, set()))

        # one-sided only
        if not (r_has ^ b_has):
            continue

        one_sided += 1

        matched_name = red if r_has else blu   # page we will open
        missing_name = blu if r_has else red   # row we will append (using CSV name)
        matched_norm = rn if r_has else bn

        page_url = link_map.get(matched_norm)
        if not page_url:
            skip_no_link += 1
            continue
        with_link += 1

        try:
            new_row = backfill_one(page_url, event_date_iso, args.tolerance_days, missing_name,
                                   user_agent=args.user_agent, cookie=args.cookie)
        except HttpError:
            # HTTP error (403/5xx) — treat as no header match for summary
            skip_no_header_match += 1
            continue
        except Exception:
            skip_no_header_match += 1
            continue

        if not new_row:
            skip_no_header_match += 1
            continue

        key = (norm_name(new_row["fighter_name"]), new_row["date_iso"])
        if key in existing:
            skip_duplicate += 1
            continue

        if not (new_row["open"] or new_row["close_low"] or new_row["close_high"]):
            skip_all_odds_missing += 1
            continue

        pending.append(new_row)
        existing.add(key)
        odds_map.setdefault(norm_name(new_row["fighter_name"]), set()).add(to_iso_date(new_row["date_iso"]))
        rows_written += 1
        headers_matched += 1

        if rows_written % max(1, args.progress_every) == 0:
            print(f"[+{rows_written}] Backfilled rows…")

    if not pending:
        print("No rows to backfill. Exiting.")
        print("\n=== Backfill diagnostics ===")
        print(f"Total fights                  : {total_fights}")
        print(f"One-sided fights              : {one_sided}")
        print(f"…with BFO link                : {with_link}")
        print(f"Headers matched (±{args.tolerance_days}d): {headers_matched}")
        print(f"Rows written                  : {rows_written}")
        print(f"Skipped (no link)             : {skip_no_link}")
        print(f"Skipped (no header/within tol): {skip_no_header_match}")
        print(f"Skipped (all odds missing)    : {skip_all_odds_missing}")
        print(f"Skipped (duplicate)           : {skip_duplicate}")
        return

    # Write out (append onto crude odds)
    cols = ["fighter_name", "date_iso", "open", "close_low", "close_high"]
    combined = pd.concat([odds[cols], pd.DataFrame(pending, columns=cols)], ignore_index=True)
    tmp = str(out_path) + ".tmp"
    combined.to_csv(tmp, index=False)
    Path(tmp).replace(out_path)

    print("\n=== Backfill summary ===")
    print(f"Total fights                  : {total_fights}")
    print(f"One-sided fights              : {one_sided}")
    print(f"…with BFO link                : {with_link}")
    print(f"Headers matched (±{args.tolerance_days}d): {headers_matched}")
    print(f"Rows written                  : {rows_written}")
    print(f"Skipped (no link)             : {skip_no_link}")
    print(f"Skipped (no header/within tol): {skip_no_header_match}")
    print(f"Skipped (all odds missing)    : {skip_all_odds_missing}")
    print(f"Skipped (duplicate)           : {skip_duplicate}")
    print(f"\nDone. Wrote → {out_path}")

if __name__ == "__main__":
    main()