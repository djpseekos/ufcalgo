# src/scraper/diagnose_bfo.py
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Iterable

from bs4 import BeautifulSoup  # pip install beautifulsoup4
from src.scraper.common.http import get_html, set_network_profile, set_identity, HttpError

INPUT_CSV = "data/curated/fighter_bfo_links.csv"
OUT_CSV = "data/diagnostics/bfo_diagnostics.csv"

DATE_REGEX = re.compile(r"\b([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})\b")
DAY_SUFFIX = re.compile(r"(st|nd|rd|th)\b", re.IGNORECASE)

BLOCK_PHRASES = [
    "access denied",
    "just a moment",       # common Cloudflare interstitial
    "verify you are a human",
    "captcha",
    "cloudflare",
    "blocked",
]

def normalize_name(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()

def extract_date_string_from_header(td_text: str) -> Optional[str]:
    if not td_text:
        return None
    m = DATE_REGEX.search(td_text)
    return m.group(1) if m else None

def parse_date_for_future_check(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    cleaned = DAY_SUFFIX.sub("", date_str)
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass
    return None

def is_unconfirmed_header(text: str) -> bool:
    return "unconfirmed" in (text or "").lower()

def spans_in_row(row) -> Dict[str, Optional[str]]:
    def t(el): return el.get_text(strip=True) if el else None
    # Scope searches to the row to avoid duplicate IDs elsewhere
    return {
        "open": t(row.select_one("span#oID0")),
        "close_low": t(row.select_one("span#oID1")),
        "close_high": t(row.select_one("span#oID2")),
    }

def read_input(path: str, limit_fighters: Optional[int]) -> List[Dict[str, str]]:
    fighters: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("fighter_name") and row.get("fighter_link_bfo"):
                fighters.append(row)
                if limit_fighters and len(fighters) >= limit_fighters:
                    break
    return fighters

def diagnose_fighter(fighter_name: str, url: str, cache_key: Optional[str]) -> Dict[str, str]:
    rec: Dict[str, str] = {
        "fighter_name": fighter_name,
        "url": url,
        "http_ok": "",
        "http_error": "",
        "html_bytes": "",
        "anti_bot_hit": "",
        "table_found": "",
        "event_headers": "0",
        "headers_skipped_unconfirmed": "0",
        "headers_skipped_future": "0",
        "headers_with_dates": "0",
        "main_rows_total": "0",
        "matched_rows": "0",
        "fights_yielded": "0",
        "name_match_example": "",
        "last_header_date_example": "",
        "missing_closing_both": "0",
        "missing_closing_one_side": "0",
        "open_missing_count": "0",
        "reason_if_empty": "",
    }

    html = ""
    try:
        html = get_html(url, cache_key=cache_key, ttl_hours=24)
        rec["http_ok"] = "yes"
        rec["html_bytes"] = str(len(html))
    except HttpError as e:
        rec["http_ok"] = "no"
        rec["http_error"] = f"HTTP {e.status}"
        rec["reason_if_empty"] = "HTTP_ERROR"
        return rec
    except Exception as e:
        rec["http_ok"] = "no"
        rec["http_error"] = f"EXC {type(e).__name__}"
        rec["reason_if_empty"] = "HTTP_EXCEPTION"
        return rec

    low = (html or "").lower()
    if any(p in low for p in BLOCK_PHRASES):
        rec["anti_bot_hit"] = "maybe"
    else:
        rec["anti_bot_hit"] = "no"

    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.team-stats-table")
    if not table:
        rec["table_found"] = "no"
        rec["reason_if_empty"] = "NO_TABLE"
        return rec

    rec["table_found"] = "yes"

    today = datetime.today().date()
    headers_total = 0
    headers_with_dates = 0
    headers_skip_unconf = 0
    headers_skip_future = 0
    main_rows_total = 0
    matched_rows = 0
    fights_yielded = 0
    missing_closing_both = 0
    missing_closing_one_side = 0
    open_missing_count = 0
    name_match_example = ""

    current_date_display = None
    current_date_dt = None
    current_block_skipped = False

    for tr in table.select("tbody > tr"):
        cls = tr.get("class") or []
        if "event-header" in cls:
            headers_total += 1
            txt = tr.get_text(" ", strip=True)
            current_date_display = extract_date_string_from_header(txt)
            rec["last_header_date_example"] = current_date_display or rec["last_header_date_example"]
            if current_date_display:
                headers_with_dates += 1
                dt = parse_date_for_future_check(current_date_display)
                current_date_dt = dt
            else:
                current_date_dt = None

            current_block_skipped = False
            if is_unconfirmed_header(txt):
                headers_skip_unconf += 1
                current_block_skipped = True
                continue
            if current_date_dt and current_date_dt.date() > today:
                headers_skip_future += 1
                current_block_skipped = True
                continue
            continue

        if current_date_display is None or current_block_skipped:
            continue

        if "main-row" not in cls:
            continue

        main_rows_total += 1
        name_cell = tr.select_one("th")
        if not name_cell:
            continue
        row_name = normalize_name(name_cell.get_text(" ", strip=True))
        if not name_match_example:
            name_match_example = row_name

        if row_name != normalize_name(fighter_name):
            continue

        matched_rows += 1
        spans = spans_in_row(tr)
        open_str = spans.get("open") or ""
        low_str = spans.get("close_low")
        high_str = spans.get("close_high")

        if not open_str:
            open_missing_count += 1

        if not low_str and not high_str:
            # would be skipped by main scraper
            missing_closing_both += 1
            continue

        if low_str and not high_str:
            missing_closing_one_side += 1
            high_str = low_str
        if high_str and not low_str:
            missing_closing_one_side += 1
            low_str = high_str

        fights_yielded += 1

    rec["event_headers"] = str(headers_total)
    rec["headers_with_dates"] = str(headers_with_dates)
    rec["headers_skipped_unconfirmed"] = str(headers_skip_unconf)
    rec["headers_skipped_future"] = str(headers_skip_future)
    rec["main_rows_total"] = str(main_rows_total)
    rec["matched_rows"] = str(matched_rows)
    rec["fights_yielded"] = str(fights_yielded)
    rec["missing_closing_both"] = str(missing_closing_both)
    rec["missing_closing_one_side"] = str(missing_closing_one_side)
    rec["open_missing_count"] = str(open_missing_count)
    rec["name_match_example"] = name_match_example

    # Reason inference if no rows yielded
    if fights_yielded == 0:
        if headers_total == 0:
            rec["reason_if_empty"] = "NO_EVENT_HEADERS"
        elif headers_total and (headers_skip_unconf + headers_skip_future) == headers_total:
            rec["reason_if_empty"] = "ALL_BLOCKS_SKIPPED"
        elif main_rows_total == 0:
            rec["reason_if_empty"] = "NO_MAIN_ROWS"
        elif matched_rows == 0:
            rec["reason_if_empty"] = "NAME_NEVER_FOUND"
        elif missing_closing_both > 0 and (missing_closing_both == matched_rows):
            rec["reason_if_empty"] = "NO_CLOSING_NUMBERS"
        elif rec["anti_bot_hit"] == "maybe":
            rec["reason_if_empty"] = "POSSIBLE_ANTIBOT"
        else:
            rec["reason_if_empty"] = "FILTERS_ELIMINATED_ALL"
    else:
        rec["reason_if_empty"] = ""

    return rec

def write_report(rows: List[Dict[str, str]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        # still create an empty file with header
        fields = ["fighter_name","url","http_ok","http_error","html_bytes","anti_bot_hit","table_found",
                  "event_headers","headers_skipped_unconfirmed","headers_skipped_future","headers_with_dates",
                  "main_rows_total","matched_rows","fights_yielded","name_match_example","last_header_date_example",
                  "missing_closing_both","missing_closing_one_side","open_missing_count","reason_if_empty"]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
        return

    fields = list(rows[0].keys())
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, out_path)

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose why BestFightOdds scraping returned empty output.")
    ap.add_argument("--input", default=INPUT_CSV, help="Path to fighter_bfo_links.csv")
    ap.add_argument("--output", default=OUT_CSV, help="Diagnostics CSV to write")
    ap.add_argument("--limit-fighters", type=int, default=None, help="Check only first N fighters")
    # networking knobs (forward to http.py)
    ap.add_argument("--rate-per-sec", type=float, default=12.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=0.2)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--user-agent", type=str, default=None)
    ap.add_argument("--contact-email", type=str, default=None)

    args = ap.parse_args(argv)

    set_network_profile(rate_per_sec=args.rate_per_sec,
                        retries=args.retries,
                        backoff=args.backoff,
                        timeout=args.timeout)
    if args.user_agent or args.contact_email:
        set_identity(user_agent=args.user_agent, contact_email=args.contact_email)

    fighters = read_input(args.input, args.limit_fighters)
    if not fighters:
        print(f"[ERROR] No fighters loaded from {args.input}. Headers needed: fighter_name,fighter_link_bfo", file=sys.stderr)
        return 2

    rows: List[Dict[str, str]] = []
    total = len(fighters)
    for i, row in enumerate(fighters, start=1):
        name = row["fighter_name"].strip()
        url = row["fighter_link_bfo"].strip()
        cache_key = row.get("fighter_id") or None
        try:
            diag = diagnose_fighter(name, url, cache_key)
            rows.append(diag)
        except Exception as e:
            rows.append({
                "fighter_name": name,
                "url": url,
                "http_ok": "",
                "http_error": f"EXC {type(e).__name__}",
                "html_bytes": "",
                "anti_bot_hit": "",
                "table_found": "",
                "event_headers": "0",
                "headers_skipped_unconfirmed": "0",
                "headers_skipped_future": "0",
                "headers_with_dates": "0",
                "main_rows_total": "0",
                "matched_rows": "0",
                "fights_yielded": "0",
                "name_match_example": "",
                "last_header_date_example": "",
                "missing_closing_both": "0",
                "missing_closing_one_side": "0",
                "open_missing_count": "0",
                "reason_if_empty": f"UNCAUGHT_EXCEPTION:{e}",
            })
        if i % 10 == 0 or i == total:
            print(f"diagnosed {i}/{total} fighters")

    write_report(rows, args.output)
    print(f"[OK] Wrote diagnostics to {args.output}")
    # Quick summary
    empties = sum(1 for r in rows if (r.get("fights_yielded") == "0"))
    print(f"Summary: {empties}/{total} fighters produced zero rows. Top reasons:")
    reasons: Dict[str,int] = {}
    for r in rows:
        reason = r.get("reason_if_empty") or "(NON_EMPTY)"
        reasons[reason] = reasons.get(reason, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())