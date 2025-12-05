# src/scraper/scrape_odds_bfo.py
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from typing import Iterable, List, Dict, Optional

from bs4 import BeautifulSoup  # pip install beautifulsoup4
# Project-local HTTP helper
from src.scraper.common.http import get_html, set_network_profile, set_identity, HttpError


INPUT_CSV = "data/curated/fighter_bfo_links.csv"
OUTPUT_CSV = "data/curated/fighters_odds_bfo.csv"


DATE_REGEX = re.compile(r"\b([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})\b")
DAY_SUFFIX = re.compile(r"(st|nd|rd|th)\b", re.IGNORECASE)
ODDS_RE = re.compile(r"[+-]\d+")


def row_fighter_name(tr) -> str:
    """Get the clean fighter name from the name cell."""
    a = tr.select_one("th a")
    if a:
        return " ".join(a.get_text(" ", strip=True).split())
    th = tr.select_one("th")
    return " ".join(th.get_text(" ", strip=True).split()) if th else ""


def normalize_name(s: str) -> str:
    """Lowercase + collapse internal whitespace for robust matching/logging."""
    return " ".join((s or "").split()).strip().lower()


def extract_date_string_from_header(td_text: str) -> Optional[str]:
    """Return the on-page date substring exactly as shown (e.g., 'Apr 3rd 2009')."""
    if not td_text:
        return None
    m = DATE_REGEX.search(td_text)
    return m.group(1) if m else None


def is_unconfirmed_header(td_text: str) -> bool:
    return "unconfirmed" in (td_text or "").lower()


def parse_date_for_future_check(date_str: str) -> Optional[datetime]:
    """Parse 'Jan 1st 2026' → datetime(2026,1,1) for future filtering."""
    if not date_str:
        return None
    cleaned = DAY_SUFFIX.sub("", date_str)
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass
    return None


def spans_in_row(row) -> Dict[str, Optional[str]]:
    """
    Within a fighter row, find the odds spans by id:
      oID0 -> open
      oID1 -> close_low
      oID2 -> close_high
    We scope searches to the row to avoid duplicate IDs on the page.
    """
    def text_or_none(el):
        return el.get_text(strip=True) if el else None

    open_span = row.select_one("span#oID0")
    low_span  = row.select_one("span#oID1")
    high_span = row.select_one("span#oID2")

    return {
        "open": text_or_none(open_span),
        "close_low": text_or_none(low_span),
        "close_high": text_or_none(high_span),
    }


def fallback_odds_from_cells(row) -> Dict[str, Optional[str]]:
    """
    Fallback if oID spans are absent:
    - Look at moneyline cells in the row, extract numeric odds.
    - Heuristic: the first number is 'open'; next two are 'close_low' and 'close_high'.
      If only one closing number exists, mirror it.
    """
    nums: List[str] = []
    for td in row.select("td.moneyline"):
        # take the first +/−ddd in each relevant cell
        text = td.get_text(" ", strip=True)
        m = ODDS_RE.search(text)
        if m:
            nums.append(m.group(0))
    open_str, low_str, high_str = None, None, None
    if nums:
        open_str = nums[0]
    if len(nums) >= 2:
        low_str = nums[1]
    if len(nums) >= 3:
        high_str = nums[2]
    if low_str and not high_str:
        high_str = low_str
    if high_str and not low_str:
        low_str = high_str
    return {"open": open_str, "close_low": low_str, "close_high": high_str}


def iter_fights(soup, fighter_name: str) -> Iterable[Dict[str, str]]:
    """
    Iterate through the table, yielding rows for the requested fighter.
    Returns dicts with keys: fighter_name, date, open, close_low, close_high
    """
    table = soup.select_one("table.team-stats-table")
    if not table:
        return  # nothing to yield

    current_date_display: Optional[str] = None
    skip_block: bool = False
    today = datetime.today().date()

    for tr in table.select("tbody > tr"):
        classes = tr.get("class") or []

        # Header rows set date context and skip flags
        if "event-header" in classes:
            header_text = tr.get_text(" ", strip=True)
            current_date_display = extract_date_string_from_header(header_text)
            skip_block = False
            if is_unconfirmed_header(header_text):
                skip_block = True
                continue
            dt = parse_date_for_future_check(current_date_display or "")
            if dt and dt.date() > today:
                skip_block = True
                continue
            continue

        # Ignore anything before first header or inside skipped block
        if current_date_display is None or skip_block:
            continue

        # We only care about the fighter rows
        if "main-row" not in classes:
            continue

        # Treat EACH main-row as the subject row (some pages don't include opponent rows)
        subj_row = tr

        # Try strict spans first; then fallback to moneyline text
        spans = spans_in_row(subj_row)
        open_str = spans.get("open")
        low_str  = spans.get("close_low")
        high_str = spans.get("close_high")

        if not (low_str and high_str):
            fb = fallback_odds_from_cells(subj_row)
            open_str = open_str or fb.get("open")
            if not low_str:
                low_str = fb.get("close_low")
            if not high_str:
                high_str = fb.get("close_high")

        # If still no closing numbers at all, skip the fight
        if not low_str and not high_str:
            continue
        # If exactly one closing number, mirror to both ends
        if low_str and not high_str:
            high_str = low_str
        if high_str and not low_str:
            low_str = high_str

        yield {
            "fighter_name": fighter_name,
            "date": current_date_display or "",
            "open": (open_str or ""),
            "close_low": (low_str or ""),
            "close_high": (high_str or ""),
        }


def read_input(path: str, limit_fighters: Optional[int]) -> List[Dict[str, str]]:
    fighters: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not (row.get("fighter_name") and row.get("fighter_link_bfo")):
                continue
            fighters.append(row)
            if limit_fighters is not None and len(fighters) >= limit_fighters:
                break
    return fighters


def write_output(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["fighter_name", "date", "open", "close_low", "close_high"]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "fighter_name": r.get("fighter_name", ""),
                "date": r.get("date", ""),
                "open": r.get("open", ""),
                "close_low": r.get("close_low", ""),
                "close_high": r.get("close_high", ""),
            })
    os.replace(tmp, path)


def scrape_for_fighter(fighter_name: str, url: str, cache_key: Optional[str]) -> List[Dict[str, str]]:
    html = get_html(url, cache_key=cache_key, ttl_hours=24)
    soup = BeautifulSoup(html, "html.parser")
    return list(iter_fights(soup, fighter_name))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape historical closing range odds from BestFightOdds.")
    parser.add_argument("--input", default=INPUT_CSV, help="Path to fighter_bfo_links.csv")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Path to write fighters_odds_bfo.csv (overwrites)")
    parser.add_argument("--limit-fighters", type=int, default=None, help="Process only the first N fighters")
    # Networking knobs (forwarded into http.py)
    parser.add_argument("--rate-per-sec", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--user-agent", type=str, default=None)
    parser.add_argument("--contact-email", type=str, default=None)

    args = parser.parse_args(argv)

    set_network_profile(
        rate_per_sec=args.rate_per_sec,
        retries=args.retries,
        backoff=args.backoff,
        timeout=args.timeout,
    )
    if args.user_agent or args.contact_email:
        set_identity(user_agent=args.user_agent, contact_email=args.contact_email)

    fighters = read_input(args.input, args.limit_fighters)
    total_fighters = len(fighters)

    results: List[Dict[str, str]] = []
    processed = 0

    for idx, row in enumerate(fighters, start=1):
        fighter_name = row["fighter_name"].strip()
        url = row["fighter_link_bfo"].strip()
        cache_key = row.get("fighter_id") or None

        try:
            rows = scrape_for_fighter(fighter_name, url, cache_key=cache_key)
            results.extend(rows)
        except HttpError as e:
            print(f"[WARN] Skipping fighter due to HTTP error: {fighter_name} :: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Skipping fighter due to parse error: {fighter_name} :: {e}", file=sys.stderr)
        finally:
            processed += 1
            if processed % 10 == 0 or processed == total_fighters:
                print(f"finished {processed}/{total_fighters} fighters")

    write_output(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())