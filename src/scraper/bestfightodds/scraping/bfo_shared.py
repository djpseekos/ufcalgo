from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

# ----------------------- Name & date helpers -----------------------

DATE_RE  = re.compile(r"\b([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})\b")
ORDINALS = re.compile(r"(st|nd|rd|th)\b", re.IGNORECASE)

def norm_name(s: str) -> str:
    s = html.unescape(str(s or ""))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def to_iso_date(s: str):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None

def parse_date_text(text: str):
    """
    Parse strings like 'Aug 17th 2025' or 'October 5th 2025' to a date.
    """
    if not text:
        return None
    m = DATE_RE.search(text)
    if not m:
        return None
    cleaned = ORDINALS.sub("", m.group(1))
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            pass
    return None

def within_tol(a, b, tol_days: int) -> bool:
    """Inclusive tolerance: abs(delta) <= tol_days."""
    if not a or not b:
        return False
    return abs((a - b).days) <= tol_days

# ----------------------- Odds parsing helpers ---------------------

def read_odds_from_row(row) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (open, close_low, close_high) strings from a single fighter row.
    If only one closing number exists, mirror it to the other; open-only is allowed.
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
    Pull the grey per-row date cell (e.g., <td class='item-non-mobile'>Aug 17th 2025</td>)
    and parse it as a date.
    """
    td = row.select_one("td.item-non-mobile")
    if not td:
        return None
    return parse_date_text(td.get_text(" ", strip=True))

# ----------------------- Page scanning logic ----------------------

def iter_event_blocks(html_text: str):
    """
    Yield (header_text, header_date, [main_rows]) for each event block.
    - header_date may be None if the header lacks a date (we'll fallback to row date).
    - main_rows is the list of 'tr.main-row' rows until the next header (usually 2).
    """
    soup = BeautifulSoup(html_text, "lxml")

    # Find the odds history table
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
            header_text = tr.get_text(" ", strip=True)
            header_date = parse_date_text(header_text)  # may be None
            # Gather following rows until next header
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