# src/scraper/common/parse.py
from __future__ import annotations
import re, datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup

def soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")

def _hdr(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt or "").strip().lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt).strip("_")
    return txt

def table_to_rows(tbl) -> List[Dict[str, str]]:
    rows = []
    headers = []
    thead = tbl.find("thead")
    if thead:
        headers = [_hdr(th.get_text(" ", strip=True)) for th in thead.find_all("th")]
    else:
        # try first row as header
        first = tbl.find("tr")
        if first:
            headers = [_hdr(th.get_text(" ", strip=True)) for th in first.find_all(["th","td"])]
    for tr in tbl.find_all("tr"):
        cells = tr.find_all(["td"])
        if not cells:
            continue
        texts = [c.get_text(" ", strip=True) for c in cells]
        if not headers or len(headers) != len(texts):
            # pad or fallback to c0,c1...
            hdrs = headers if headers and len(headers) == len(texts) else [f"c{i}" for i in range(len(texts))]
        else:
            hdrs = headers
        rows.append({h: t for h, t in zip(hdrs, texts)})
    return rows

def to_int(x: str) -> int | None:
    if x is None: return None
    x = re.sub(r"[^\d\-]", "", x)
    if x == "" or x == "-": return None
    try: return int(x)
    except: return None

def to_float(x: str) -> float | None:
    if x is None: return None
    x = x.replace(",", "")
    try: return float(x)
    except: return None

def to_date(x: str) -> str | None:
    # returns ISO yyyy-mm-dd if recognizable
    x = (x or "").strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(x, fmt).date().isoformat()
        except: pass
    return None

def time_str_to_seconds(x: str) -> int | None:
    # "4:32" -> 272 ; "12:05" -> 725
    x = (x or "").strip()
    m = re.match(r"^(\d+):(\d{2})$", x)
    if not m: return None
    return int(m.group(1)) * 60 + int(m.group(2))

def height_to_cm(x: str) -> float | None:
    # "6' 2\"" or 6'2" -> cm
    if not x: return None
    m = re.match(r"^\s*(\d+)'\s*(\d+)\s*\"", x)
    if not m: return None
    feet, inches = int(m.group(1)), int(m.group(2))
    return round((feet * 12 + inches) * 2.54, 1)

def reach_to_cm(x: str) -> float | None:
    # '74"' -> cm
    if not x: return None
    m = re.match(r'^\s*(\d+)\s*"?\s*$', x)
    if not m: return None
    inches = int(m.group(1))
    return round(inches * 2.54, 1)