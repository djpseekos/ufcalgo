# src/scraper/common/ids.py
from __future__ import annotations
import re
from urllib.parse import urlparse
from typing import Optional
import unicodedata

_UFC_EVENT_RX = re.compile(r"/event-details/([^/?#]+)", re.IGNORECASE)
_UFC_EVENT_FALLBACK_RX = re.compile(r"/event/([^/?#]+)", re.IGNORECASE)
_UFC_FIGHT_RX = re.compile(r"/fight-details/([^/?#]+)", re.IGNORECASE)
_UFC_FIGHTER_RX = re.compile(r"/fighter-details/([^/?#]+)", re.IGNORECASE)

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9\-]+", "-", s.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def event_id_from_url(url: str) -> str:
    m = _UFC_EVENT_RX.search(url)
    if m:
        return m.group(1).lower()
    m2 = _UFC_EVENT_FALLBACK_RX.search(url)
    if m2:
        return m2.group(1).lower()
    return _norm(url)

def fight_id_from_url(url: str) -> str:
    m = _UFC_FIGHT_RX.search(url)
    return m.group(1).lower() if m else _norm(url)

def fighter_id_from_url(url: str) -> str:
    m = _UFC_FIGHTER_RX.search(url)
    return m.group(1).lower() if m else _norm(url)

def safe_slug(s: str) -> str:
    return _norm(s)