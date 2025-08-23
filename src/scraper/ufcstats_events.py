# src/scraper/ufcstats_events.py
from __future__ import annotations

import os
import re
import sys
import argparse
from datetime import date
from typing import List, Dict, Optional

import pandas as pd

from .common.http import get_html, HttpError, set_identity
from .common.parse import soup, to_date
from .common.ids import event_id_from_url
from .common.io import upsert_csv, update_manifest

# Default index page that lists ALL completed events (single page)
DEFAULT_EVENTS_INDEX = "http://www.ufcstats.com/statistics/events/completed?page=all"

def _split_location(loc_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    UFCStats usually shows 'City, State/Province, Country' OR 'City, Country'.
    We'll return (city, country) by taking the first token as city,
    and the last token as country (trimmed).
    """
    if not loc_text:
        return None, None
    parts = [p.strip() for p in loc_text.split(",") if p.strip()]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]

def _parse_events_from_index(html: str) -> List[Dict]:
    """
    Parse the completed events table. Each row contains an <a> to the event page,
    plus date and location cells.
    """
    doc = soup(html)
    rows: List[Dict] = []

    # The index page contains tables; we'll scan for links containing "/event/"
    for a in doc.select("a[href*='/event/']"):
        href = a.get("href", "")
        if not href:
            continue
        # Expect a parent row with sibling cells for date/location
        tr = a.find_parent("tr")
        if not tr:
            continue

        # Extract columns by text; UFCStats typically has Date in a <td> near the link
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        # Heuristic: last two cells are Date and Location on the index page.
        # We'll also search for recognizable date format.
        parsed_date = None
        parsed_loc = None

        # Try to pick a date-like cell
        for cell in tds:
            iso = to_date(cell)
            if iso:
                parsed_date = iso
                break

        # Location: pick the longest non-empty cell that isn't the event name and isn't the date
        candidates = [c for c in tds if c and c != a.get_text(" ", strip=True) and to_date(c) is None]
        if candidates:
            parsed_loc = max(candidates, key=len)

        event_url = href.strip()
        eid = event_id_from_url(event_url)
        city, country = _split_location(parsed_loc or "")

        rows.append({
            "event_id": eid,
            "event_url": event_url,
            "date": parsed_date,
            "city": city,
            "country": country,
            "venue": None,       # filled after visiting event page
            "is_apex": 0,        # computed after venue extraction
        })

    # De-duplicate by event_id (index page can occasionally repeat header rows)
    dedup: Dict[str, Dict] = {}
    for r in rows:
        dedup[r["event_id"]] = r
    return list(dedup.values())

def _parse_event_details(html: str) -> dict:
    """
    Visit an event page and try to extract Venue and Location (to correct city/country if needed).
    Returns dict with possible keys: venue, location_text
    """
    doc = soup(html)
    info = {"venue": None, "location_text": None}

    # UFCStats event page has a box list with entries like 'Location: …'
    # We'll search all list items for 'Location:' and any 'Arena:'/'Venue:' labels.
    for li in doc.select("ul.b-list__box-list li"):
        text = li.get_text(" ", strip=True)
        if not text or ":" not in text:
            continue
        label, value = [t.strip() for t in text.split(":", 1)]
        label_l = label.lower()
        if label_l == "location":
            info["location_text"] = value
        elif label_l in ("arena", "venue"):
            info["venue"] = value

    # Fallback: sometimes venue might be embedded elsewhere; do a greedy search
    if not info["venue"]:
        # Look for any element with 'Apex' or 'Center' etc. near the top info box
        maybe = doc.find(text=re.compile(r"Apex|Center|Arena|Stadium", re.I))
        if maybe:
            info["venue"] = str(maybe).strip()

    return info

def _enrich_with_details(rows: List[Dict]) -> List[Dict]:
    enriched = []
    for r in rows:
        try:
            html = get_html(r["event_url"], cache_key=f"event_{r['event_id']}")
            details = _parse_event_details(html)
        except HttpError:
            details = {"venue": None, "location_text": None}

        # If event page provided a location, re-split into city/country
        if details.get("location_text"):
            city, country = _split_location(details["location_text"])
            if city:
                r["city"] = city
            if country:
                r["country"] = country

        # Venue and is_apex derivation
        venue = details.get("venue")
        if venue:
            r["venue"] = venue
            r["is_apex"] = 1 if re.search(r"\bapex\b", venue, re.I) else 0
        else:
            # If we can’t find a venue, keep is_apex at 0 (context_enrich can revisit later)
            r["venue"] = None
            r["is_apex"] = 0

        enriched.append(r)
    return enriched

def scrape_events(index_url: str = DEFAULT_EVENTS_INDEX,
                  since: Optional[str] = None,
                  limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape the events index and visit each event for details.
    Filters by --since (YYYY-MM-DD) and --limit (number of events kept, newest-first by date).
    """
    index_html = get_html(index_url, cache_key="events_index_all")
    base_rows = _parse_events_from_index(index_html)

    # Filter by since date if provided
    if since:
        base_rows = [r for r in base_rows if r.get("date") and r["date"] >= since]

    # Sort newest first by date
    base_rows = sorted(base_rows, key=lambda r: (r.get("date") or "0000-00-00"), reverse=True)

    # Apply limit after sorting
    if limit is not None:
        base_rows = base_rows[:int(limit)]

    # Enrich with per-event details (venue, corrected location)
    rows = _enrich_with_details(base_rows)

    # Final tidy DataFrame for output schema
    df = pd.DataFrame(rows, columns=[
        "event_id", "date", "city", "country", "venue", "is_apex"
    ])
    return df

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape UFCStats completed events → events.csv")
    parser.add_argument("--index-url", default=DEFAULT_EVENTS_INDEX, help="UFCStats completed events index")
    parser.add_argument("--since", default=None, help="Min event date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of events (newest first)")
    parser.add_argument("--out", default="data/curated/events.csv", help="Output CSV path")
    parser.add_argument("--contact", default=None, help="Contact email for HTTP headers (polite scraping)")
    args = parser.parse_args(argv)

    if args.contact:
        set_identity(contact_email=args.contact)

    df = scrape_events(index_url=args.index_url, since=args.since, limit=args.limit)

    # Upsert by event_id so repeated runs update any improved parsing
    upsert_csv(df, args.out, keys=["event_id"])
    update_manifest("events.csv", rows=len(df))

    # Console summary
    print(f"[events] wrote {len(df)} rows → {args.out}")
    if df["is_apex"].sum():
        print(f"[events] apex-flagged: {int(df['is_apex'].sum())}")
    missing_venue = df["venue"].isna().sum()
    if missing_venue:
        print(f"[events] missing venue: {missing_venue} (can be filled later by context_enrich)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())