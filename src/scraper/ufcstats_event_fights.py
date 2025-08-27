# src/scraper/event_fights.py
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Iterable

import pandas as pd

from .common import http as http_common
from .common.parse import soup
from .common.ids import fighter_id_from_url
from .common.io import load_csv, upsert_csv, update_manifest

# Slightly quicker network profile (http_common still throttles per host)
http_common._cfg.rate_per_sec = 4.0
http_common._cfg.retries = 2
http_common._cfg.backoff = 0.3

EVENTS_CSV        = "data/curated/events.csv"
FIGHTS_EVENT_CSV  = "data/curated/fights_event.csv"   # separate file (do not override fights.csv)

# Fight id from data-link or onclick="doNav('/fight-details/<ID>')"
_FIGHTID_RX = re.compile(r"/fight-details/([^'\"/?#]+)", re.IGNORECASE)


def _fight_id_from_attrs(tr) -> Optional[str]:
    for attr in ("data-link", "onclick"):
        val = (tr.get(attr) or "").strip()
        if not val:
            continue
        m = _FIGHTID_RX.search(val)
        if m:
            return m.group(1)
    return None


def _clean(txt: str | None) -> str | None:
    if txt is None:
        return None
    return re.sub(r"\s+", " ", txt.strip())


def _infer_weight_class(cells: List) -> Optional[str]:
    """Prefer the last cell that contains 'weight' or 'women'."""
    candidates = []
    for td in cells:
        txt = _clean(td.get_text(" ", strip=True)) or ""
        low = txt.lower()
        if "weight" in low or "women" in low:
            candidates.append(txt)
    if candidates:
        return candidates[-1]

    # broader fallback
    for td in reversed(cells):
        txt = _clean(td.get_text(" ", strip=True)) or ""
        low = txt.lower()
        if any(w in low for w in ("weight", "feather", "light", "heavy", "bantam", "fly", "straw")):
            return txt or None
    return None


def _parse_event_fights(event_id: str, event_url: str) -> List[Dict]:
    """
    Parse one UFCStats event page into fight rows.

    Winner logic (event summary page only):
      - r_fighter_name is always the winner if there was a winner → "R"
      - If result cell text has 'draw' → "D"
      - If result cell text has 'nc' / 'no contest' → "NC"
    """
    html = http_common.get_html(
        event_url,
        cache_key=f"event_{event_id}",
        ttl_hours=24,
        timeout=6,
        headers={"Referer": "https://www.ufcstats.com/"},
    )
    doc = soup(html)
    out: List[Dict] = []

    rows = doc.select("table.b-fight-details__table tr.b-fight-details__table-row")
    if not rows:
        rows = doc.select("table.b-fight-details__table tr")

    for tr in rows:
        fid = _fight_id_from_attrs(tr)
        if not fid:
            a = tr.select_one("a[href*='/fight-details/']")
            if a:
                m = _FIGHTID_RX.search(a.get("href", ""))
                if m:
                    fid = m.group(1)
        if not fid:
            continue

        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        result_td = tds[0]

        # Fighter anchors: first is winner, second is loser
        fighters = tr.select("a[href*='/fighter-details/']")
        if len(fighters) < 2:
            continue
        r_a, b_a = fighters[0], fighters[1]

        r_name = _clean(r_a.get_text(" ", strip=True))
        b_name = _clean(b_a.get_text(" ", strip=True))
        r_fid  = fighter_id_from_url(r_a.get("href", ""))
        b_fid  = fighter_id_from_url(b_a.get("href", ""))

        # Winner detection simplified
        res_txt = (_clean(result_td.get_text(" ", strip=True)) or "").lower()
        if "draw" in res_txt:
            winner_corner = "D"
        elif "nc" in res_txt or "no contest" in res_txt:
            winner_corner = "NC"
        else:
            winner_corner = "R"  # winner always listed first

        weight_class = _infer_weight_class(tds)
        fight_url = f"https://www.ufcstats.com/fight-details/{fid}"

        out.append({
            "fight_id": fid,
            "event_id": event_id,
            "r_fighter_id": r_fid,
            "b_fighter_id": b_fid,
            "r_fighter_name": r_name,
            "b_fighter_name": b_name,
            "weight_class": weight_class,
            "winner_corner": winner_corner,
            "fight_url": fight_url,
        })

    return list({r["fight_id"]: r for r in out}.values())

def _iter_events(events_df: pd.DataFrame, limit_events: Optional[int]) -> Iterable[tuple[str, str]]:
    df = events_df.copy()
    if "date" in df.columns:
        df = df.sort_values("date", ascending=False)
    if limit_events:
        df = df.head(limit_events)

    for _, ev in df.iterrows():
        event_id = str(ev["event_id"])
        event_url = ev.get("event_url") or f"https://www.ufcstats.com/event-details/{event_id}"
        yield (event_id, event_url)


def scrape_event_fights(limit_events: Optional[int] = None, workers: int = 8) -> pd.DataFrame:
    events = load_csv(EVENTS_CSV)
    if events.empty:
        raise SystemExit("events.csv not found or empty — run events scraper first.")

    jobs = list(_iter_events(events, limit_events))
    if not jobs:
        return pd.DataFrame()

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(_parse_event_fights, eid, url): (eid, url) for (eid, url) in jobs}
        for fut in as_completed(futures):
            eid, url = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:
                print(f"[event_fights] WARNING: failed event {eid} ({url}): {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    cols = [
        "fight_id","event_id",
        "r_fighter_id","b_fighter_id","r_fighter_name","b_fighter_name",
        "weight_class","winner_corner","fight_url"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].sort_values(["event_id","fight_id"]).drop_duplicates("fight_id", keep="last")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scrape UFCStats fights per event → fights_event.csv (no override)")
    ap.add_argument("--limit-events", type=int, default=None, help="Only process N newest events")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent workers to parse events")
    ap.add_argument("--out", default=FIGHTS_EVENT_CSV, help="Output CSV path (upsert on fight_id)")
    args = ap.parse_args(argv)

    df = scrape_event_fights(limit_events=args.limit_events, workers=args.workers)
    if df.empty:
        print("[event_fights] parsed 0 rows (check selectors or cached HTML).")
        return 0

    upsert_csv(df, args.out, keys=["fight_id"])
    update_manifest("fights_event.csv", rows=len(df))
    print(f"[event_fights] wrote {len(df)} rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())