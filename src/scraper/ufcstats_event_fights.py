# src/scraper/ufcstats_event_fights.py
from __future__ import annotations

import argparse
import re
from typing import Optional, List, Dict

import pandas as pd

from .common.http import get_html
from .common.parse import soup
from .common.ids import fighter_id_from_url
from .common.io import load_csv, upsert_csv, update_manifest

EVENTS_CSV = "data/curated/events.csv"
FIGHTS_CSV = "data/curated/fights.csv"

# Grab fight id from either data-link=".../fight-details/<ID>" or onclick="doNav('/fight-details/<ID>')"
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

def _parse_event_fights(event_id: str, event_url: str) -> List[Dict]:
    """
    Parse one event page. Structure observed:

      <table class="b-fight-details__table">
        <tbody>
          <tr class="b-fight-details__table-row" data-link="http://ufcstats.com/fight-details/<FID>">
            <td> WIN/LOSS </td>
            <td><a href="/fighter-details/<RID>">Red Fighter</a></td>
            <td><a href="/fighter-details/<BID>">Blue Fighter</a></td>
            ... (KD/STR/TD/SUB)
            <td>Weight Class</td>
          </tr>
          ...
        </tbody>
      </table>
    """
    html = get_html(event_url, cache_key=f"event_{event_id}")
    doc = soup(html)
    out: List[Dict] = []

    # Be specific to the main bout table; allow for minor class variations
    rows = doc.select("table.b-fight-details__table tbody tr.b-fight-details__table-row")
    if not rows:
        # fallback: some pages omit tbody or class on tr
        rows = doc.select("table.b-fight-details__table tr")

    for tr in rows:
        fid = _fight_id_from_attrs(tr)
        if not fid:
            continue

        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        # Winner marker (first td). If it says WIN, assume left (red) side.
        wl_txt = (tds[0].get_text(" ", strip=True) or "").lower()
        if "win" in wl_txt:
            winner_corner = "R"
        elif "loss" in wl_txt:
            winner_corner = "B"
        else:
            winner_corner = None

        # Fighter anchors: 2nd td = red, 3rd td = blue
        r_a = tds[1].select_one("a[href*='/fighter-details/']")
        b_a = tds[2].select_one("a[href*='/fighter-details/']")

        r_name = _clean(r_a.get_text(" ", strip=True)) if r_a else None
        b_name = _clean(b_a.get_text(" ", strip=True)) if b_a else None
        r_fid  = fighter_id_from_url(r_a.get("href", "")) if r_a else None
        b_fid  = fighter_id_from_url(b_a.get("href", "")) if b_a else None

        # Weight class is typically the last cell
        weight_class = _clean(tds[-1].get_text(" ", strip=True)) if tds else None

        fight_url = f"https://www.ufcstats.com/fight-details/{fid}"

        out.append({
            "fight_id": fid,
            "event_id": event_id,
            "r_fighter_id": r_fid,
            "b_fighter_id": b_fid,
            "r_fighter_name": r_name,
            "b_fighter_name": b_name,
            "weight_class": weight_class,
            # defer these to the fight-details/rounds scraper
            "scheduled_rounds": None,
            "method": None,
            "end_round": None,
            "end_time_sec": None,
            "is_title": None,
            "judge_scores": None,
            "winner_corner": winner_corner,
            "fight_url": fight_url,
        })

    # Dedup by fight_id
    return list({r["fight_id"]: r for r in out}.values())

def scrape_event_fights(limit_events: Optional[int] = None) -> pd.DataFrame:
    events = load_csv(EVENTS_CSV)
    if events.empty:
        raise SystemExit("events.csv not found or empty — run events scraper first.")
    events = events.sort_values("date", ascending=False)
    if limit_events:
        events = events.head(limit_events)

    all_rows: list[dict] = []
    for _, ev in events.iterrows():
        event_id = ev["event_id"]
        # events.csv may not include event_url; reconstruct if missing
        event_url = ev.get("event_url") or f"https://www.ufcstats.com/event-details/{event_id}"
        rows = _parse_event_fights(event_id, event_url)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    cols = [
        "fight_id","event_id",
        "r_fighter_id","b_fighter_id","r_fighter_name","b_fighter_name",
        "weight_class","scheduled_rounds","method","end_round","end_time_sec",
        "is_title","judge_scores","winner_corner","fight_url"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    return df

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scrape UFCStats fights for each event → fights.csv")
    ap.add_argument("--limit-events", type=int, default=None, help="Only process N newest events")
    ap.add_argument("--out", default=FIGHTS_CSV)
    args = ap.parse_args(argv)

    df = scrape_event_fights(limit_events=args.limit_events)
    if df.empty:
        print("[fights] parsed 0 rows (check selectors or cached HTML).")
        return 0

    upsert_csv(df, args.out, keys=["fight_id"])
    update_manifest("fights.csv", rows=len(df))
    print(f"[fights] wrote {len(df)} rows → {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())