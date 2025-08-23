from __future__ import annotations
import argparse, re
from typing import Optional, List, Dict
import pandas as pd

from .common.http import get_html, HttpError
from .common.parse import soup
from .common.ids import fight_id_from_url, fighter_id_from_url
from .common.io import load_csv, upsert_csv, update_manifest

EVENTS_CSV = "data/curated/events.csv"
FIGHTS_CSV = "data/curated/fights.csv"

def _clean(txt: str) -> str:
    return re.sub(r"\s+", " ", (txt or "").strip())

def _parse_event_fights(event_id: str, event_url: str) -> List[Dict]:
    """
    Parse a UFCStats event page into fight rows.
    """
    html = get_html(event_url, cache_key=f"event_{event_id}")
    doc = soup(html)
    out: List[Dict] = []

    # UFCStats event pages have a main table with bouts.
    # Be robust: scan all rows that contain a fight-details link.
    for a in doc.select("a[href*='/fight-details/']"):
        href_fight = a.get("href", "").strip()
        if not href_fight:
            continue
        tr = a.find_parent("tr")
        if not tr:
            continue

        fight_id = fight_id_from_url(href_fight)

        # Grab fighter anchor tags in this row (two fighter-details links)
        fighter_links = tr.select("a[href*='/fighter-details/']")
        if len(fighter_links) < 2:
            # Some pages split across nested rows; try the next row too
            sibling = tr.find_next_sibling("tr")
            if sibling:
                fighter_links = sibling.select("a[href*='/fighter-details/']")

        if len(fighter_links) >= 2:
            r_link, b_link = fighter_links[0], fighter_links[1]
            r_name = _clean(r_link.get_text(" ", strip=True))
            b_name = _clean(b_link.get_text(" ", strip=True))
            r_fid = fighter_id_from_url(r_link.get("href", ""))
            b_fid = fighter_id_from_url(b_link.get("href", ""))
        else:
            # Fallback: skip if we can't see both fighters
            r_name = b_name = r_fid = b_fid = None

        # Cells often contain method, round/time, weight class, etc.
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        text_row = " | ".join(tds)

        # Heuristics
        method = None
        is_title = 1 if re.search(r"\btitle\b", text_row, re.I) else 0
        judge_scores = None
        weight_class = None
        scheduled_rounds = None
        end_round = None
        end_time_sec = None
        winner_corner = None

        # Weight class (look for strings like "Lightweight", "Welterweight")
        wc = re.search(r"(Strawweight|Flyweight|Bantamweight|Featherweight|Lightweight|Welterweight|Middleweight|Light Heavyweight|Heavyweight|Catch Weight|Women.+?)", text_row, re.I)
        if wc:
            weight_class = wc.group(1)

        # Method + round/time (e.g., "KO/TKO", "Submission", "Decision (Unanimous)")
        m_method = re.search(r"(KO/TKO|Submission|Decision.*?|DQ|No Contest|TKO|KO)", text_row, re.I)
        if m_method:
            method = m_method.group(1)

        # End round/time like "3 4:12" or "5 0:32"
        m_end = re.search(r"\b(\d+)\s+(\d{1,2}:\d{2})\b", text_row)
        if m_end:
            end_round = int(m_end.group(1))
            mm, ss = m_end.group(2).split(":")
            end_time_sec = int(mm) * 60 + int(ss)

        # Scheduled rounds often appear as "3 Rnd" / "5 Rnd"
        m_sched = re.search(r"\b(3|5)\s*Rnd", text_row, re.I)
        if m_sched:
            scheduled_rounds = int(m_sched.group(1))

        # Judge scores present? (keep the raw string if we see "Dec")
        if method and method.lower().startswith("decision"):
            judge_scores = text_row

        # Winner corner: try to detect "win" marker near names; fallback unknown
        # Some pages mark winners with a "W/L" column — too brittle to parse reliably here.
        # We'll leave winner_corner=None (optional) and infer later if needed.
        row = {
            "fight_id": fight_id,
            "event_id": event_id,
            "r_fighter_id": r_fid,
            "b_fighter_id": b_fid,
            "r_fighter_name": r_name,
            "b_fighter_name": b_name,
            "weight_class": weight_class,
            "scheduled_rounds": scheduled_rounds,
            "method": method,
            "end_round": end_round,
            "end_time_sec": end_time_sec,
            "is_title": is_title,
            "judge_scores": judge_scores,
            "winner_corner": winner_corner,
            "fight_url": href_fight,
        }
        out.append(row)

    # Dedup by fight_id
    seen = {}
    for r in out:
        seen[r["fight_id"]] = r
    return list(seen.values())

def scrape_event_fights(limit_events: Optional[int] = None) -> pd.DataFrame:
    events = load_csv(EVENTS_CSV)
    if events.empty:
        raise SystemExit("events.csv not found or empty — run events scraper first.")
    events = events.sort_values("date", ascending=False)
    if limit_events:
        events = events.head(limit_events)

    all_rows: list[dict] = []
    for _, ev in events.iterrows():
        rows = _parse_event_fights(ev["event_id"], ev["event_url"] if "event_url" in ev else f"http://www.ufcstats.com/event-details/{ev['event_id']}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    cols = [
        "fight_id","event_id","r_fighter_id","b_fighter_id","r_fighter_name","b_fighter_name",
        "weight_class","scheduled_rounds","method","end_round","end_time_sec",
        "is_title","judge_scores","winner_corner","fight_url"
    ]
    df = df.reindex(columns=cols)
    return df

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scrape UFCStats fights for each event → fights.csv")
    ap.add_argument("--limit-events", type=int, default=None, help="Only process N newest events")
    ap.add_argument("--out", default=FIGHTS_CSV)
    args = ap.parse_args(argv)

    df = scrape_event_fights(limit_events=args.limit_events)
    if df.empty:
        print("[fights] parsed 0 rows (check site structure).")
        return 0

    upsert_csv(df, args.out, keys=["fight_id"])
    update_manifest("fights.csv", rows=len(df))
    print(f"[fights] wrote {len(df)} rows → {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())