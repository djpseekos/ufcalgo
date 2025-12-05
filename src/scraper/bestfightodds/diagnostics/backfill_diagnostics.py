# src/scraper/bestfightodds/diagnostics/backfill_diagnostics.py
from __future__ import annotations

import argparse
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from src.scraper.common.http import get_html, set_network_profile, set_identity, HttpError
from src.scraper.bestfightodds.scraping.backfill_odds import (
    norm_name, to_iso_date, parse_date_text, within_tol,
    looks_like_cloudflare,
)

BASE_HEADERS = {
    "Referer": "https://www.bestfightodds.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

def build_headers(user_agent: Optional[str], cookie: Optional[str]) -> Dict[str, str]:
    h = dict(BASE_HEADERS)
    if user_agent:
        h["User-Agent"] = user_agent
    if cookie:
        h["Cookie"] = cookie
    return h

def build_odds_map(odds_df: pd.DataFrame) -> Dict[str, set]:
    mp: Dict[str, set] = {}
    for _, r in odds_df.iterrows():
        n = norm_name(r.get("fighter_name", ""))
        d = to_iso_date(r.get("date_iso", ""))
        if n and d:
            mp.setdefault(n, set()).add(d)
    return mp

def build_link_map(links_df: pd.DataFrame) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    for _, r in links_df.iterrows():
        n = norm_name(r.get("fighter_name", ""))
        u = r.get("fighter_link_bfo", "")
        if n and u and n not in mp:
            mp[n] = u
    return mp

def iter_event_blocks(html: str):
    soup = BeautifulSoup(html, "lxml")
    table = None
    for t in soup.select("table.team-stats-table"):
        if "odds history for" in (t.get("summary") or "").lower():
            table = t; break
    if table is None:
        table = soup.find("table")
        if table is None:
            return
    tbody = table.find("tbody") or table
    rows = tbody.find_all("tr", recursive=False)
    i = 0
    while i < len(rows):
        tr = rows[i]
        classes = set(tr.get("class") or [])
        if "event-header" in classes:
            header_text = tr.get_text(" ", strip=True)
            header_date = parse_date_text(header_text)
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

def main():
    ap = argparse.ArgumentParser(description="Deep diagnostics for BFO backfill matching.")
    ap.add_argument("--mode", choices=["random", "first"], default="random")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--tolerance-days", type=int, default=1)
    ap.add_argument("--fights", default="data/curated/fights.csv")
    ap.add_argument("--odds", default="data/curated/fighters_odds_bfo_iso.csv")
    ap.add_argument("--links", default="data/curated/fighter_bfo_links.csv")
    ap.add_argument("--rate-per-sec", type=float, default=8.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=0.25)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--user-agent", type=str, default=os.environ.get("BFO_UA"))
    ap.add_argument("--cookie", type=str, default=os.environ.get("BFO_COOKIE"))
    args = ap.parse_args()

    set_network_profile(rate_per_sec=args.rate_per_sec, retries=args.retries, backoff=args.backoff, timeout=args.timeout)
    set_identity(user_agent=args.user_agent)

    fights = pd.read_csv(args.fights, dtype=str).fillna("")
    odds   = pd.read_csv(args.odds, dtype=str).fillna("")
    links  = pd.read_csv(args.links, dtype=str).fillna("")

    odds_map = build_odds_map(odds)
    link_map = build_link_map(links)

    # find one-sided fights
    candidates: List[Dict] = []
    for _, fr in fights.iterrows():
        date_iso = str(fr.get("date",""))
        d = to_iso_date(date_iso)
        if not d:
            continue
        r = str(fr.get("r_fighter_name","")); rn = norm_name(r)
        b = str(fr.get("b_fighter_name","")); bn = norm_name(b)
        r_has = any(within_tol(d2, d, args.tolerance_days) for d2 in odds_map.get(rn, set()))
        b_has = any(within_tol(d2, d, args.tolerance_days) for d2 in odds_map.get(bn, set()))
        if r_has ^ b_has:
            matched = r if r_has else b
            missing = b if r_has else r
            matched_norm = rn if r_has else bn
            url = link_map.get(matched_norm)
            if url:
                candidates.append({
                    "fight_id": fr.get("fight_id",""),
                    "event_id": fr.get("event_id",""),
                    "date_iso": date_iso,
                    "matched_name": matched,
                    "missing_name": missing,
                    "url": url,
                    "side": "red" if r_has else "blue",
                })

    print(f"Total fights: {len(fights)} | One-sided fights: {len(candidates)}\n")

    if not candidates:
        return

    if args.mode == "random":
        random.shuffle(candidates)
    take = candidates[:max(1, args.samples)]

    for i, c in enumerate(take, start=1):
        print("================ BACKFILL DIAGNOSTIC CASE ================")
        print(f"[{i}] fight_id={c['fight_id']}  event_id={c['event_id']}")
        print(f"Target ISO date : {c['date_iso']} (±{args.tolerance_days}d)")
        print(f"Matched fighter : {c['matched_name']}  (side: {c['side']})")
        print(f"Missing fighter : {c['missing_name']}")
        print(f"BFO URL         : {c['url']}")

        headers = build_headers(args.user_agent, args.cookie)

        try:
            html = get_html(c["url"], cache_key=f"diag_{norm_name(c['matched_name'])}", ttl_hours=0, headers=headers)
        except HttpError as e:
            print(f"HTTP ERROR: {e}")
            continue
        except Exception as e:
            print(f"FETCH ERROR: {e}")
            continue

        if looks_like_cloudflare(html):
            print("ANTI-BOT PAGE DETECTED (Cloudflare). Matching cannot proceed.")
            print("Hint: supply --cookie 'cf_clearance=...; __cf_bm=...' and a modern --user-agent.")
            continue

        print(f"HTML bytes      : {len(html)}")

        # Scan blocks and compute deltas
        target = to_iso_date(c["date_iso"])
        soup = BeautifulSoup(html, "lxml")
        table = soup.select_one("table.team-stats-table") or soup.find("table")
        print("Table present   :", bool(table))

        blocks = list(iter_event_blocks(html))
        print("Blocks found    :", len(blocks))

        deltas: List[Tuple[int,str,str]] = []  # (abs_delta, header_text, eff_date)
        matched_block = None
        for (htext, hdate, mains) in blocks:
            if len(mains) < 2:
                continue
            eff = hdate or parse_date_text(mains[0].get_text(" ", strip=True))
            if not eff and len(mains) > 1:
                eff = parse_date_text(mains[1].get_text(" ", strip=True))
            if not eff:
                continue
            delta = abs((eff - target).days)
            deltas.append((delta, htext, str(eff)))
            if delta <= args.tolerance_days and matched_block is None:
                matched_block = (mains[0], mains[1], eff, htext)

        if matched_block:
            print("MATCH: block within tolerance found.")
            print("Effective date  :", matched_block[2])
            print("Header snippet  :", matched_block[3][:120].replace("\n"," "))
            # Peek at odds spans presence in opponent row
            opp_row = matched_block[1]
            has_o0 = bool(opp_row.select_one("span#oID0"))
            has_o1 = bool(opp_row.select_one("span#oID1"))
            has_o2 = bool(opp_row.select_one("span#oID2"))
            print(f"Opponent spans  : oID0={has_o0} oID1={has_o1} oID2={has_o2}")
        else:
            print("NO MATCH within tolerance.")
            deltas.sort(key=lambda x: x[0])
            print("Closest headers :")
            for dd, ht, eff in deltas[:5]:
                print(f"  Δ={dd:>2}  eff={eff}  header='{ht[:80].replace(chr(10),' ')}'")

        print("=====================================================\n")

if __name__ == "__main__":
    main()