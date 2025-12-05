from __future__ import annotations

import argparse
import csv
import html
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from src.scraper.common.http import (
    get_html,
    set_network_profile,
    set_identity,
)

# --------------------------- Config / Paths ---------------------------

DEFAULT_FIGHTS = "data/curated/fights.csv"
DEFAULT_ODDS   = "data/curated/fighters_odds_bfo_iso.csv"
DEFAULT_LINKS  = "data/curated/fighter_bfo_links.csv"
DEFAULT_OUT    = "data/diagnostics/bfo_name_mismatches.csv"

BFO_HEADERS = {
    "Referer": "https://www.bestfightodds.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

DATE_RE   = re.compile(r"\b([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})\b")
ORDINALS  = re.compile(r"(st|nd|rd|th)\b", re.IGNORECASE)
TOKEN_RE  = re.compile(r"[a-z0-9']+")

# --------------------------- Small utilities -------------------------

def norm_name(s: str) -> str:
    """ASCII-fold, lowercase, keep letters/numbers/spaces/'-, collapse spaces."""
    s = html.unescape(str(s or ""))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    return TOKEN_RE.findall(norm_name(s))

def given_and_surnames(name: str) -> Tuple[str, List[str]]:
    """Heuristic: first token = given name; rest = surnames (can be multiple)."""
    t = tokens(name)
    if not t:
        return "", []
    if len(t) == 1:
        return t[0], [t[0]]
    return t[0], t[1:]

def surname_subset(a: str, b: str) -> bool:
    """True if the shorter surname set is a subset of the longer (handles double surnames)."""
    _, sa = given_and_surnames(a)
    _, sb = given_and_surnames(b)
    if not sa or not sb:
        return False
    A, B = set(sa), set(sb)
    return A.issubset(B) or B.issubset(A)

def levenshtein(a: str, b: str) -> int:
    """Tiny DP Levenshtein for short names (no external deps)."""
    a, b = norm_name(a), norm_name(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            ins = dp[j - 1] + 1
            dele = dp[j] + 1
            sub = prev + (ca != cb)
            prev, dp[j] = dp[j], min(ins, dele, sub)
    return dp[-1]

def parse_bfo_date(text: str) -> Optional[datetime.date]:
    """Extract 'Apr 3rd 2009' or 'October 5th 2025' from a header string."""
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
            continue
    return None

def to_iso_date(s: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None

def within_tol(d1: Optional[datetime.date], d2: Optional[datetime.date], tol_days: int) -> Tuple[bool, Optional[int]]:
    if not d1 or not d2:
        return (False, None)
    delta = (d1 - d2).days
    return (abs(delta) <= tol_days, delta if abs(delta) <= tol_days else None)

# --------------------------- Data structures -------------------------

@dataclass
class OneSidedFight:
    fight_id: str
    event_id: str
    date: str                      # ISO string from fights.csv
    matched_side: str              # 'red' or 'blue'
    matched_fighter: str
    db_opponent: str               # the other side in fights.csv

# --------------------------- Core builders ---------------------------

def build_odds_map(odds: pd.DataFrame) -> Dict[str, set]:
    """name -> set of ISO dates present in odds."""
    mp: Dict[str, set] = {}
    for _, r in odds.iterrows():
        n = norm_name(r.get("fighter_name", ""))
        d = to_iso_date(r.get("date_iso", ""))
        if n and d:
            mp.setdefault(n, set()).add(d)
    return mp

def build_link_map(links: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    normalized fighter_name -> {url:str, fighter_id:str}
    """
    out: Dict[str, Dict[str, str]] = {}
    for _, r in links.iterrows():
        n = norm_name(r.get("fighter_name", ""))
        u = r.get("fighter_link_bfo", "")
        fid = str(r.get("fighter_id", "") or "")
        if n and u and n not in out:
            out[n] = {"url": u, "fighter_id": fid}
    return out

def iter_one_sided_fights(
    fights: pd.DataFrame,
    odds_map: Dict[str, set],
    tol_days: int,
) -> Iterable[OneSidedFight]:
    for _, r in fights.iterrows():
        d_iso = to_iso_date(r.get("date", ""))
        rname = r.get("r_fighter_name", "")
        bname = r.get("b_fighter_name", "")
        rn, bn = norm_name(rname), norm_name(bname)

        r_match = any(within_tol(d_iso, od, tol_days)[0] for od in odds_map.get(rn, set()))
        b_match = any(within_tol(d_iso, od, tol_days)[0] for od in odds_map.get(bn, set()))

        if (r_match ^ b_match):  # exactly one
            if r_match:
                yield OneSidedFight(
                    fight_id=str(r.get("fight_id", "")),
                    event_id=str(r.get("event_id", "")),
                    date=str(r.get("date", "")),
                    matched_side="red",
                    matched_fighter=rname,
                    db_opponent=bname,
                )
            else:
                yield OneSidedFight(
                    fight_id=str(r.get("fight_id", "")),
                    event_id=str(r.get("event_id", "")),
                    date=str(r.get("date", "")),
                    matched_side="blue",
                    matched_fighter=bname,
                    db_opponent=rname,
                )

# --------------------------- BFO parsing -----------------------------

def cache_key_for_bfo_url(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    return f"bfo_fighter/{slug}"

def find_opponent_on_bfo_page(html: str, page_fighter_name: str, fight_date_iso: str, tol_days: int) -> Tuple[Optional[str], Optional[str], Optional[int], str]:
    """
    Return (opponent_name, opponent_url, offset_days, reason_if_fail)
    Uses header date ±tol and the 'header → two main-row' structure.
    """
    soup = BeautifulSoup(html, "lxml")

    # find the odds table
    table = None
    for t in soup.select("table.team-stats-table"):
        if "odds history for" in (t.get("summary") or "").lower():
            table = t
            break
    if table is None:
        # fallback: first table
        table = soup.find("table")
        if table is None:
            return (None, None, None, "NO_TABLE")

    tbody = table.find("tbody") or table
    target_date = to_iso_date(fight_date_iso)
    if not target_date:
        return (None, None, None, "BAD_TARGET_DATE")

    rows = tbody.find_all("tr", recursive=False)
    i = 0
    while i < len(rows):
        tr = rows[i]
        classes = set(tr.get("class") or [])

        if "event-header" in classes:
            header_text = tr.get_text(" ", strip=True)
            header_date = parse_bfo_date(header_text)
            ok, offset = within_tol(header_date, target_date, tol_days)

            # Collect the next two main rows (if they exist)
            # Regardless of ok flag, still advance i; but only inspect pair if ok.
            main_rows: List = []
            j = i + 1
            while j < len(rows) and len(main_rows) < 2:
                clj = set(rows[j].get("class") or [])
                if "main-row" in clj:
                    main_rows.append(rows[j])
                elif "event-header" in clj:
                    break
                j += 1

            # If header date within tolerance and we have a pair, inspect it
            if ok and len(main_rows) >= 2:
                # first main-row is page fighter row; second is opponent
                opp_row = main_rows[1]

                a = opp_row.select_one("th.oppcell a") or opp_row.select_one("a[href^='/fighters/']")
                if not a or not a.get("href"):
                    return (None, None, None, "OPPONENT_ANCHOR_MISSING")

                opp_name = a.get_text(strip=True)
                opp_url = "https://www.bestfightodds.com" + a.get("href")
                return (opp_name, opp_url, offset, "")

            i = j
            continue

        i += 1

    return (None, None, None, "NO_HEADER_DATE_MATCH")

# --------------------------- Driver / CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Find fights with only one matched odds row, pull BFO opponent, and compare names."
    )
    ap.add_argument("--fights", default=DEFAULT_FIGHTS)
    ap.add_argument("--odds", default=DEFAULT_ODDS)
    ap.add_argument("--links", default=DEFAULT_LINKS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tolerance-days", type=int, default=1)
    ap.add_argument("--rate-per-sec", type=float, default=10.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=0.25)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None, help="Stop after N one-sided fights (for testing)")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    set_network_profile(rate_per_sec=args.rate_per_sec, retries=args.retries, backoff=args.backoff, timeout=args.timeout)
    set_identity()

    fights = pd.read_csv(args.fights, dtype=str).fillna("")
    odds   = pd.read_csv(args.odds, dtype=str).fillna("")
    links  = pd.read_csv(args.links, dtype=str).fillna("")

    odds_map = build_odds_map(odds)
    link_map = build_link_map(links)

    one_sided = list(iter_one_sided_fights(fights, odds_map, tol_days=args.tolerance_days))
    if args.limit is not None:
        one_sided = one_sided[: max(0, int(args.limit))]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "fight_id","event_id","date",
        "matched_side","matched_fighter",
        "db_opponent","bfo_opponent","bfo_opponent_url",
        "offset_days",
        "name_equal_norm","surname_subset","lev_distance",
        "reason",
    ]

    written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for idx, item in enumerate(one_sided, start=1):
            mf_norm = norm_name(item.matched_fighter)
            link = link_map.get(mf_norm)
            if not link:
                w.writerow({
                    "fight_id": item.fight_id, "event_id": item.event_id, "date": item.date,
                    "matched_side": item.matched_side, "matched_fighter": item.matched_fighter,
                    "db_opponent": item.db_opponent,
                    "bfo_opponent": "", "bfo_opponent_url": "",
                    "offset_days": "",
                    "name_equal_norm": "", "surname_subset": "", "lev_distance": "",
                    "reason": "NO_LINK_FOR_MATCHED_FIGHTER",
                })
                continue

            try:
                html = get_html(link["url"], cache_key=cache_key_for_bfo_url(link["url"]), ttl_hours=24, headers=BFO_HEADERS)
            except Exception as e:
                w.writerow({
                    "fight_id": item.fight_id, "event_id": item.event_id, "date": item.date,
                    "matched_side": item.matched_side, "matched_fighter": item.matched_fighter,
                    "db_opponent": item.db_opponent,
                    "bfo_opponent": "", "bfo_opponent_url": "",
                    "offset_days": "", "name_equal_norm": "", "surname_subset": "", "lev_distance": "",
                    "reason": f"HTTP_ERROR:{e}",
                })
                continue

            opp_name, opp_url, off, reason = find_opponent_on_bfo_page(
                html=html,
                page_fighter_name=item.matched_fighter,
                fight_date_iso=item.date,
                tol_days=args.tolerance_days,
            )

            if not opp_name:
                w.writerow({
                    "fight_id": item.fight_id, "event_id": item.event_id, "date": item.date,
                    "matched_side": item.matched_side, "matched_fighter": item.matched_fighter,
                    "db_opponent": item.db_opponent,
                    "bfo_opponent": "", "bfo_opponent_url": "",
                    "offset_days": off if off is not None else "",
                    "name_equal_norm": "", "surname_subset": "", "lev_distance": "",
                    "reason": reason or "UNKNOWN",
                })
                continue

            eq_norm = (norm_name(item.db_opponent) == norm_name(opp_name))
            ssub = surname_subset(item.db_opponent, opp_name)
            lev  = levenshtein(item.db_opponent, opp_name)

            w.writerow({
                "fight_id": item.fight_id, "event_id": item.event_id, "date": item.date,
                "matched_side": item.matched_side, "matched_fighter": item.matched_fighter,
                "db_opponent": item.db_opponent,
                "bfo_opponent": opp_name, "bfo_opponent_url": opp_url,
                "offset_days": off if off is not None else 0,
                "name_equal_norm": int(eq_norm),
                "surname_subset": int(ssub),
                "lev_distance": lev,
                "reason": "",
            })
            written += 1

            if idx % max(1, args.progress_every) == 0:
                print(f"[{idx}/{len(one_sided)}] wrote {written} rows → {out_path}")

    print(f"Done. Wrote {written} rows to {out_path}. One-sided fights scanned: {len(one_sided)}")

if __name__ == "__main__":
    main()