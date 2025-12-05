# src/scraper/ufcstats_fighters.py
from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pathlib
import random
import time

from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

from ..common import http as http_common
from ..common.parse import soup
from ..common.io import upsert_csv, update_manifest

FIGHTERS_OUT = "data/curated/fighters.csv"
DEBUG_DIR = "data/debug"
SITE_ROOT = "http://www.ufcstats.com/"

DIAG_DIR = pathlib.Path("data/debug/fighters_index")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- HTTP throttling/retries ----------------------------- #
http_common._cfg.rate_per_sec = 4.0
http_common._cfg.retries = 2
http_common._cfg.backoff = 0.3

# ----------------------------- helpers ------------------------------------- #

_ID_RX = re.compile(r"/fighter-details/([a-f0-9]+)", re.I)
_MMSS = re.compile(r"^(\d{1,2})\s*['′]\s*(\d{1,2})\s*[\"″]?$")  # 5' 11"
# Reach parsing: accept 78", 78″, 78 in, 78in, just 78, or centimeters like 198 cm
_INCH_RX_GENERIC = re.compile(r"(\d+(?:\.\d+)?)\s*(?:in|\"|″)?\b", re.I)
_CM_RX = re.compile(r"(\d+(?:\.\d+)?)\s*cm\b", re.I)
_LB_RX = re.compile(r"(\d+(?:\.\d+)?)\s*lb\.?", re.I)
_RECORD_RX = re.compile(r"\b(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\b")
_SPACEY = re.compile(r"\s+")
_NON = {"", "-", "—", "–", "N/A", "n/a"}

def _letters_from(letters: List[Tuple[str, str]], start_from: Optional[str]) -> List[Tuple[str, str]]:
    if not start_from:
        return letters
    ch = start_from.strip().lower()[:1]
    if not ch or ch < "a" or ch > "z":
        return letters
    return [(c, u) for (c, u) in letters if c.lower() >= ch]

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _clean(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return _SPACEY.sub(" ", x).strip()

def _maybe_none(x: Optional[str]) -> Optional[str]:
    x = _clean(x) if x is not None else None
    if x is None:
        return None
    return None if x in _NON else x

def _inches_from_height(ht: Optional[str]) -> Optional[int]:
    s = _maybe_none(ht)
    if s is None:
        return None
    m = _MMSS.search(s)
    if not m:
        return None
    feet = int(m.group(1))
    inches = int(m.group(2))
    return feet * 12 + inches

def _inches_from_reach(s: Optional[str]) -> Optional[float]:
    s = _maybe_none(s)
    if s is None:
        return None
    # Support centimeters, convert to inches
    m_cm = _CM_RX.search(s)
    if m_cm:
        try:
            cm = float(m_cm.group(1))
            return round(cm / 2.54, 2)
        except Exception:
            pass
    # Generic inches (", ″, in, or number)
    m = _INCH_RX_GENERIC.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _pounds_from_weight(s: Optional[str]) -> Optional[float]:
    s = _maybe_none(s)
    if s is None:
        return None
    m = _LB_RX.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

# ------------------- diagnostics helpers ----------------------------------- #

def _save_diag_blob(name: str, html: Optional[str]) -> None:
    try:
        if not html:
            return
        p = DIAG_DIR / name
        head = html[:2048]
        with open(p, "wb") as f:
            f.write(head.encode("utf-8", "ignore"))
    except Exception:
        pass

def _why_index_rejected(html: Optional[str]) -> str:
    if not html:
        return "empty-or-none"
    low = html.lower()
    if any(tok in low for tok in ("cf-browser-verification", "attention required", "checking your browser", "captcha")):
        return "bot-wall-or-challenge"
    if "b-statistics__table" not in low:
        return "missing-table"
    if "/fighter-details/" not in low:
        return "missing-profile-links"
    return "unknown"

# --------- page-type validators -------------------------------------------- #

def _looks_like_profile(html: str) -> bool:
    if not html:
        return False
    low = html.lower()
    return ("b-list__info-box" in low) or ("fighter details" in low)

def _looks_like_index(html: str) -> bool:
    if not html:
        return False
    low = html.lower()
    if any(tok in low for tok in ("cf-browser-verification", "attention required", "checking your browser", "captcha")):
        return False
    has_table = "b-statistics__table" in low
    has_profile_links = "/fighter-details/" in low
    return has_table and has_profile_links

# --------------------------- URL utilities --------------------------------- #

def _ensure_abs(url: str) -> str:
    return url if url.lower().startswith(("http://", "https://")) else urljoin(SITE_ROOT, url)

def _with_proto(url: str, proto: str) -> str:
    abs_url = _ensure_abs(url)
    u = urlparse(abs_url)
    return urlunparse((proto, u.netloc, u.path, u.params, u.query, u.fragment))

def _set_query_param(url: str, key: str, val: str) -> str:
    u = urlparse(_ensure_abs(url))
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q[key] = val
    new_q = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))

# --------------------------- HTTP fetch w/ protocols ------------------------ #

def _fetch_with_protocols(
    url: str,
    cache_key: Optional[str],
    referer: Optional[str],
    expect: str = "profile",
) -> Optional[str]:

    def ok(html: Optional[str]) -> bool:
        if expect == "index":
            return _looks_like_index(html or "")
        return _looks_like_profile(html or "")

    def fetch(proto: str, use_cache: bool) -> Optional[str]:
        final_url = _with_proto(url, proto)
        ck = f"{proto}_{cache_key}" if (cache_key and use_cache) else None
        headers = {
            "Referer": referer or SITE_ROOT,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.7",
        }
        if not use_cache:
            headers.update({"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"})
        try:
            time.sleep(random.uniform(0.05, 0.15))
            html = http_common.get_html(
                final_url,
                cache_key=ck,
                ttl_hours=720 if use_cache else 0,
                timeout=12,
                headers=headers,
            )
            key = (ck or cache_key or final_url).replace("/", "_").replace(":", "_").replace("&", "_")
            name = f"{key}_{'cache' if use_cache else 'fresh'}_{proto}.html"
            _save_diag_blob(name, html)
            return html
        except Exception as e:
            print(f"[diag] exception fetching {final_url} ({proto},{'cache' if use_cache else 'fresh'}): {e}")
            return None

    for proto in ("http", "https"):
        html = fetch(proto, use_cache=True)
        if ok(html):
            return html
        print(f"[diag] validator reject ({expect}) on {url} [{proto},cache]: "
              f"{_why_index_rejected(html) if expect=='index' else 'profile-mismatch'}")

    for proto in ("https", "http"):
        html = fetch(proto, use_cache=False)
        if ok(html):
            return html
        print(f"[diag] validator reject ({expect}) on {url} [{proto},fresh]: "
              f"{_why_index_rejected(html) if expect=='index' else 'profile-mismatch'}")

    return None

# --------------------------- data model ------------------------------------ #

@dataclass(slots=True)
class FighterRow:
    fighter_id: str
    fighter_name: Optional[str] = None
    height_in: Optional[int] = None
    weight_lb: Optional[float] = None
    reach_in: Optional[float] = None
    stance: Optional[str] = None
    dob: Optional[str] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    draws: Optional[int] = None
    profile_url: Optional[str] = None

# --------------------------- parsing: index -------------------------------- #

def _discover_letters() -> List[Tuple[str, str]]:
    """
    Returns list of (char, ABSOLUTE url_with_page_all).
    If the site chrome isn't available, fall back to A–Z.
    """
    base = urljoin(SITE_ROOT, "statistics/fighters")
    html = _fetch_with_protocols(base, cache_key="fighters_index", referer=None, expect="index")
    if not html:
        return [(c, _set_query_param(urljoin(SITE_ROOT, f"statistics/fighters?char={c}"), "page", "all"))
                for c in list("abcdefghijklmnopqrstuvwxyz")]

    doc = soup(html)
    out: List[Tuple[str, str]] = []
    for a in doc.select(".b-statistics__nav a, .b-statistics__sub-nav a, .b-statistics__nav-link a"):
        href = a.get("href") or ""
        if "statistics/fighters" not in href and "statistics/fighters" not in urljoin(SITE_ROOT, href):
            continue
        href_abs = _ensure_abs(href)
        href_abs = _set_query_param(href_abs, "page", "all")
        m = re.search(r"[?&]char=([^&]+)", href_abs, re.I)
        ch = (m.group(1) if m else a.get_text(strip=True) or "").lower()
        if ch:
            out.append((ch, href_abs))

    seen = set()
    uniq: List[Tuple[str, str]] = []
    for ch, u in out:
        if (ch, u) in seen:
            continue
        seen.add((ch, u))
        uniq.append((ch, u))
    if not uniq:
        uniq = [(c, _set_query_param(urljoin(SITE_ROOT, f"statistics/fighters?char={c}"), "page", "all"))
                for c in list("abcdefghijklmnopqrstuvwxyz")]
    return uniq

def _parse_letter_table(html: str) -> List[dict]:
    """
    Returns rows with fighter ids & display name (First + Last).
    """
    doc = soup(html)
    tbl = doc.select_one("table.b-statistics__table")
    out: List[dict] = []
    if not tbl:
        return out
    for tr in tbl.select("tbody tr"):
        aid = None
        for a in tr.select('a[href*="/fighter-details/"]'):
            m = _ID_RX.search(a.get("href", ""))
            if m:
                aid = m.group(1)
                break
        if not aid:
            continue
        tds = tr.find_all("td")
        first = _maybe_none(tds[0].get_text(" ", strip=True)) if len(tds) > 0 else None
        last  = _maybe_none(tds[1].get_text(" ", strip=True)) if len(tds) > 1 else None
        display = _clean(((first or "") + " " + (last or "")).strip()) or None
        out.append({"fighter_id": aid, "fighter_name": display})
    return out

# ------------------------- parsing: profile page ---------------------------- #

def _parse_info_boxes(doc) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns (anthro_map, career_map) with raw strings as shown in the two info boxes.
    """
    anthro: Dict[str, str] = {}
    career: Dict[str, str] = {}

    for box in doc.select("div.b-list__info-box"):
        for li in box.select("li"):
            raw = li.get_text(" ", strip=True)
            if not raw or ":" not in raw:
                continue
            label, val = raw.split(":", 1)
            label = _clean(label).lower()
            val = _clean(val)
            if not label:
                continue
            if any(key in label for key in ("height", "weight", "reach", "stance", "dob")):
                anthro[label] = val
            else:
                career[label] = val
    return anthro, career

def _parse_record_header(doc) -> Optional[Tuple[int,int,int]]:
    """
    Prefer the header badge: <span class="b-content__title-record">Record: 15-3-0</span>
    Fallbacks:
      - Any 'Record:' text found in the title block
      - Last resort: regex over full page text
    """
    node = doc.select_one("span.b-content__title-record")
    if node:
        txt = node.get_text(" ", strip=True).replace("\xa0", " ")
        m = _RECORD_RX.search(txt)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))

    title_block = doc.select_one(".b-content__title") or doc.select_one(".b-content__title-highlight")
    if title_block:
        txt = title_block.get_text(" ", strip=True).replace("\xa0", " ")
        m = _RECORD_RX.search(txt)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))

    txt = doc.get_text(" ", strip=True).replace("\xa0", " ")
    m = _RECORD_RX.search(txt)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None

def _parse_record_from_table(doc) -> Optional[Tuple[int,int,int]]:
    tbl = doc.select_one("table.b-fight-details__table")
    if not tbl:
        return None
    w = l = d = 0
    for tr in tbl.select("tbody tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        res = (tds[0].get_text(" ", strip=True) or "").upper()
        if res.startswith("NEXT"):
            continue
        if res.startswith("WIN"):
            w += 1
        elif res.startswith("LOSS"):
            l += 1
        elif res.startswith("DRAW"):
            d += 1
        elif res.startswith("NC"):
            pass
    return (w, l, d)

def _parse_profile_minimal(f: dict, letter_url: str) -> FighterRow:
    fid = f["fighter_id"]
    prof_url = urljoin(SITE_ROOT, f"fighter-details/{fid}")
    html = _fetch_with_protocols(prof_url, cache_key=f"fighter_{fid}", referer=letter_url, expect="profile")
    name = _maybe_none(f.get("fighter_name"))
    if not html:
        return FighterRow(fighter_id=fid, fighter_name=name, profile_url=prof_url)

    doc = soup(html)

    # Anthropometrics
    anthro_map, career_map = _parse_info_boxes(doc)
    height_in = _inches_from_height(anthro_map.get("height"))
    reach_in  = _inches_from_reach(anthro_map.get("reach"))
    weight_lb = _pounds_from_weight(anthro_map.get("weight"))
    stance    = _maybe_none(anthro_map.get("stance"))
    dob       = _maybe_none(anthro_map.get("dob"))

    # Record: header → career-box (W/L/D) → table
    rec = _parse_record_header(doc)
    if not rec:
        w = career_map.get("w"); l = career_map.get("l"); d = career_map.get("d")
        if all(v is not None for v in (w, l, d)):
            try:
                rec = (int(w), int(l), int(d))
            except Exception:
                rec = None
    if not rec:
        rec = _parse_record_from_table(doc)

    wins = losses = draws = None
    if rec:
        wins, losses, draws = rec

    # Name fallback from page header if listing name missing
    if not name:
        hdr = doc.select_one(".b-content__title, h2, h3")
        if hdr:
            name = _maybe_none(hdr.get_text(" ", strip=True))

    return FighterRow(
        fighter_id=fid,
        fighter_name=name,
        height_in=height_in,
        weight_lb=weight_lb,
        reach_in=reach_in,
        stance=stance,
        dob=dob,
        wins=wins,
        losses=losses,
        draws=draws,
        profile_url=prof_url,
    )

# ------------------------------- main crawl -------------------------------- #

def _dump_debug_fighter(fid: str) -> None:
    _ensure_dir(DEBUG_DIR)
    for proto in ("http", "https"):
        src = f"data/raw/html/{proto}_fighter_{fid}.html"
        if os.path.exists(src) and os.path.getsize(src) > 0:
            try:
                shutil.copyfile(src, os.path.join(DEBUG_DIR, f"debug_{proto}_fighter_{fid}.html"))
            except Exception:
                pass

def crawl_fighters(only_chars: Optional[str] = None, workers: int = 8, start_from: Optional[str] = None) -> pd.DataFrame:
    letters = _discover_letters()
    letters = _letters_from(letters, start_from)

    if only_chars:
        want = set(only_chars.lower().replace(",", " ").split())
        letters = [(ch, url) for ch, url in letters if ch.lower() in want]

    print(f"[fighters] planning letters: {', '.join(c for c, _ in letters)}")
    if only_chars:
        want = set(only_chars.lower().replace(",", " ").split())
        letters = [(ch, url) for ch, url in letters if ch.lower() in want]

    all_rows: List[FighterRow] = []
    skipped: List[str] = []
    total_scraped = 0

    for ch, url in letters:
        url = _ensure_abs(url)
        html = _fetch_with_protocols(
            url,
            cache_key=f"fighters_{ch}_all",
            referer=SITE_ROOT,
            expect="index",
        )

        # Fallback: try paginated pages 1..5 if page=all fails
        if not html:
            base_no_all = re.sub(r"[?&]page=all", "", url)
            got_any = False
            for p in range(1, 6):
                page_url = _set_query_param(base_no_all, "page", str(p))
                html = _fetch_with_protocols(
                    page_url,
                    cache_key=f"fighters_{ch}_p{p}",
                    referer=SITE_ROOT,
                    expect="index",
                )
                if html:
                    url = page_url
                    got_any = True
                    break
            if not got_any:
                print(f"[fighters] failed to fetch index for char={ch} (reason={_why_index_rejected(html)})")
                continue

        listings = _parse_letter_table(html)
        if not listings:
            print(f"[fighters] no listings for char={ch}")
            continue

        print(f"[fighters] found {len(listings)} listings for char={ch} …")

        # dedupe by fighter_id
        seen = set()
        todo = [l for l in listings if not (l["fighter_id"] in seen or seen.add(l["fighter_id"]))]

        def _job(lrow: dict):
            fid = lrow["fighter_id"]
            try:
                row = _parse_profile_minimal(lrow, letter_url=url)
                return fid, row, None
            except Exception as e:
                return fid, None, str(e)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(_job, lr): lr["fighter_id"] for lr in todo}
            total = len(futs)
            done = 0
            for fut in as_completed(futs):
                fid = futs[fut]
                done += 1
                try:
                    fid, row, err = fut.result()
                    if err or row is None:
                        skipped.append(fid)
                        _dump_debug_fighter(fid)
                    else:
                        all_rows.append(row)
                        total_scraped += 1
                except Exception:
                    skipped.append(fid)
                    _dump_debug_fighter(fid)
                if done % 50 == 0 or done == total:
                    print(f"[fighters] char={ch} parsed {done}/{total} profiles…", flush=True)

        print(f"[fighters] completed char={ch} → {len(todo)} profiles attempted, running total={total_scraped}")

    # ---------- finalize AFTER the loop ----------
    df = pd.DataFrame([asdict(r) for r in all_rows])

    cols = [
        "fighter_id",
        "fighter_name",
        "height_in",
        "weight_lb",
        "reach_in",
        "stance",
        "dob",
        "wins",
        "losses",
        "draws",
        "profile_url",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    if not df.empty:
        df = df[cols]

    if skipped:
        try:
            pd.Series(skipped, name="fighter_id").to_csv(
                "data/curated/_fighters_skipped.csv", index=False
            )
            print(f"[fighters] skipped {len(skipped)} profiles → data/curated/_fighters_skipped.csv")
        except Exception:
            pass

    return df

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scrape UFCStats fighters: id, name, anthropometrics, record, url")
    ap.add_argument("--only-chars", default=None, help="Limit to these initial buckets (e.g. 'a,c,d' or 'xyz')")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent profile fetches")
    ap.add_argument("--out", default=FIGHTERS_OUT, help="Output CSV path")
    ap.add_argument("--from-letter", default=None, help="Resume from this letter onward (e.g., 'd')")
    args = ap.parse_args(argv)

    df = crawl_fighters(only_chars=args.only_chars, workers=args.workers, start_from=args.from_letter)

    if df.empty:
        print("[fighters] parsed 0 rows.")
        return 0

    upsert_csv(df, args.out, keys=["fighter_id"])
    update_manifest(os.path.basename(args.out), rows=len(df))
    print(f"[fighters] wrote/updated {len(df)} fighters → {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())