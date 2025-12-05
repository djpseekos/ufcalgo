# src/scraper/bfo_search_links.py
from __future__ import annotations

import argparse
import csv
import html
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from bs4 import BeautifulSoup

from src.scraper.common.http import get_html, set_network_profile, set_identity

SEARCH_URL = "https://www.bestfightodds.com/search"
BFO_HEADERS = {"Referer": "https://www.bestfightodds.com/", "Accept-Language": "en-US,en;q=0.9"}

# ---------------------- normalization / tokens ----------------------

TOKEN_RE = re.compile(r"[a-z0-9']+")

def norm_name(s: str) -> str:
    """lower → strip accents → keep letters/digits/spaces/'- → collapse spaces"""
    s = html.unescape(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    return TOKEN_RE.findall(norm_name(s))

def cache_key_for_search(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9\-_.]+", "_", name.strip())[:80]
    return f"bfo_search/{safe}"

def given_and_surnames(tok: List[str]) -> tuple[str, List[str]]:
    """
    Heuristic:
      - first token -> given name
      - remaining tokens -> surnames (can be 1 or 2 for hispanic/portuguese)
      - if only one token, treat it as both given & surname
    """
    if not tok:
        return "", []
    if len(tok) == 1:
        return tok[0], [tok[0]]
    return tok[0], tok[1:]

# -------------------------- fuzzy scoring ---------------------------

def subset_match(a: List[str], b: List[str]) -> bool:
    if not a or not b: return False
    sa, sb = set(a), set(b)
    return sa.issubset(sb) or sb.issubset(sa)

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 0.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

def given_name_compatible(ga: str, gb: str) -> bool:
    """Require same given name, or a ≥3-char shared prefix (blocks 'Thaddeus' vs 'Alexander')."""
    if not ga or not gb:
        return False
    if ga == gb:
        return True
    pref = min(len(ga), len(gb))
    return pref >= 3 and (ga.startswith(gb) or gb.startswith(ga))

def score_name(target: str, cand: str) -> float:
    """
    Score in [0,1]. We’ll still gate on surname subset + given-name compatibility separately.
    """
    tn, cn = norm_name(target), norm_name(cand)
    if not tn or not cn: return 0.0
    if tn == cn: return 1.0
    tt, ct = tokens(target), tokens(cand)
    score = 0.0
    # surname subset gets a big boost (handles 'carlos leal' ⊆ 'carlos leal miranda')
    _, ts = given_and_surnames(tt)
    _, cs = given_and_surnames(ct)
    if subset_match(ts, cs):
        score += 0.55
    # jaccard over all tokens
    score += 0.30 * jaccard(tt, ct)
    # small boost for prefix agreement across full name
    if cn.startswith(tn) or tn.startswith(cn):
        score += 0.10
    # tiny boost if last tokens equal (often a terminal surname)
    if tt and ct and tt[-1] == ct[-1]:
        score += 0.05
    return min(score, 0.99)

# ---------------- parse a single search result page -----------------

def extract_fighter_link_from_search(html_text: str, target_name: str, min_score: float = 0.70) -> Optional[Tuple[str, str, str]]:
    """
    Return (anchor_text, absolute_url, match_type) where match_type ∈ {'exact','fuzzy'},
    or None if no acceptable match.
    """
    soup = BeautifulSoup(html_text, "lxml")
    anchors = soup.select('a[href^="/fighters/"]')
    if not anchors:
        return None

    tn = norm_name(target_name)
    tt = tokens(target_name)
    g_t, surn_t = given_and_surnames(tt)

    candidates = []
    for a in anchors:
        text = a.get_text(strip=True) or ""
        href = a.get("href") or ""
        if not text or not href.startswith("/fighters/"):
            continue
        candidates.append((text, href))

    # 1) exact first
    for text, href in candidates:
        if norm_name(text) == tn:
            return text, f"https://www.bestfightodds.com{href}", "exact"

    # 2) fuzzy with surname-subset + given-name compatibility gates
    best = None
    best_score = 0.0
    for text, href in candidates:
        ct = tokens(text)
        g_c, surn_c = given_and_surnames(ct)
        if not subset_match(surn_t, surn_c):
            continue
        if not given_name_compatible(g_t, g_c):
            continue
        s = score_name(target_name, text)
        if s > best_score:
            best_score = s
            best = (text, href)

    if best and best_score >= min_score:
        text, href = best
        return text, f"https://www.bestfightodds.com{href}", "fuzzy"

    return None

# ------------------------------- worker --------------------------------

def resolve_one(name: str, cache_key: str, min_score: float) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    url = f"{SEARCH_URL}?{urlencode({'query': name})}"
    try:
        html_text = get_html(url, cache_key=cache_key, ttl_hours=24, headers=BFO_HEADERS)
        parsed = extract_fighter_link_from_search(html_text, name, min_score=min_score)
        if parsed:
            text, fighter_url, match_type = parsed
            return match_type, fighter_url, text
        return None, None, None
    except Exception:
        return None, None, None

# -------------------------------- CLI / main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Map fighters to BestFightOdds links (fast, parallel, alias-aware).")
    ap.add_argument("--fighters", default="data/curated/fighters.csv")
    ap.add_argument("--out", default="data/curated/fighter_bfo_links.csv")
    ap.add_argument("--threads", type=int, default=24, help="Concurrent worker threads")
    ap.add_argument("--rate-per-sec", type=float, default=12.0, help="Per-host throttle (see http.py)")
    ap.add_argument("--min-score", type=float, default=0.70, help="Minimum fuzzy score to accept a match")
    ap.add_argument("--progress-every", type=int, default=200, help="Progress print frequency")
    args = ap.parse_args()

    set_network_profile(rate_per_sec=args.rate_per_sec, retries=2, backoff=0.2, timeout=8.0)
    set_identity()  # use defaults; override with env if needed

    df = pd.read_csv(args.fighters)
    if "fighter_id" not in df.columns or "fighter_name" not in df.columns:
        raise SystemExit("fighters.csv must contain fighter_id and fighter_name columns")

    rows = df[["fighter_id", "fighter_name"]].dropna().astype(str).to_dict("records")
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    misses = 0
    seen_ids = set()

    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        fut_to_row = {}
        for r in rows:
            fid, name = r["fighter_id"], r["fighter_name"]
            if fid in seen_ids or not name:
                continue
            seen_ids.add(fid)
            fut = ex.submit(resolve_one, name, cache_key_for_search(name), args.min_score)
            fut_to_row[fut] = (fid, name)

        done = 0
        total = len(fut_to_row)
        for fut in as_completed(fut_to_row):
            fid, name = fut_to_row[fut]
            match_type, url, text = fut.result()
            done += 1

            if url:
                results.append({"fighter_id": fid, "fighter_name": name, "fighter_link_bfo": url})
                tag = "OK" if match_type == "exact" else f"OK~{match_type.upper()}"
                if done % args.progress_every == 0:
                    print(f"[{tag}] {name} -> {text} :: {url}   ({done}/{total})")
            else:
                misses += 1
                if done % args.progress_every == 0:
                    print(f"[MISS] {name}   ({done}/{total})")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fighter_id", "fighter_name", "fighter_link_bfo"])
        w.writeheader()
        w.writerows(results)

    print(f"[DONE] wrote {len(results)} rows → {out_path}  (misses: {misses})")

if __name__ == "__main__":
    main()