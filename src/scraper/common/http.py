from __future__ import annotations

import os
import re
import time
import json
import random
import pathlib
import threading
from dataclasses import dataclass
from typing import Optional, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------- Errors & Config ---------------------------------

class HttpError(RuntimeError):
    def __init__(self, url: str, status: int, body_excerpt: str = ""):
        super().__init__(f"HTTP {status} for {url} :: {body_excerpt[:200]}")
        self.url, self.status, self.body_excerpt = url, status, body_excerpt


_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

@dataclass
class HttpConfig:
    timeout: int = 30
    retries: int = 4
    backoff: float = 1.0
    rate_per_sec: float = 1.5       # polite default ≈ 1–2 req/sec per host
    user_agent: str = _DEFAULT_UA
    contact_email: Optional[str] = None

_cfg = HttpConfig()

def set_identity(user_agent: Optional[str] = None, contact_email: Optional[str] = None):
    """Optionally override default UA / contact for all requests."""
    global _session_singleton
    if user_agent:
        _cfg.user_agent = user_agent
    if contact_email:
        _cfg.contact_email = contact_email

    if _session_singleton is not None:
        if user_agent:
            _session_singleton.headers["User-Agent"] = user_agent
        if contact_email:
            _session_singleton.headers["From"] = contact_email

# ----------------------------- Utilities -------------------------------------

_RATE_BUCKETS: Dict[str, float] = {}
_RATE_LOCK = threading.Lock()
_session_singleton: Optional[requests.Session] = None

def _ensure_dir(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def _rate_key_from_url(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url)
    return (m.group(1).lower() if m else "default")

def _throttle(rate_key: str):
    """Token-bucket-ish throttle: limit requests per host."""
    now = time.time()
    with _RATE_LOCK:
        last = _RATE_BUCKETS.get(rate_key, 0.0)
        min_interval = 1.0 / max(_cfg.rate_per_sec, 0.1)
        wait = (last + min_interval) - now
        if wait > 0:
            time.sleep(wait)
        _RATE_BUCKETS[rate_key] = time.time() + random.uniform(0.0, 0.08)

def _log_http(url: str, status: int, cached: bool, ms: int):
    try:
        day = time.strftime("%Y%m%d")
        path = f"logs/http_{day}.jsonl"
        _ensure_dir(path)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "url": url,
            "status": status,
            "cached": cached,
            "ms": ms,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never break scraping

def _session() -> requests.Session:
    """Singleton session with browsery defaults + retries."""
    global _session_singleton
    if _session_singleton:
        return _session_singleton

    s = requests.Session()
    headers = {
        "User-Agent": _cfg.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.ufcstats.com/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if _cfg.contact_email:
        headers["From"] = _cfg.contact_email
    s.headers.update(headers)

    retry = Retry(
        total=_cfg.retries,
        connect=_cfg.retries,
        read=_cfg.retries,
        backoff_factor=_cfg.backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    _session_singleton = s
    return s

# ----------------------------- Public API ------------------------------------

def get_html(
    url: str,
    cache_key: str | None = None,
    ttl_hours: int = 24,
    timeout: int = 30,
    headers: dict | None = None,
) -> str:
    """
    Fetch HTML with on-disk cache, polite headers, retries, and graceful fallback.
    We protect against empty/garbage bodies poisoning the cache.
    """
    cache_path = f"data/raw/html/{cache_key}.html" if cache_key else None

    def _looks_like_html(txt: str) -> bool:
        if not txt:
            return False
        low = txt.lower()
        return ("<html" in low) or ("<!doctype" in low) or (len(txt) >= 512)

    # 0) Serve fresh cache if present and valid
    if cache_path and os.path.exists(cache_path):
        try:
            is_fresh = (time.time() - os.path.getmtime(cache_path)) < ttl_hours * 3600
            if os.path.getsize(cache_path) < 64:
                os.remove(cache_path)
            elif is_fresh:
                with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
                    cached = f.read()
                if _looks_like_html(cached):
                    _log_http(url, 200, cached=True, ms=0)
                    return cached
                else:
                    os.remove(cache_path)
        except Exception:
            pass

    sess = _session()
    host_key = _rate_key_from_url(url)
    last_err = None

    # 1) Network fetch
    _throttle(host_key)
    t0 = time.time()
    try:
        resp = sess.get(url, timeout=timeout, allow_redirects=True, headers=headers)
        resp.raise_for_status()
        html = resp.text or ""

        if cache_path and _looks_like_html(html):
            _ensure_dir(cache_path)
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(html)
            os.replace(tmp_path, cache_path)

        _log_http(url, getattr(resp, "status_code", 0) or 200, cached=False, ms=int((time.time() - t0) * 1000))
        return html
    except Exception as e:
        last_err = e
        # 2) fallback to any stale cache
        if cache_path and os.path.exists(cache_path):
            try:
                if os.path.getsize(cache_path) >= 64:
                    with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
                        html = f.read()
                    if _looks_like_html(html):
                        _log_http(url, 0, cached=True, ms=int((time.time() - t0) * 1000))
                        return html
            except Exception:
                pass

    # 3) Give up
    status = getattr(last_err, "response", None).status_code if hasattr(last_err, "response") and last_err.response else 0
    _log_http(url, status or 0, cached=False, ms=int((time.time() - t0) * 1000))
    raise HttpError(url, status, str(last_err) if last_err else "")