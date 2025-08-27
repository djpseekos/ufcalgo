# src/scraper/common/http.py
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
    timeout: float = 8.0          # default network timeout (s)
    retries: int = 2              # total retries (urllib3 Retry)
    backoff: float = 0.2          # retry backoff factor
    rate_per_sec: float = 12.0    # per-host request rate ceiling
    user_agent: str = _DEFAULT_UA
    contact_email: Optional[str] = None

_cfg = HttpConfig()

def set_network_profile(rate_per_sec: float = 12.0, retries: int = 2, backoff: float = 0.2, timeout: float = 8.0):
    """Adjust global HTTP pacing/retry behavior on the fly."""
    _cfg.rate_per_sec = max(0.1, float(rate_per_sec))
    _cfg.retries = max(0, int(retries))
    _cfg.backoff = max(0.0, float(backoff))
    _cfg.timeout = max(1e-3, float(timeout))

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

def _now() -> float:
    # monotonic avoids time jumps affecting pacing
    return time.monotonic()

def _throttle(rate_key: str):
    """Leaky-bucket throttle per host without sleeping under the global lock."""
    min_interval = 1.0 / max(_cfg.rate_per_sec, 0.1)

    while True:
        now = time.time()
        with _RATE_LOCK:
            last = _RATE_BUCKETS.get(rate_key, 0.0)
            next_allowed = max(last, now)
            wait = next_allowed - now

            if wait <= 0:
                # reserve the next slot and go
                _RATE_BUCKETS[rate_key] = now + min_interval + random.uniform(0.0, 0.02)
                return
        # sleep OUTSIDE the lock
        time.sleep(min(wait, 0.5))

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
        # logging must never break scraping
        pass

def _session() -> requests.Session:
    """Singleton session with browsery defaults + retries + large pools."""
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
        # requests handles gzip/deflate/br automatically via Accept-Encoding
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
    # Larger pools so multiple workers can reuse connections efficiently
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=64,
        pool_maxsize=64,
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    _session_singleton = s
    return s

# ----------------------------- Public API ------------------------------------

def get_html(
    url: str,
    cache_key: str | None = None,
    ttl_hours: int = 24,
    timeout: float | None = None,
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
        # Allow smaller bodies — UFCStats pages can be compact
        return ("<html" in low) or ("<!doctype" in low) or (len(txt) >= 256)

    # 0) Serve fresh cache if present and valid
    if cache_path and os.path.exists(cache_path):
        try:
            if os.path.getsize(cache_path) < 64:
                os.remove(cache_path)
            else:
                is_fresh = (time.time() - os.path.getmtime(cache_path)) < ttl_hours * 3600
                if is_fresh:
                    with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
                        cached = f.read()
                    if _looks_like_html(cached):
                        _log_http(url, 200, cached=True, ms=0)
                        return cached
                    else:
                        os.remove(cache_path)
        except Exception:
            # fall through to network
            pass

    sess = _session()
    host_key = _rate_key_from_url(url)
    last_err = None

    # 1) Network fetch
    _throttle(host_key)
    t0 = _now()
    eff_timeout = _cfg.timeout if timeout is None else timeout
    try:
        # Use same timeout for connect/read
        resp = sess.get(url, timeout=(eff_timeout, eff_timeout), allow_redirects=True, headers=headers)
        # Requests w/ Retry won't raise_for_status on 5xx due to raise_on_status=False; we still log and cache on 200s.
        status = getattr(resp, "status_code", 0) or 0
        html = resp.text or ""
        if status >= 400:
            # Let Retry have handled transient errors; if we still got >=400, decide whether to serve stale cache.
            raise HttpError(url, status, html[:200])

        if cache_path and _looks_like_html(html):
            _ensure_dir(cache_path)
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(html)
            os.replace(tmp_path, cache_path)

        _log_http(url, status or 200, cached=False, ms=int((_now() - t0) * 1000))
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
                        _log_http(url, 0, cached=True, ms=int((_now() - t0) * 1000))
                        return html
            except Exception:
                pass

    # 3) Give up
    status = 0
    if isinstance(last_err, HttpError):
        status = last_err.status
    else:
        # try to extract from requests exception
        try:
            status = getattr(getattr(last_err, "response", None), "status_code", 0) or 0
        except Exception:
            pass
    _log_http(url, status or 0, cached=False, ms=int((_now() - t0) * 1000))
    # propagate a clean error
    raise HttpError(url, status, str(last_err) if last_err else "")