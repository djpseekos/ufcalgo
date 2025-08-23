# src/scraper/common/http.py
from __future__ import annotations
import os, re, time, json, random, pathlib, threading
from dataclasses import dataclass
from typing import Optional, Dict
import requests

_DEFAULT_UA = "UFC-AlgScraper/1.0 (+contact: data-team@example.com)"
_RATE_BUCKETS: Dict[str, float] = {}
_RATE_LOCK = threading.Lock()

class HttpError(RuntimeError):
    def __init__(self, url: str, status: int, body_excerpt: str = ""):
        super().__init__(f"HTTP {status} for {url} :: {body_excerpt[:200]}")
        self.url, self.status, self.body_excerpt = url, status, body_excerpt

@dataclass
class HttpConfig:
    timeout: int = 20
    retries: int = 3
    backoff: float = 1.6
    rate_per_sec: float = 1.5   # ~1–2 req/sec by default
    user_agent: str = _DEFAULT_UA
    contact_email: Optional[str] = None

_cfg = HttpConfig()

def set_identity(user_agent: Optional[str] = None, contact_email: Optional[str] = None):
    if user_agent:
        _cfg.user_agent = user_agent
    if contact_email:
        _cfg.contact_email = contact_email

def _headers(extra: Optional[dict] = None) -> dict:
    h = {"User-Agent": _cfg.user_agent}
    if _cfg.contact_email:
        h["From"] = _cfg.contact_email
    if extra:
        h.update(extra)
    return h

def _rate_key_from_url(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url)
    return (m.group(1).lower() if m else "default")

def _throttle(rate_key: str):
    # Token-bucket-ish: ensure at most rate_per_sec per key
    now = time.time()
    with _RATE_LOCK:
        last = _RATE_BUCKETS.get(rate_key, 0.0)
        min_interval = 1.0 / max(_cfg.rate_per_sec, 0.1)
        wait = (last + min_interval) - now
        if wait > 0:
            time.sleep(wait)
        _RATE_BUCKETS[rate_key] = time.time() + random.uniform(0, 0.08)

def _ensure_dir(p: str):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)

def _log_http(url: str, status: int, cached: bool, ms: int):
    day = time.strftime("%Y%m%d")
    path = f"logs/http_{day}.jsonl"
    _ensure_dir(path)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": url, "status": status, "cached": cached, "ms": ms}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def get_html(url: str, *, cache_key: Optional[str] = None, ttl_hours: int = 24*30,
             headers: Optional[dict] = None, rate_key: Optional[str] = None) -> str:
    """
    Polite GET with per-domain throttling, retries, and optional on-disk cache.
    Stores/reads cache at data/raw/html/{cache_key}.html
    """
    if cache_key:
        cache_path = f"data/raw/html/{cache_key}.html"
        if os.path.exists(cache_path):
            age_h = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age_h <= ttl_hours:
                with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
                _log_http(url, 200, True, 0)
                return html

    sess = requests.Session()
    hdrs = _headers(headers)
    rk = rate_key or _rate_key_from_url(url)
    backoff = _cfg.backoff
    err: Optional[Exception] = None
    start = time.time()

    for attempt in range(_cfg.retries + 1):
        try:
            _throttle(rk)
            resp = sess.get(url, headers=hdrs, timeout=_cfg.timeout)
            ms = int((time.time() - start) * 1000)
            if 200 <= resp.status_code < 300:
                html = resp.text
                if cache_key:
                    _ensure_dir(f"data/raw/html/{cache_key}.html")
                    with open(f"data/raw/html/{cache_key}.html", "w", encoding="utf-8") as f:
                        f.write(html)
                _log_http(url, resp.status_code, False, ms)
                return html
            elif resp.status_code in (429, 500, 502, 503, 504):
                # transient
                time.sleep(backoff + random.uniform(0, 0.3))
                backoff *= 1.7
                continue
            else:
                _log_http(url, resp.status_code, False, ms)
                raise HttpError(url, resp.status_code, resp.text[:300])
        except requests.RequestException as e:
            err = e
            time.sleep(backoff + random.uniform(0, 0.3))
            backoff *= 1.7

    # all retries failed
    raise HttpError(url, getattr(err, "response", None).status_code if hasattr(err, "response") and err.response else 0, str(err) if err else "")