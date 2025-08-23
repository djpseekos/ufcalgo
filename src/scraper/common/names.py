# src/scraper/common/names.py
from __future__ import annotations
import unicodedata, re

_WHITES = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s-]")

def canon(s: str) -> str:
    """
    Canonicalize a fighter name string:
    - strip accents
    - lowercase
    - drop punctuation
    - collapse whitespace
    - normalize hyphens/underscores to spaces
    """
    if not s:
        return ""
    # remove accents
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = s.replace("’", "'").replace("`", "'")
    # remove punctuation
    s = _PUNCT.sub(" ", s)
    # normalize separators
    s = s.replace("_", " ").replace("-", " ")
    # collapse whitespace
    s = _WHITES.sub(" ", s).strip()
    return s

def pair_key(a: str, b: str) -> str:
    """
    Build a canonical key for a fight matchup.
    Example:
      pair_key("Kamaru Usman", "Leon Edwards")
      -> "kamaru usman__leon edwards"
    """
    ca, cb = canon(a), canon(b)
    return "__".join(sorted([ca, cb]))