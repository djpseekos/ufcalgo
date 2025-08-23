# src/scraper/common/io.py
from __future__ import annotations
import os, json, pathlib, time
from typing import List, Iterable
import pandas as pd

def _ensure_dir(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def write_csv(df: pd.DataFrame, path: str, *, index: bool = False) -> None:
    _ensure_dir(path)
    df.to_csv(path, index=index)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def upsert_csv(df: pd.DataFrame, path: str, keys: List[str], *, index: bool = False) -> None:
    _ensure_dir(path)
    if os.path.exists(path):
        cur = pd.read_csv(path)
    else:
        cur = pd.DataFrame(columns=df.columns)

    if not set(keys).issubset(df.columns):
        missing = set(keys) - set(df.columns)
        raise ValueError(f"Missing key columns in df: {missing}")

    if cur.empty:
        out = df.copy()
    else:
        # Merge: replace rows with matching keys, append new
        # Build a composite key
        def keyify(frame: pd.DataFrame) -> pd.Series:
            return frame[keys].astype(str).agg("§".join, axis=1)

        cur["_k"] = keyify(cur)
        df["_k"] = keyify(df)

        to_replace = cur["_k"].isin(df["_k"])
        cur = cur.loc[~to_replace, cur.columns]  # keep only non-overlapping
        out = pd.concat([cur.drop(columns=["_k"], errors="ignore"),
                         df.drop(columns=["_k"], errors="ignore")],
                        ignore_index=True)

    out.to_csv(path, index=index)

def update_manifest(table: str, rows: int, path: str = "data/curated/manifest.json") -> None:
    _ensure_dir(path)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            man = json.load(f)
    else:
        man = {}
    man[table] = {"rows": int(rows), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)

def append_log(line: dict, path: str):
    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")