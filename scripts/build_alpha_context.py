#!/usr/bin/env python3
"""Build a compact Alpha Dashboard context JSON from latest hl-trader artifacts.

Advisory-only: intended to feed an LLM for a normie-friendly market overview + bias.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


STATE_DIR = os.getenv("HL_TRADER_STATE_DIR", "./data/hl-trader")
CTX_GLOB = os.path.join(STATE_DIR, "hl_trader_context_*.json")


def _latest_path(pattern: str) -> str | None:
    paths = glob.glob(pattern)
    if not paths:
        return None
    paths.sort(key=os.path.getmtime)
    return paths[-1]


def _direction_counts(symbol_directions: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _sym, d in (symbol_directions or {}).items():
        key = (d or "UNKNOWN").upper()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _top_trends(symbol_trends: Dict[str, Any], n: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    for sym, blob in (symbol_trends or {}).items():
        if not isinstance(blob, dict):
            continue
        score = blob.get("score")
        if score is None:
            continue
        try:
            score_f = float(score)
        except Exception:
            continue
        rows.append(
            {
                "symbol": sym,
                "direction": (blob.get("direction") or "").upper(),
                "regime": blob.get("regime"),
                "score": score_f,
            }
        )

    bullish_rows = [r for r in rows if r["score"] > 0]
    bearish_rows = [r for r in rows if r["score"] < 0]

    bullish = sorted(bullish_rows, key=lambda r: r["score"], reverse=True)[:n]
    bearish = sorted(bearish_rows, key=lambda r: r["score"])[:n]
    return bullish, bearish


def main() -> None:
    path = _latest_path(CTX_GLOB)
    if not path:
        out = {
            "ok": False,
            "error": f"No context artifacts found at {CTX_GLOB}",
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    mtime = os.path.getmtime(path)
    age_sec = time.time() - mtime

    with open(path, "r") as f:
        ctx = json.load(f)

    symbol_directions = ctx.get("symbol_directions") or {}
    symbol_trends = ctx.get("symbol_trends") or {}

    bullish, bearish = _top_trends(symbol_trends)

    out = {
        "ok": True,
        "source": {
            "path": path,
            "mtime_ts": mtime,
            "age_sec": age_sec,
        },
        "global_context_compact": ctx.get("global_context_compact"),
        "direction_counts": _direction_counts(symbol_directions),
        "top_bullish": bullish,
        "top_bearish": bearish,
    }

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
