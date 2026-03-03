#!/usr/bin/env python3
"""Generate a symbol-specific trade setup from local Symbol Watcher data + HL live mid.

Constraints (boss rules):
- Advisory only (no execution).
- Use ONLY local data + Hyperliquid API for live price.
- Output must be one-shot, normie-friendly.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import urllib.request

HL_TRADER_DIR = os.getenv("HL_TRADER_DIR", "./hl-trader")
if HL_TRADER_DIR not in sys.path:
    sys.path.insert(0, HL_TRADER_DIR)

# LLM helper (used only for short reasoning blurb; deterministic fallback if unavailable)
from openclaw_agent_client import openclaw_agent_turn  # type: ignore

TRACKER_SYMBOL_DIR = Path(os.getenv("TRACKER_SYMBOL_DIR", "./data/tracker/symbol"))
HL_INFO_URL = "https://api.hyperliquid.xyz/info"

DEFAULT_REASONING_AGENT_ID = os.getenv("SYMBOL_SETUP_REASONING_AGENT_ID", "acp-alpha-analyst")
DEFAULT_REASONING_THINKING = os.getenv("SYMBOL_SETUP_REASONING_THINKING", "low")
DEFAULT_REASONING_TIMEOUT_SEC = float(os.getenv("SYMBOL_SETUP_REASONING_TIMEOUT_SEC", "30"))


def _utc_now() -> float:
    return time.time()


def _minutes_ago(iso_ts: str) -> Optional[int]:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(max(0, (_utc_now() - dt.timestamp()) // 60))
    except Exception:
        return None


def _hl_all_mids() -> Dict[str, str]:
    payload = {"type": "allMids"}
    req = urllib.request.Request(
        HL_INFO_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _resolve_symbol(request_sym: str, mids: Dict[str, str]) -> str:
    sym = (request_sym or "").strip().upper()
    if not sym:
        return ""

    # Exact match
    if sym in mids:
        return sym

    # Common: plural token name asked vs perp ticker (VIRTUALS -> VIRTUAL)
    if sym.endswith("S") and sym[:-1] in mids:
        return sym[:-1]

    # Try stripping common suffixes
    for suf in ("-PERP", "PERP"):
        if sym.endswith(suf):
            base = sym[: -len(suf)]
            if base in mids:
                return base

    return sym


def _load_tracker_snapshot(sym: str) -> Optional[Dict[str, Any]]:
    p = TRACKER_SYMBOL_DIR / f"{sym.lower()}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _fmt_usd(x: float) -> str:
    if x >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"${x/1_000:.2f}K"
    return f"${x:.2f}"


def _nudge_away_from_entry(entry: float, level: float, *, direction: str, min_move: float) -> float:
    """Ensure a level isn't equal/too-close to entry (numeric)."""
    if entry <= 0:
        return level
    eps = max(1e-9, entry * 1e-6)
    if abs(level - entry) <= eps:
        if direction == "above":
            return entry + min_move
        if direction == "below":
            return entry - min_move
    return level


def _ensure_fmt_not_equal(entry: float, level: float, *, direction: str, step: float) -> float:
    """Ensure formatted price string differs from entry (presentation-level guard)."""
    if entry <= 0:
        return level
    if step <= 0:
        return level
    e = _fmt_price(float(entry))
    x = float(level)
    # Avoid infinite loops; 5 nudges max
    for _ in range(5):
        if _fmt_price(x) != e:
            return x
        x = x + step if direction == "above" else x - step
    return x


async def _llm_reasoning(compact: Dict[str, Any]) -> str:
    """Return 2-4 bullets grounded in compact payload."""
    prompt = (
        "You are writing a SHORT reasoning blurb for a perp trade setup.\n"
        "Use ONLY the provided JSON. Do not use outside knowledge.\n"
        "Return 2-4 bullets, each starting with '- '.\n"
        "IMPORTANT: Write like a human. Do NOT mention JSON keys or function/field names (e.g., no 'flow.cohort_delta_signal').\n"
        "Do NOT use backticks/code formatting.\n"
        "Be concrete: talk about whale flow, crowd lean, funding pressure, nearby liquidation levels, and volatility in plain English.\n"
        "No hype.\n\n"
        "Input JSON:\n" + json.dumps(compact, ensure_ascii=False)
    )

    _meta, text = await openclaw_agent_turn(
        message=prompt,
        session_id=f"symbol_setup_reasoning_{int(time.time())}_{os.getpid()}",
        agent_id=DEFAULT_REASONING_AGENT_ID,
        thinking=DEFAULT_REASONING_THINKING,
        timeout_sec=DEFAULT_REASONING_TIMEOUT_SEC,
        openclaw_cmd=(os.getenv("OPENCLAW_CMD") or "openclaw").strip(),
    )
    out = (text or "").strip()
    bullets: List[str] = []
    for ln in out.splitlines():
        s = ln.strip()
        if not s.startswith("-"):
            continue
        s = "- " + s.lstrip("- ").strip()
        # sanitize: remove code-ish tokens / key references
        s = s.replace("`", "")
        for bad in ("flow.", "crowd.", "funding.", "liquidation.", "vol.", "live_price", "confidence"):
            s = s.replace(bad, "")
        # common artifacts
        s = s.replace(" low-", " low")
        bullets.append(s.strip())
    return "\n".join(bullets[:4]).strip()


def _fallback_reasoning(compact: Dict[str, Any]) -> List[str]:
    out: List[str] = []

    flow = str(((compact.get("flow") or {}).get("cohort_delta_signal")) or "").strip()
    if flow:
        if "WHALE" in flow.upper():
            out.append(f"- Whale flow looks bearish ({flow}).")
        else:
            out.append(f"- Flow read: {flow}.")

    crowd = (compact.get("crowd") or {})
    if crowd.get("lean"):
        pct = crowd.get("lean_pct")
        if pct is not None:
            try:
                out.append(f"- The crowd is leaning {str(crowd.get('lean')).upper()} (~{float(pct):.0f}%).")
            except Exception:
                out.append(f"- The crowd is leaning {str(crowd.get('lean')).upper()}.")

    liq = (compact.get("liquidation") or {})
    if liq.get("nearest_long_liq") is not None and compact.get("live_price") is not None:
        out.append(
            f"- A nearby downside liquidation pocket sits around {_fmt_price(float(liq['nearest_long_liq']))} (below spot)."
        )

    funding = (compact.get("funding") or {})
    if funding.get("direction") and funding.get("annualized_pct") is not None:
        try:
            out.append(
                f"- Funding is about {float(funding['annualized_pct']):.2f}% annualized ({str(funding['direction']).replace('_', ' ').lower()})."
            )
        except Exception:
            pass

    atr_pct = (compact.get("vol") or {}).get("atr_pct")
    if atr_pct is not None:
        try:
            if float(atr_pct) > 0:
                out.append(f"- Volatility is moderate (ATR ~{float(atr_pct):.2f}%).")
        except Exception:
            pass

    return out[:4]


def _pick_bias(snapshot: Dict[str, Any]) -> Tuple[str, str]:
    """Return (bias, conviction) where conviction is low|medium|high."""
    ps = snapshot.get("perp_signals") or {}
    short_hits = 0
    long_hits = 0
    if isinstance(ps, dict):
        for _k, v in ps.items():
            if not isinstance(v, dict):
                continue
            sig = str(v.get("signal") or "").upper()
            if sig == "SHORT":
                short_hits += 1
            elif sig == "LONG":
                long_hits += 1

    # Most of our tracker signals are SHORT/NEUTRAL today; keep logic conservative.
    if short_hits >= 2 and long_hits == 0:
        return "SHORT", "low"
    if long_hits >= 2 and short_hits == 0:
        return "LONG", "low"
    return "MIXED", "low"


def _levels(snapshot: Dict[str, Any], live_price: float) -> Dict[str, Any]:
    atr = snapshot.get("atr") or {}
    atr_abs = None
    atr_pct = None
    if isinstance(atr, dict):
        try:
            atr_abs = float(atr.get("atr_abs")) if atr.get("atr_abs") is not None else None
        except Exception:
            atr_abs = None
        try:
            atr_pct = float(atr.get("atr_pct")) if atr.get("atr_pct") is not None else None
        except Exception:
            atr_pct = None

    lh = snapshot.get("liquidation_heatmap") or {}
    nearest_long = None
    most_huntable_long = None
    if isinstance(lh, dict):
        try:
            nearest_long = float(lh.get("nearest_long_liq")) if lh.get("nearest_long_liq") is not None else None
        except Exception:
            nearest_long = None
        mhl = lh.get("most_huntable_long")
        if isinstance(mhl, dict):
            try:
                most_huntable_long = float(mhl.get("price")) if mhl.get("price") is not None else None
            except Exception:
                most_huntable_long = None

    # Clamp huntable levels: if far from live price, ignore (likely not actionable).
    try:
        lp = float(live_price)
    except Exception:
        lp = None

    if lp and lp > 0:
        max_dev = 0.10  # 10%

        def _within_10pct(x: float) -> bool:
            return abs(x - lp) / lp <= max_dev

        if nearest_long is not None and not _within_10pct(nearest_long):
            nearest_long = None

        if most_huntable_long is not None and not _within_10pct(most_huntable_long):
            most_huntable_long = None

    # Fallback ATR abs if missing
    if atr_abs is None and atr_pct is not None:
        atr_abs = float(live_price) * (atr_pct / 100.0)
    if atr_abs is None:
        atr_abs = float(live_price) * 0.02  # conservative fallback

    # Build zones
    sell_lo = live_price + 0.5 * atr_abs
    sell_hi = live_price + 1.8 * atr_abs
    short_inv = sell_hi + 0.9 * atr_abs

    tp1_short = live_price - 1.4 * atr_abs
    tp2_short = nearest_long if nearest_long is not None else live_price - 3.0 * atr_abs

    # Long zone anchored on liquidation support if we have it
    buy_anchor = nearest_long if nearest_long is not None else live_price - 2.0 * atr_abs
    buy_lo = buy_anchor
    buy_hi = buy_anchor + 1.2 * atr_abs

    long_inv = (most_huntable_long - 0.3 * atr_abs) if most_huntable_long is not None else (buy_lo - 1.0 * atr_abs)

    tp1_long = live_price - 0.0 * atr_abs  # back to current area
    tp2_long = sell_hi

    return {
        "atr_abs": atr_abs,
        "atr_pct": atr_pct,
        "sell_zone": (sell_lo, sell_hi),
        "short_invalidation": short_inv,
        "short_tps": (tp1_short, tp2_short),
        "buy_zone": (buy_lo, buy_hi),
        "long_invalidation": long_inv,
        "long_tps": (tp1_long, tp2_long),
        "nearest_long_liq": nearest_long,
        "most_huntable_long": most_huntable_long,
    }


def _fmt_price(x: float) -> str:
    # adaptive decimals
    if x < 1:
        return f"${x:.4f}"
    if x < 100:
        return f"${x:.3f}"
    return f"${x:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--side", choices=["LONG", "SHORT"], default=None, help="Force a single-side output")
    args = ap.parse_args()

    mids = _hl_all_mids()
    sym = _resolve_symbol(args.symbol, mids)
    if not sym:
        print("Invalid symbol")
        return 2

    mid_raw = mids.get(sym)
    live_price = float(mid_raw) if mid_raw is not None else None

    snap = _load_tracker_snapshot(sym)
    if snap is None:
        # We can still return something minimal using live price only.
        print(
            f"{args.symbol.strip().upper()} Trade Setup\n\n"
            f"Live price (HL): {_fmt_price(live_price) if live_price is not None else 'n/a'}\n"
            "Data freshness: n/a\n\n"
            "We don’t have Symbol Watcher data for this ticker yet in our local cache."
        )
        return 0

    # If tracker price exists, prefer it as local mid; otherwise use HL mid.
    tracker_price = None
    try:
        tracker_price = float(snap.get("price")) if snap.get("price") is not None else None
    except Exception:
        tracker_price = None
    if live_price is None:
        live_price = tracker_price

    gen_at = str(snap.get("generated_at") or "").strip()
    mins = _minutes_ago(gen_at) if gen_at else None

    bias, conviction = _pick_bias(snap)
    if args.side in ("LONG", "SHORT"):
        bias = str(args.side)
    lvl = _levels(snap, float(live_price or 0.0))

    summary = snap.get("summary") or {}
    long_ct = summary.get("long_count")
    short_ct = summary.get("short_count")
    net_bias = summary.get("net_bias")
    net_bias_pct = summary.get("net_bias_pct")
    oi = summary.get("market_oi")

    funding = snap.get("funding") or {}
    fund_ann = funding.get("annualized_pct")
    fund_dir = funding.get("direction")

    cd = snap.get("cohort_delta") or {}
    cd_signal = cd.get("signal")

    lh = snap.get("liquidation_heatmap") or {}
    nearest_long = lvl.get("nearest_long_liq")
    most_huntable_long = lvl.get("most_huntable_long")

    # Build output
    lines = []
    title_sym = args.symbol.strip().upper()
    lines.append(f"{title_sym} Trade Setup")
    if sym != title_sym:
        lines.append(f"HL ticker: {sym}")
    lines.append("")

    # TLDR
    tldr = "Mixed signals. Use a levels plan (SR_Limit style), not a market-chase signal." if bias == "MIXED" else f"Directional lean is {bias}."
    lines.append(f"TL;DR: {tldr}")

    # Live price + freshness
    lp = _fmt_price(float(live_price)) if live_price is not None else "n/a"
    lines.append(f"Live price (HL): {lp}")
    if mins is None:
        lines.append("Data freshness: unknown")
    elif mins == 0:
        lines.append("Data freshness: updated <1 minute ago")
    else:
        lines.append(f"Data freshness: updated ~{mins} minutes ago")
    lines.append("")

    # Direction + style
    if bias == "MIXED":
        lines.append("Direction + style: SR_Limit (levels-based)")
    else:
        # low conviction still -> SR limit; high conviction would be current-price signal (future)
        lines.append(f"Direction: {bias} (confidence: {conviction})")
    lines.append("")

    # Action setup(s)
    if bias == "SHORT":
        (sell_lo, sell_hi) = lvl["sell_zone"]
        (tp1, tp2) = lvl["short_tps"]
        inv = lvl["short_invalidation"]

        entry = float(live_price or 0.0)
        atr_pct = lvl.get("atr_pct")
        min_move = max(
            entry * 0.002,
            0.01 if entry >= 1 else (0.0001 if entry > 0 else 0.0),
            (entry * float(atr_pct) / 100.0 * 0.25) if (atr_pct and entry > 0) else 0.0,
        )

        inv = _nudge_away_from_entry(entry, float(inv), direction="above", min_move=min_move)
        tp1 = _nudge_away_from_entry(entry, float(tp1), direction="below", min_move=min_move)
        tp2 = _nudge_away_from_entry(entry, float(tp2), direction="below", min_move=max(min_move * 1.5, min_move))
        if tp2 >= tp1:
            tp2 = tp1 - max(min_move * 0.5, 1e-9)

        # presentation-level guard
        inv = _ensure_fmt_not_equal(entry, float(inv), direction="above", step=min_move)
        tp1 = _ensure_fmt_not_equal(entry, float(tp1), direction="below", step=min_move)
        tp2 = _ensure_fmt_not_equal(entry, float(tp2), direction="below", step=min_move)
        if _fmt_price(float(tp2)) == _fmt_price(float(tp1)):
            tp2 = float(tp2) - min_move

        lines.append("Action setup")
        lines.append(f"SHORT signal: {lp}")
        lines.append(f"Stop Loss: above {_fmt_price(float(inv))}")
        lines.append(f"Take Profit: {_fmt_price(float(tp1))} then {_fmt_price(float(tp2))}")
    elif bias == "LONG":
        (buy_lo, buy_hi) = lvl["buy_zone"]
        (tp1, tp2) = lvl["long_tps"]
        inv = lvl["long_invalidation"]

        entry = float(live_price or 0.0)
        atr_pct = lvl.get("atr_pct")
        min_move = max(
            entry * 0.002,
            0.01 if entry >= 1 else (0.0001 if entry > 0 else 0.0),
            (entry * float(atr_pct) / 100.0 * 0.25) if (atr_pct and entry > 0) else 0.0,
        )

        inv = _nudge_away_from_entry(entry, float(inv), direction="below", min_move=min_move)
        tp1 = _nudge_away_from_entry(entry, float(tp1), direction="above", min_move=min_move)
        tp2 = _nudge_away_from_entry(entry, float(tp2), direction="above", min_move=max(min_move * 1.5, min_move))
        if tp2 <= tp1:
            tp2 = tp1 + max(min_move * 0.5, 1e-9)

        # presentation-level guard
        inv = _ensure_fmt_not_equal(entry, float(inv), direction="below", step=min_move)
        tp1 = _ensure_fmt_not_equal(entry, float(tp1), direction="above", step=min_move)
        tp2 = _ensure_fmt_not_equal(entry, float(tp2), direction="above", step=min_move)
        if _fmt_price(float(tp2)) == _fmt_price(float(tp1)):
            tp2 = float(tp2) + min_move

        lines.append("Action setup")
        lines.append(f"LONG signal: {lp}")
        lines.append(f"Stop Loss: below {_fmt_price(float(inv))}")
        lines.append(f"Take Profit: {_fmt_price(float(tp1))} then {_fmt_price(float(tp2))}")
    else:
        # Mixed: output both sides unless caller forced a side.
        (sell_lo, sell_hi) = lvl["sell_zone"]
        (tp1s, tp2s) = lvl["short_tps"]
        invs = lvl["short_invalidation"]

        (buy_lo, buy_hi) = lvl["buy_zone"]
        (tp1l, tp2l) = lvl["long_tps"]
        invl = lvl["long_invalidation"]

        lines.append("Action setups")
        lines.append(f"Setup A — SHORT idea")
        lines.append(f"Sell zone: {_fmt_price(float(sell_lo))} – {_fmt_price(float(sell_hi))}")
        lines.append(f"Stop Loss: above {_fmt_price(float(invs))}")
        lines.append(f"Take Profit: {_fmt_price(float(tp1s))} then {_fmt_price(float(tp2s))}")
        lines.append("")
        lines.append(f"Setup B — LONG idea")
        lines.append(f"Buy zone: {_fmt_price(float(buy_lo))} – {_fmt_price(float(buy_hi))}")
        lines.append(f"Stop Loss: below {_fmt_price(float(invl))}")
        lines.append(f"Take Profit: {_fmt_price(float(tp1l))} then {_fmt_price(float(tp2l))}")

    lines.append("")

    # Key levels
    levels = []
    if nearest_long is not None:
        levels.append(f"Nearest downside liquidation: {_fmt_price(float(nearest_long))}")
    if most_huntable_long is not None:
        levels.append(f"Most huntable downside pocket: {_fmt_price(float(most_huntable_long))}")
    if levels:
        lines.append("Key levels")
        lines.extend([f"- {x}" for x in levels])
        lines.append("")

    # Short reasoning (LLM; fallback deterministic)
    compact = {
        "symbol": sym,
        "side": bias,
        "live_price": float(live_price) if live_price is not None else None,
        "confidence": conviction,
        "flow": {
            "cohort_delta_signal": cd_signal,
        },
        "crowd": {
            "long_count": long_ct,
            "short_count": short_ct,
            "lean": net_bias,
            "lean_pct": net_bias_pct,
        },
        "funding": {
            "annualized_pct": fund_ann,
            "direction": fund_dir,
        },
        "liquidation": {
            "nearest_long_liq": lvl.get("nearest_long_liq"),
            "nearest_short_liq": lvl.get("nearest_short_liq"),
        },
        "vol": {
            "atr_pct": lvl.get("atr_pct"),
        },
    }

    bullets: List[str] = []
    try:
        import asyncio

        txt = asyncio.run(_llm_reasoning(compact))
        if txt:
            bullets = [ln.strip() for ln in txt.splitlines() if ln.strip().startswith("-")]
    except Exception:
        bullets = []

    if not bullets:
        bullets = _fallback_reasoning(compact)

    if bullets:
        lines.append("Reasoning")
        lines.extend(bullets)
        lines.append("")

    # Orderbook walls (not yet wired)
    lines.append("Orderbook walls")
    lines.append("- Nearest big bid: not available yet")
    lines.append("- Nearest big ask: not available yet")

    print("\n".join(lines).strip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
