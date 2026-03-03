#!/usr/bin/env python3
"""Generate up to N best trade signals right now (LLM-first with deterministic fallback).

Data constraints:
- Use ONLY local Symbol Watcher snapshots (TRACKER_SYMBOL_DIR/*.json)
- Use Hyperliquid /info allMids for live price refresh.

Output constraints:
- Normie-friendly, one-shot.
- Max 2 signals.
- One side per symbol (no long+short for same symbol).
- If LLM times out/fails -> deterministic fallback.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.request

HL_TRADER_DIR = os.getenv("HL_TRADER_DIR", "./hl-trader")
if HL_TRADER_DIR not in sys.path:
    sys.path.insert(0, HL_TRADER_DIR)

from openclaw_agent_client import openclaw_agent_turn  # type: ignore

SYMBOL_DIR = Path(os.getenv("TRACKER_SYMBOL_DIR", "./data/tracker/symbol"))
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
SETUP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_symbol_setup.py")


def _utc_iso(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hl_all_mids() -> Dict[str, str]:
    payload = {"type": "allMids"}
    req = urllib.request.Request(
        HL_INFO_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _minutes_ago(iso_ts: str) -> Optional[int]:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(max(0, (time.time() - dt.timestamp()) // 60))
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _resolve_symbol(sym: str, mids: Dict[str, str]) -> str:
    s = (sym or "").strip().upper()
    if not s:
        return ""
    if s in mids:
        return s
    if s.endswith("S") and s[:-1] in mids:
        return s[:-1]
    for suf in ("-PERP", "PERP"):
        if s.endswith(suf):
            b = s[: -len(suf)]
            if b in mids:
                return b
    return s


def _load_snapshot(sym: str) -> Optional[Dict[str, Any]]:
    p = SYMBOL_DIR / f"{sym.lower()}.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _pick_side(snapshot: Dict[str, Any]) -> str:
    """Heuristic single-side pick for fallback and as an LLM hint."""
    cd = snapshot.get("cohort_delta") or {}
    sig = str(cd.get("signal") or "").upper()
    if "WHALE" in sig and "DUMP" in sig:
        return "SHORT"
    if "WHALE" in sig and ("LOAD" in sig or "BUY" in sig):
        return "LONG"
    if "TRAPPED" in sig and "LONG" in sig:
        return "SHORT"
    if "TRAPPED" in sig and "SHORT" in sig:
        return "LONG"

    sdc = snapshot.get("smart_dumb_cvd") or {}
    div_sig = str(sdc.get("divergence_signal") or "").upper()
    sigf = str(sdc.get("significance") or "").upper()
    if div_sig == "DUMB_BULLISH" and sigf in {"EXTREME", "HIGH"}:
        return "SHORT"
    if div_sig == "DUMB_BEARISH" and sigf in {"EXTREME", "HIGH"}:
        return "LONG"

    # Perp signals vote
    ps = snapshot.get("perp_signals") or {}
    short_hits = 0
    long_hits = 0
    if isinstance(ps, dict):
        for _k, v in ps.items():
            if not isinstance(v, dict):
                continue
            s = str(v.get("signal") or "").upper()
            if s == "SHORT":
                short_hits += 1
            elif s == "LONG":
                long_hits += 1
    if short_hits > long_hits:
        return "SHORT"
    if long_hits > short_hits:
        return "LONG"

    # Crowd lean fallback
    summary = snapshot.get("summary") or {}
    nb = str(summary.get("net_bias") or "").upper()
    if nb == "LONG":
        return "LONG"
    if nb == "SHORT":
        return "SHORT"
    return "SHORT"


def _score_snapshot(snapshot: Dict[str, Any], *, live_price: float) -> float:
    summary = snapshot.get("summary") or {}
    net_bias_pct = _safe_float(summary.get("net_bias_pct")) or 50.0
    crowd_skew = abs(net_bias_pct - 50.0) / 50.0

    sdc = snapshot.get("smart_dumb_cvd") or {}
    divz = abs(_safe_float(sdc.get("divergence_z")) or 0.0)

    cd = snapshot.get("cohort_delta") or {}
    sig = str(cd.get("signal") or "").upper()
    whale_bonus = 0.0
    if "WHALE" in sig:
        whale_bonus = 0.75

    lh = snapshot.get("liquidation_heatmap") or {}
    nl = _safe_float(lh.get("nearest_long_liq"))
    actionable = 0.0
    if nl is not None and live_price > 0:
        if abs(nl - live_price) / live_price <= 0.10:
            actionable = 1.0

    atr = snapshot.get("atr") or {}
    atr_pct = _safe_float(atr.get("atr_pct"))
    atr_ok = 1.0 if (atr_pct is not None and atr_pct > 0) else 0.0

    # Penalize stale
    mins = _minutes_ago(str(snapshot.get("generated_at") or ""))
    stale_pen = 0.0
    if mins is not None and mins > 10:
        stale_pen = 1.5

    return 1.2 * divz + 0.6 * crowd_skew + whale_bonus + 0.5 * actionable + 0.4 * atr_ok - stale_pen


def _compact(snapshot: Dict[str, Any], *, sym: str, live_price: float) -> Dict[str, Any]:
    summary = snapshot.get("summary") or {}
    lh = snapshot.get("liquidation_heatmap") or {}
    sdc = snapshot.get("smart_dumb_cvd") or {}
    cd = snapshot.get("cohort_delta") or {}
    funding = snapshot.get("funding") or {}
    atr = snapshot.get("atr") or {}

    return {
        "symbol": sym,
        "generated_at": snapshot.get("generated_at"),
        "freshness_min": _minutes_ago(str(snapshot.get("generated_at") or "")),
        "live_price": live_price,
        "summary": {
            "market_oi": summary.get("market_oi"),
            "long_count": summary.get("long_count"),
            "short_count": summary.get("short_count"),
            "net_bias": summary.get("net_bias"),
            "net_bias_pct": summary.get("net_bias_pct"),
            "avg_leverage": summary.get("avg_leverage"),
            "coverage_pct": summary.get("coverage_pct"),
        },
        "liquidation": {
            "nearest_long_liq": lh.get("nearest_long_liq"),
            "nearest_short_liq": lh.get("nearest_short_liq"),
            "most_huntable_long": lh.get("most_huntable_long"),
            "most_huntable_short": lh.get("most_huntable_short"),
        },
        "flow": {
            "cohort_delta_signal": cd.get("signal"),
            "smart_vs_dumb_delta": cd.get("smart_vs_dumb_delta"),
            "whale_vs_retail_delta": cd.get("whale_vs_retail_delta"),
            "dumb_divergence_signal": sdc.get("divergence_signal"),
            "dumb_divergence_z": sdc.get("divergence_z"),
            "dumb_significance": sdc.get("significance"),
        },
        "funding": {
            "annualized_pct": funding.get("annualized_pct"),
            "direction": funding.get("direction"),
        },
        "atr": {
            "atr_pct": atr.get("atr_pct"),
        },
        "perp_signals": snapshot.get("perp_signals"),
        "side_hint": _pick_side(snapshot),
    }


def _run_setup(sym: str, side: str) -> str:
    side = side.upper().strip()
    if side not in {"LONG", "SHORT"}:
        side = "SHORT"
    return subprocess.check_output(["python3", SETUP_SCRIPT, sym, "--side", side], text=True).strip()


def _llm_prompt(payload: Dict[str, Any]) -> str:
    rules = [
        "Use ONLY the provided JSON. Do NOT use outside knowledge.",
        "Return plain text. No JSON.",
        "Pick 1-2 trades total (max 2).",
        "Each symbol must appear at most once (one side only).",
        "IMPORTANT: Only output your PICKS, not full setups.",
        "Format MUST be exactly one pick per line: '<SYMBOL> <SIDE>' (example: 'BTC SHORT').",
        "SIDE must be LONG or SHORT.",
        "Choose the side that best matches the data; do not output both sides for a symbol.",
    ]

    return (
        "You are selecting the BEST 1-2 trade picks from Hyperliquid tracker data.\n"
        "Follow rules strictly.\n\nRules:\n- "
        + "\n- ".join(rules)
        + "\n\nInput JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


async def _llm_generate(payload: Dict[str, Any], *, agent_id: str, thinking: str, timeout_sec: float) -> Tuple[Optional[Dict[str, Any]], str]:
    session_id = f"best_trade_signals_{int(time.time())}_{os.getpid()}"
    return await openclaw_agent_turn(
        message=_llm_prompt(payload),
        session_id=session_id,
        agent_id=agent_id,
        thinking=thinking,
        timeout_sec=timeout_sec,
        openclaw_cmd=(os.getenv("OPENCLAW_CMD") or "openclaw").strip(),
    )


def _fallback_text(picks: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("BEST TRADE SIGNALS")
    lines.append("(fallback: LLM unavailable — using deterministic engine)")
    lines.append("")

    for i, p in enumerate(picks, start=1):
        sym = p["symbol"]
        side = p["side"]
        setup = _run_setup(sym, side)
        lines.append(f"#{i} {sym} {side}")
        lines.append(setup)
        lines.append("")

    return "\n".join(lines).strip()


def _parse_picks(text: str, allowed: List[str]) -> List[Tuple[str, str]]:
    allowed_set = set([s.upper() for s in allowed])
    out: List[Tuple[str, str]] = []
    used = set()
    for raw in (text or "").strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        sym = parts[0].upper().strip()
        side = parts[1].upper().strip()
        if sym not in allowed_set:
            continue
        if side not in {"LONG", "SHORT"}:
            continue
        if sym in used:
            continue
        used.add(sym)
        out.append((sym, side))
        if len(out) >= 2:
            break
    return out


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--agent", default=os.getenv("BEST_TRADE_AGENT_ID", "acp-alpha-analyst"))
    ap.add_argument("--thinking", default=os.getenv("BEST_TRADE_THINKING", "low"))
    ap.add_argument("--timeout", type=float, default=float(os.getenv("BEST_TRADE_TIMEOUT_SEC", "120")))
    args = ap.parse_args()

    max_out = max(1, min(2, int(args.max)))
    cand_n = max(3, min(30, int(args.candidates)))

    mids = _hl_all_mids()

    scored: List[Tuple[float, str, Dict[str, Any], float]] = []
    for p in SYMBOL_DIR.glob("*.json"):
        try:
            snap = json.loads(p.read_text())
            if not isinstance(snap, dict):
                continue
        except Exception:
            continue
        sym = str(snap.get("symbol") or p.stem).upper()
        sym = _resolve_symbol(sym, mids)
        if not sym or sym not in mids:
            continue
        live_price = _safe_float(mids.get(sym))
        if live_price is None or live_price <= 0:
            continue
        s = _score_snapshot(snap, live_price=live_price)
        scored.append((s, sym, snap, live_price))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:cand_n]

    # Build compact candidate list
    candidates = []
    for score, sym, snap, live_price in top:
        candidates.append({"score": score, "data": _compact(snap, sym=sym, live_price=live_price)})

    # Deterministic fallback picks: top N unique
    fallback_picks = []
    used = set()
    for score, sym, snap, live_price in top:
        if sym in used:
            continue
        used.add(sym)
        fallback_picks.append({"symbol": sym, "side": _pick_side(snap), "score": score})
        if len(fallback_picks) >= max_out:
            break

    payload = {
        "task": "best_trade_signals",
        "generated_at": _utc_iso(),
        "max_signals": max_out,
        "instruction": "Pick 1-2 symbols with a single side each. Output picks only (one per line): <SYMBOL> <SIDE>.",
        "candidates": candidates,
        "constraints": {
            "one_side_per_symbol": True,
            "max_signals": max_out,
            "data_only": True,
        },
    }

    meta, text = await _llm_generate(payload, agent_id=str(args.agent), thinking=str(args.thinking), timeout_sec=float(args.timeout))

    allowed_syms = [c["data"]["symbol"] for c in candidates if isinstance(c, dict) and isinstance(c.get("data"), dict) and c["data"].get("symbol")]
    picks: List[Tuple[str, str]] = []
    if text.strip():
        picks = _parse_picks(text, allowed=allowed_syms)

    # If LLM didn't give valid picks, use deterministic fallback
    if not picks:
        picks = [(p["symbol"], p["side"]) for p in fallback_picks[:max_out]]

    # Build deterministic final setups for the chosen picks
    out_lines: List[str] = []
    out_lines.append("BEST TRADE SIGNALS")
    out_lines.append(f"Generated: {_utc_iso()}")
    out_lines.append("")

    for i, (sym, side) in enumerate(picks[:max_out], start=1):
        out_lines.append(f"#{i} {sym} {side}")
        out_lines.append(_run_setup(sym, side))
        out_lines.append("")

    print("\n".join(out_lines).strip())
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
