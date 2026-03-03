#!/usr/bin/env python3
"""Generate a normie-friendly Alpha Dashboard analysis (advisory-only).

Reads latest hl_trader_context_*.json via build_alpha_context.py and asks a
named OpenClaw agent (acp-alpha-analyst) for a market overview + directional bias.

Output: plain text (for chat / ACP deliverable).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

HL_TRADER_DIR = os.getenv("HL_TRADER_DIR", "./hl-trader")
if HL_TRADER_DIR not in sys.path:
    sys.path.insert(0, HL_TRADER_DIR)

from openclaw_agent_client import openclaw_agent_turn  # type: ignore


def _utc_iso(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_context_json() -> Dict[str, Any]:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_alpha_context.py")
    out = subprocess.check_output(["python3", script], text=True)
    return json.loads(out)


def _prompt(payload: Dict[str, Any]) -> str:
    focus = str(payload.get("focus") or "").strip()

    rules = [
        "Advisory-only. Do NOT give execution instructions.",
        "Do NOT mention order types (no 'limit/market/chase', no FAST/RESTING).",
        "Be normie-friendly, concise, and confident-but-humble.",
        "Always start with a TL;DR (1-2 short lines).",
        "Then include a single-line: Bias: LONG|SHORT|NEUTRAL (weak|moderate|strong).",
        "Then include labeled sections with bullets:",
        "- What’s driving it (3-6 bullets)",
        "- What to watch / what changes the bias (3-6 bullets)",
        "Keep it skimmable; avoid long paragraphs.",
        "Optionally include Top bullish / Top bearish tickers (max 5 each) if present in data.",
        "If data is stale (>10 minutes), explicitly warn at the top.",
    ]

    if focus:
        rules.append(f"User focus: '{focus}'. Tilt the writeup toward this focus (best-effort), but DO NOT hallucinate data not present in the JSON.")

    return (
        "You are acp-alpha-analyst. Produce an Alpha Dashboard summary from this JSON.\n"
        "Return plain text (no JSON, no markdown tables).\n\n"
        + "Rules:\n- "
        + "\n- ".join(rules)
        + "\n\nInput JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thinking", default=os.getenv("ACP_ALPHA_THINKING", "medium"))
    ap.add_argument("--agent", default=os.getenv("ACP_ALPHA_AGENT_ID", "acp-alpha-analyst"))
    ap.add_argument("--timeout", type=float, default=float(os.getenv("ACP_ALPHA_TIMEOUT_SEC", "45")))
    ap.add_argument("--focus", default=os.getenv("ACP_ALPHA_FOCUS", ""))
    args = ap.parse_args()

    focus = str(args.focus or "").strip()

    ctx = _build_context_json()
    msg = {
        "task": "alpha_dashboard",
        "generated_at": _utc_iso(),
        "focus": focus or None,
        "alpha_context": ctx,
    }

    session_id = f"acp_alpha_{int(time.time())}_{os.getpid()}"
    meta, text = await openclaw_agent_turn(
        message=_prompt(msg),
        session_id=session_id,
        agent_id=str(args.agent),
        thinking=str(args.thinking),
        timeout_sec=float(args.timeout),
        openclaw_cmd=(os.getenv("OPENCLAW_CMD") or "openclaw").strip(),
    )

    if not text.strip():
        # Keep failure visible to caller.
        err = {
            "ok": False,
            "error": "empty_llm_output",
            "meta": meta,
        }
        print(json.dumps(err, indent=2, sort_keys=True))
        return 2

    print(text.strip())
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
