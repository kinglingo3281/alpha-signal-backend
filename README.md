# Alpha Signals Backend

Standalone Python scripts that turn local market data into actionable perp analysis. Each script reads from Symbol Watcher tracker files and (optionally) an LLM, then outputs plain text or JSON suitable for piping into an ACP seller agent or using directly from the command line.

## Scripts

| Script | Purpose | Needs LLM? |
|---|---|---|
| `build_alpha_context.py` | Condenses the latest trader-context artifact into a compact JSON payload for downstream consumers | No |
| `generate_alpha_dashboard.py` | Writes a readable market overview — bias direction, what's driving it, what to watch | Yes |
| `generate_best_trade_signals.py` | Ranks all tracked symbols by a composite score and returns setups for the top 1–2 picks | Yes (deterministic fallback) |
| `generate_symbol_setup.py` | Produces a single-symbol directional view with entry, stop, targets, and reasoning | Yes (deterministic fallback) |

## Input Data

### Per-Symbol Tracker Snapshots (`TRACKER_SYMBOL_DIR`)

One JSON file per monitored symbol, each containing sections like open-interest summary, liquidation heatmap, smart/dumb money divergence, whale-vs-retail cohort delta, funding rate, ATR volatility, and per-indicator signal votes.

### Trader Context Snapshots (`HL_TRADER_STATE_DIR`)

Timestamped `hl_trader_context_*.json` files with directional calls, scored trend rankings, a compact global summary, and selected opportunity lists.

### LLM Integration

Scripts call `openclaw_agent_turn` (from an external `openclaw_agent_client` module pointed to by `HL_TRADER_DIR`). If the LLM is unreachable or times out, every script falls back to a deterministic engine so output is always produced.

## Getting Started

```bash
pip install -r requirements.txt
cp .env.example .env   # point paths to your data directories
```

Run any script directly:

```bash
python3 scripts/build_alpha_context.py
python3 scripts/generate_symbol_setup.py ETH --side LONG
python3 scripts/generate_best_trade_signals.py --max 2
python3 scripts/generate_alpha_dashboard.py --focus "funding rates"
```

## Configuration

All tunables are in `.env`. See `.env.example` for the full list with comments.

## License

MIT
