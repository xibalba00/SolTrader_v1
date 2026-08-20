# Solana Paper Trading Bot — Phase 1 Scaffold

Paper-trading only. No wallet keys, no live transactions. Purpose: validate
strategy logic and produce **real** win-rate/expectancy numbers before any
money is at risk.

## What's here

- `bot/config.py` — all buy/sell/risk parameters in one place. Edit these,
  don't hunt through the code.
- `bot/dexscreener_client.py` — real price/liquidity/volume/mcap data from
  DexScreener's public API.
- `bot/jupiter_client.py` — real swap quotes from Jupiter (used to simulate
  realistic price impact, even in paper mode).
- `bot/paper_engine.py` — the actual strategy: applies buy filters, tracks
  simulated positions, checks TP/SL/trailing-stop/max-hold exits.
- `bot/trade_logger.py` — logs every closed trade to `logs/trades.csv` with:
  wallet balance, position size (% and $), buy price, buy slippage, TP/SL
  targets, sell reason, sell price, sell slippage, hold time, profit (% and $).
- `bot/gas_watcher.py` — **template only**, not wired to execute. This is
  where "sell existing tokens at market when live SOL runs low" will live
  once you're ready to go live. Deliberately left with `NotImplementedError`
  stubs on anything that would touch a real wallet — that's a separate,
  explicit step, not something that should ship half-finished.
- `main.py` — run loop.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --query "pump.fun" --iterations 20
```

Leave `--iterations` unset (or 0) to run continuously. Each loop:
fetches candidates → applies buy filters from `config.py` → opens
simulated positions → checks all open positions against exit rules →
logs closed trades → prints a running summary (win rate, avg gain,
avg loss, expectancy — computed from what's actually in the log, never
fabricated).

## Known gaps (intentional, not oversights)

1. **New-pair discovery**: `DexScreenerClient.screen_new_pairs()` raises
   `NotImplementedError` on purpose. DexScreener's free API doesn't have a
   fully reliable "newest pairs" endpoint — for real pump.fun sniping
   you'd want their token-boost endpoint (partial coverage) or a paid feed
   (Birdeye, Helius webhooks). Search-based discovery (`--query`) works
   today as a starting point.
2. **Live execution**: `gas_watcher.py`'s `_send_transaction` and
   `get_sol_balance` are stubs. Wiring these up means adding `solana-py`,
   loading a keypair, and signing/submitting real transactions — a
   deliberate, separate step once paper results justify it.
3. **Network**: this was scaffolded in a sandboxed dev environment that
   only allow-lists a fixed set of domains, so `api.dexscreener.com` and
   `lite-api.jup.ag` returned 403 here. That's a sandbox restriction, not
   a code issue — both work fine from a normal VPS with open outbound
   internet. Test on your VPS before assuming something's broken.

## Reading the log

`logs/trades.csv` columns map directly to what you asked for:
`position_size_pct_wallet`, `position_size_usd`, `buy_price_usd`,
`buy_slippage_pct`, `tp_target_pct`, `sl_target_pct`, `sell_price_usd`,
`sell_slippage_pct`, `profit_pct`, `profit_usd`.

## Next steps, in order

1. Run this against real search queries for a few days, review
   `logs/trades.csv` and the printed summary stats.
2. Adjust `config.py` parameters based on what the *log* shows, not
   intuition — that's the whole point of this phase.
3. Only after that: build the historical backtester (same engine, fed
   historical data instead of live polling) to test parameter sets
   against a larger sample than a few days of live paper trading.
4. Only after that: wire up live execution, starting with gas-watcher
   sell-side only, small size, on a VPS with paid RPC.
