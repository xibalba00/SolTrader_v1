# Smart-Money Radar — separate experiment, separate directory

This is the "Historical Trader-Weighted Market Universe" idea: instead of
sourcing backtest candidates from today's GeckoTerminal trending list,
source them from tokens that a scored cohort of active/profitable Solana
wallets actually bought. Explicitly kept in its own directory/repo per your
call — nothing here touches `bot/config.py`, `bot/paper_engine.py`, or the
main trade logs.

Labeled outputs from this pipeline as **"smart-money-cohort performance,"**
never blended into the main strategy's win-rate/expectancy numbers.

## Why this is two scripts, not one

Vybe's free tier is **4 requests/minute, 12,000 credits/month**. That's the
entire design constraint. Everything here is a slow, resumable batch job —
not something you run inline in the live paper-trading loop.

## Pipeline

```
seed_tokens.txt (token mints — pull from your existing dexscreener_client
                  discovery, or GeckoTerminal trending, or hand-picked)
        |
        v  wallet_radar_phase1.py
Vybe /v4/tokens/{mint}/top-pnl-traders, per seed token
        |
        v
logs/wallet_appearances.csv  (raw, append-only)
        |
        v  aggregate + score
logs/wallet_cohort.csv   <-  scored, categorized wallet panel
        |
        v  wallet_radar_phase2.py
Vybe /v4/trades, per monitored wallet
        |
        v
logs/wallet_trades_raw.csv  (raw, append-only)
        |
        v  aggregate
logs/token_discovery.csv   <-  candidate tokens + discovery_score +
                                first_monitored_buy_utc
        |
        v  (not built yet — see "Next step" below)
GeckoTerminal OHLCV (before_timestamp) -> backtest_engine.py
```

## Time/credit budget, roughly

- Phase 1, 50 seed tokens: 50 calls, ~12-13 min, ~50-100 credits (top-pnl-traders
  is likely a cheap call).
- Phase 2, 300 wallets: 300 calls, ~75 min, maybe 300-600 credits (trade-history
  calls are probably pricier — unconfirmed, the client assumes 2 credits/call
  as a placeholder, tighten this once you see real usage in Vybe's dashboard).

Both scripts checkpoint after every successful call (`logs/phase*_processed_*.txt`)
so an SSH drop or a Ctrl+C mid-run costs you nothing — rerun the same command
and it picks up where it left off, skipping already-harvested tokens/wallets.

## Before running for real

1. **Verify the Vybe endpoints.** `bot/vybe_client.py` has `VERIFY:` comments
   on the base URL, auth header name, and the wallet-trade-history endpoint
   specifically — those were the least consistently documented across public
   sources. Hit each one manually with curl once, using your real key, and
   confirm the response shape matches what the client parses before trusting
   a multi-hour harvest run.
2. **Start tiny.** Run Phase 1 with ~10 seed tokens and `--credit-budget 200`
   first, look at `logs/wallet_cohort.csv`, sanity-check it's not just
   returning the same 5 bot/MM wallets on every token, before committing a
   real credit budget to a full run.
3. **Set `VYBE_API_KEY`** as an environment variable before running either
   script.

## Next step (not built here)

`token_discovery.csv` gives you `token_mint` + `first_monitored_buy_utc`.
The remaining piece is a small adapter that:
  - looks up the pool address for each mint (GeckoTerminal search or
    DexScreener `get_token_pairs`),
  - pulls OHLCV via `geckoterminal_client.py`'s existing `before_timestamp`
    support,
  - **truncates candles to `first_monitored_buy_utc` onward** — this is the
    information-timing guard from the original write-up (section 12).
    Feeding the strategy candles from before any monitored wallet touched
    the token would leak information a live scan wouldn't have had.
  - runs it through `backtest_engine.py` with `mode="smart_money_backtest"`
    and a distinct log path (e.g. `logs/smart_money_trades.csv`), so results
    never mix with the main paper/backtest numbers.

Worth building only after Phase 1 + Phase 2 produce a `token_discovery.csv`
that looks sane on manual inspection — no point wiring up the backtest
adapter against data you haven't eyeballed yet.
