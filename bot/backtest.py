"""
Run a backtest against real historical Solana pool data.

Usage:
    python backtest.py --pages 2
    python backtest.py --timeframe minute --aggregate 15
    python backtest.py --dates 2026-08-10,2026-08-05,2026-08-01,2026-07-28,2026-07-23

Pulls candidate pools from GeckoTerminal's trending + top-volume lists
(today's snapshot — see bot/backtest_engine.py docstring for the
survivorship-bias caveat this implies), fetches each pool's real
historical candles, and runs your current config.py buy/sell rules
against that history. Logs every simulated trade to
logs/backtest_trades.csv (same format as paper trading).

Reports THIS RUN's stats separately from the full historical log file,
so a run that logs zero new trades can't silently show you stale
numbers from a previous run.

--dates filters this run's trades to specific calendar entry dates and
prints a per-date breakdown plus a combined total across just those
days. Dates older than a pool's available history simply won't have
any trades — that's expected, not an error.
"""

import argparse
from datetime import datetime, timezone
from collections import defaultdict

from bot.config import CONFIG
from bot.geckoterminal_client import GeckoTerminalClient
from bot.backtest_engine import BacktestEngine
from bot.trade_logger import stats_from_rows, log_session_rules

# GeckoTerminal only accepts these specific aggregate values per timeframe.
# Anything else 400s — validating here fails fast with ONE clear message
# instead of the same error repeated across every pool.
VALID_AGGREGATES = {
    "day": [1],
    "hour": [1, 4, 12],
    "minute": [1, 5, 15],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1, help="Pages of trending+top pools to pull as candidates (20 pools/page)")
    parser.add_argument("--timeframe", type=str, default="hour", choices=["day", "hour", "minute"])
    parser.add_argument("--aggregate", type=int, default=1, help="Candle aggregation. Valid: day=1, hour=1/4/12, minute=1/5/15")
    parser.add_argument("--dates", type=str, default=None, help="Comma-separated YYYY-MM-DD dates to summarize this run's trades by (UTC entry date)")
    args = parser.parse_args()

    valid = VALID_AGGREGATES[args.timeframe]
    if args.aggregate not in valid:
        print(f"ERROR: aggregate={args.aggregate} isn't valid for timeframe='{args.timeframe}'.")
        print(f"Valid aggregates for '{args.timeframe}': {valid}")
        print("(GeckoTerminal has no native 30-minute candle — closest is minute/15.)")
        return

    gecko = GeckoTerminalClient()
    session_id = datetime.now(timezone.utc).strftime("backtest_%Y%m%d_%H%M%S")
    log_session_rules(CONFIG.buy, CONFIG.sell, CONFIG.risk, session_id=session_id, rules_log_path="logs/backtest_session_rules.csv")
    print(f"Session ID: {session_id}  (also written to logs/backtest_session_rules.csv — this is the join key)")

    print("Fetching candidate pools (trending + top volume)...")
    pools = gecko.get_trending_pools(pages=args.pages) + gecko.get_top_pools(pages=args.pages)
    seen = set()
    unique_pools = []
    for p in pools:
        addr = p.get("attributes", {}).get("address")
        if addr and addr not in seen:
            seen.add(addr)
            unique_pools.append(p)
    print(f"Backtesting {len(unique_pools)} unique pools...")

    engine = BacktestEngine(session_id=session_id)
    print(f"Buy params: {CONFIG.buy}")
    print(f"Sell params: {CONFIG.sell}")
    print(f"Candles: {args.timeframe}/{args.aggregate}")
    print("-" * 60)

    rows_before = len(engine.logger.read_all())  # so we can isolate THIS run's rows after

    for idx, pool in enumerate(unique_pools, 1):
        name = pool.get("attributes", {}).get("name", "?")
        try:
            n = engine.run_pool(pool, timeframe=args.timeframe, aggregate=args.aggregate)
            if n:
                print(f"[{idx}/{len(unique_pools)}] {name}: {n} trade(s)")
        except Exception as e:
            print(f"[{idx}/{len(unique_pools)}] {name}: skipped ({e})")

    all_rows = engine.logger.read_all()
    session_rows = all_rows[rows_before:]  # only what THIS run actually logged

    print("-" * 60)
    if not session_rows:
        print("This run logged ZERO new trades (see errors above if unexpected — often an invalid")
        print("timeframe/aggregate combo, or every candidate simply never passed the buy filters).")
        print(f"Historical log (all runs combined) still has {len(all_rows)} trade(s) — unchanged by this run.")
        return

    print("This run's stats:", stats_from_rows(session_rows))
    print(f"Ending simulated balance: ${engine.balance_usd:.2f}")
    print("Full trade log: logs/backtest_trades.csv")

    if args.dates:
        requested_dates = set(d.strip() for d in args.dates.split(","))
        by_date = defaultdict(list)
        for r in session_rows:
            entry_date = r["entry_time_utc"][:10]  # YYYY-MM-DD prefix of ISO timestamp
            if entry_date in requested_dates:
                by_date[entry_date].append(r)

        print("-" * 60)
        print("Per-date breakdown (requested dates only):")
        combined = []
        for d in sorted(requested_dates):
            rows_for_date = by_date.get(d, [])
            combined.extend(rows_for_date)
            print(f"  {d}: {stats_from_rows(rows_for_date)}")
        print(f"Combined across all requested dates: {stats_from_rows(combined)}")


if __name__ == "__main__":
    main()
