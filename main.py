"""
Entry point for paper trading.

Usage:
    python main.py                          # real discovery: token-profiles + boosts, filtered to Solana
    python main.py --query "some token name" # fallback: keyword search instead of discovery
    python main.py --iterations 20           # run 20 loops instead of forever

Discovery mode (default) pulls candidates from DexScreener's actual
token-profiles and token-boosts endpoints, filters to Solana, and fetches
full pair metrics for each — this is the real "find tradeable tokens"
path, not a placeholder. See dexscreener_client.discover_candidate_pairs().

This runs the paper engine in a loop: fetch candidates, apply buy
filters, open simulated positions, check open positions against TP/SL/
trailing-stop/max-hold rules, log every closed trade, print a running
stats summary.

If the SAME error repeats several times in a row, the loop stops itself
instead of silently retrying forever — a real incident (June 2026) ran
for two days straight failing every single iteration on a missing
method, with the try/except swallowing it quietly the whole time.
That should never happen silently again.

Nothing in this file touches a real wallet.
"""

import argparse
import os
import time
import traceback
from datetime import datetime, timezone

from bot.config import CONFIG
from bot.dexscreener_client import DexScreenerClient
from bot.paper_engine import PaperEngine
from bot.trade_logger import TradeLogger, log_session_rules

MAX_CONSECUTIVE_FAILURES = 5
PAPER_LOG_PATH = "logs/trades.csv"


def start_fresh_log(log_path: str) -> None:
    """Every fresh `python main.py` launch resets the paper balance to
    starting_balance_usd in memory — but the trade log on disk previously
    kept accumulating across restarts, so status() silently mixed old and
    new trades together and looked out of sync with the reset balance.
    Archiving the old log (not deleting — nothing is lost) keeps balance
    and trade history consistent every time you restart."""
    if os.path.exists(log_path):
        archived = log_path.replace(".csv", f"_archived_{int(time.time())}.csv")
        os.rename(log_path, archived)
        print(f"Archived previous session's log to '{archived}' — starting fresh to match the reset ${CONFIG.risk.starting_balance_usd:.2f} balance.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Use keyword search instead of real discovery")
    parser.add_argument("--iterations", type=int, default=0, help="0 = run forever")
    args = parser.parse_args()

    start_fresh_log(PAPER_LOG_PATH)
    session_id = datetime.now(timezone.utc).strftime("paper_%Y%m%d_%H%M%S")
    log_session_rules(CONFIG.buy, CONFIG.sell, CONFIG.risk, session_id=session_id)
    engine = PaperEngine(session_id=session_id)
    dex = DexScreenerClient(chain=CONFIG.dexscreener_chain)

    print(f"Session ID: {session_id}  (also written to logs/session_rules.csv — this is the join key)")
    print(f"Starting paper trading. Balance: ${engine.balance_usd:.2f}")
    print(f"Buy params: {CONFIG.buy}")
    print(f"Sell params: {CONFIG.sell}")
    print(f"Discovery mode: {'keyword search' if args.query else 'token-profiles + boosts (Solana)'}")
    print("-" * 60)

    i = 0
    consecutive_failures = 0
    while True:
        i += 1
        try:
            if args.query:
                candidates = dex.search_pairs(args.query)
            else:
                candidates = dex.discover_candidate_pairs()
            engine.run_once(candidates)
            status = engine.status()
            print(f"[iter {i}] candidates={len(candidates)} {status}")
            consecutive_failures = 0  # reset on any success
        except Exception:
            consecutive_failures += 1
            print(f"[iter {i}] error #{consecutive_failures} in a row:")
            traceback.print_exc()  # full traceback, not just str(e) — so the real cause is visible immediately
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"STOPPING: {MAX_CONSECUTIVE_FAILURES} consecutive failures. "
                      f"This is almost certainly a real bug, not a transient network blip — fix it before restarting.")
                return

        if args.iterations and i >= args.iterations:
            break
        time.sleep(CONFIG.poll_interval_seconds)


if __name__ == "__main__":
    main()
