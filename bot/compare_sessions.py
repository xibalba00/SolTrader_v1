"""
Joins session_rules.csv + trades.csv (or their backtest equivalents) so
you can look at the last few sessions' RULES and RESULTS together in one
place, without manually cross-referencing two files. This is the actual
tool for "adjust the rules considering the past couple sessions."

Usage:
    python compare_sessions.py                    # last 4 paper trading sessions
    python compare_sessions.py --n 3               # last 3 sessions
    python compare_sessions.py --mode backtest      # compare backtest sessions instead
"""

import argparse
import csv
import os

from bot.trade_logger import stats_from_rows


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, help="How many most-recent sessions to compare")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "backtest"])
    args = parser.parse_args()

    if args.mode == "paper":
        rules_path, trades_path = "logs/session_rules.csv", "logs/trades.csv"
    else:
        rules_path, trades_path = "logs/backtest_session_rules.csv", "logs/backtest_trades.csv"

    rules_rows = read_csv(rules_path)
    trade_rows = read_csv(trades_path)

    if not rules_rows:
        print(f"No sessions found in {rules_path} yet.")
        return

    recent_sessions = rules_rows[-args.n:]

    for session in recent_sessions:
        sid = session["session_id"]
        this_session_trades = [r for r in trade_rows if r.get("session_id") == sid]
        stats = stats_from_rows(this_session_trades)

        print("=" * 70)
        print(f"SESSION: {sid}  (started {session.get('session_start_utc', '?')})")
        print("-" * 70)
        buy_fields = {k[4:]: v for k, v in session.items() if k.startswith("buy_")}
        sell_fields = {k[5:]: v for k, v in session.items() if k.startswith("sell_")}
        risk_fields = {k[5:]: v for k, v in session.items() if k.startswith("risk_")}
        print(f"  Buy:  {buy_fields}")
        print(f"  Sell: {sell_fields}")
        print(f"  Risk: {risk_fields}")
        print(f"  Results: {stats}")

    print("=" * 70)
    print(f"Compared {len(recent_sessions)} session(s). Look for which rule changes")
    print("between sessions correlate with better/worse win_rate_pct and expectancy_pct")
    print("before making the next adjustment.")


if __name__ == "__main__":
    main()
