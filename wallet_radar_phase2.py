"""
Phase 2 of the smart-money-cohort experiment: turn the Phase-1 wallet
cohort into a token-discovery table.

    logs/wallet_cohort.csv (from phase 1)
        |
        v
    for each monitored wallet: Vybe /v4/trades (buy events, real timestamps)
        |
        v
    logs/wallet_trades_raw.csv  (append-only, resumable)
        |
        v
    aggregate per token: how many monitored wallets bought it, how much
    volume, how early relative to each other
        |
        v
    logs/token_discovery.csv  <-  this is the candidate list to feed into
    backtest_engine.py / GeckoTerminal historical OHLCV

IMPORTANT — information-timing constraint (see idea write-up section 12):
a token only counts as "discovered" as of the EARLIEST monitored-wallet buy
timestamp seen for it. When you later feed a discovered token into the
backtest engine, only use OHLCV from that discovery timestamp onward —
using the full history (including candles before any monitored wallet
touched it) would leak information your live strategy wouldn't have had.
`first_monitored_buy_utc` in the output exists specifically so the backtest
step can enforce that cutoff.

Usage:
    python wallet_radar_phase2.py --cohort-csv logs/wallet_cohort.csv --top-n 300
    python wallet_radar_phase2.py --cohort-csv logs/wallet_cohort.csv --top-n 300 --categories high_pnl,high_volume
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

from bot.vybe_client import VybeClient

RAW_TRADES_CSV = "logs/wallet_trades_raw.csv"
DISCOVERY_CSV = "logs/token_discovery.csv"
CHECKPOINT_FILE = "logs/phase2_processed_wallets.txt"

RAW_TRADE_FIELDS = [
    "wallet_address", "token_mint", "trade_timestamp_utc", "side",
    "amount_usd", "program", "harvested_at_utc",
]

DISCOVERY_FIELDS = [
    "token_mint", "monitored_wallets_count", "total_volume_usd",
    "trade_count", "first_monitored_buy_utc", "discovery_score",
]


def load_cohort(cohort_csv: str, top_n: int, categories: list[str] | None) -> list[str]:
    with open(cohort_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if categories:
        rows = [r for r in rows if any(c in (r.get("category") or "").split("|") for c in categories)]
    rows.sort(key=lambda r: float(r["trader_score"]), reverse=True)
    return [r["wallet_address"] for r in rows[:top_n]]


def load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def mark_processed(wallet: str) -> None:
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(wallet + "\n")


def append_raw_trades(rows: list[dict]) -> None:
    needs_header = not os.path.exists(RAW_TRADES_CSV)
    os.makedirs(os.path.dirname(RAW_TRADES_CSV), exist_ok=True)
    with open(RAW_TRADES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_TRADE_FIELDS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def harvest_trades(wallets: list[str], client: VybeClient, trades_per_wallet: int) -> None:
    already_done = load_checkpoint()
    todo = [w for w in wallets if w not in already_done]
    print(f"{len(already_done)} wallets already harvested (skipping), {len(todo)} remaining this run.")

    for i, wallet in enumerate(todo, 1):
        try:
            trades = client.get_wallet_trades(wallet, limit=trades_per_wallet)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {wallet}: skipped ({e})")
            continue

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows = []
        for t in trades:
            # VERIFY: field names against your actual /v4/trades response —
            # this maps the most commonly-documented shape but Vybe's raw
            # trade schema wasn't confirmed as precisely as top-pnl-traders.
            side = (t.get("side") or t.get("type") or "").lower()
            if side not in ("buy", "sell"):
                continue
            rows.append({
                "wallet_address": wallet,
                "token_mint": t.get("mintAddress") or t.get("tokenMint") or t.get("baseMint"),
                "trade_timestamp_utc": t.get("blockTime") or t.get("timestamp"),
                "side": side,
                "amount_usd": float(t.get("valueUsd") or t.get("amountUsd") or 0),
                "program": t.get("programName") or t.get("dex") or "",
                "harvested_at_utc": now,
            })
        append_raw_trades([r for r in rows if r["token_mint"]])
        mark_processed(wallet)
        print(f"[{i}/{len(todo)}] {wallet}: {len(rows)} buy/sell events (credits spent so far: {client.limiter.credits_spent})")


@dataclass
class TokenAgg:
    wallets: set = field(default_factory=set)
    total_volume: float = 0.0
    trade_count: int = 0
    timestamps: list = field(default_factory=list)


def build_discovery_table(min_wallets: int) -> None:
    if not os.path.exists(RAW_TRADES_CSV):
        print(f"No {RAW_TRADES_CSV} yet — run the harvest step first.")
        return

    agg: dict[str, TokenAgg] = defaultdict(TokenAgg)
    with open(RAW_TRADES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["side"] != "buy":
                continue
            t = agg[row["token_mint"]]
            t.wallets.add(row["wallet_address"])
            t.total_volume += float(row["amount_usd"] or 0)
            t.trade_count += 1
            if row["trade_timestamp_utc"]:
                t.timestamps.append(row["trade_timestamp_utc"])

    if not agg:
        print("No buy events recorded yet.")
        return

    counts = [len(t.wallets) for t in agg.values()]
    volumes = [t.total_volume for t in agg.values()]
    max_count, max_volume = max(counts) or 1, max(volumes) or 1

    rows = []
    for mint, t in agg.items():
        if len(t.wallets) < min_wallets:
            continue
        # Simple blended score: half wallet-count coverage, half volume —
        # both normalized against this batch's own max so the score is
        # comparable across discovery runs of different sizes. Refine once
        # you can check this against actual forward returns.
        score = 50 * (len(t.wallets) / max_count) + 50 * (t.total_volume / max_volume)
        rows.append({
            "token_mint": mint,
            "monitored_wallets_count": len(t.wallets),
            "total_volume_usd": round(t.total_volume, 2),
            "trade_count": t.trade_count,
            "first_monitored_buy_utc": min(t.timestamps) if t.timestamps else "",
            "discovery_score": round(score, 2),
        })

    rows.sort(key=lambda r: r["discovery_score"], reverse=True)

    with open(DISCOVERY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DISCOVERY_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} candidate tokens (>= {min_wallets} monitored wallets) to {DISCOVERY_CSV}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-csv", type=str, default="logs/wallet_cohort.csv")
    parser.add_argument("--top-n", type=int, default=300, help="How many wallets from the cohort to pull trade history for")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated: only pull wallets tagged with these categories (default: all)")
    parser.add_argument("--trades-per-wallet", type=int, default=200)
    parser.add_argument("--min-wallets", type=int, default=3, help="Drop tokens seen by fewer than this many monitored wallets")
    parser.add_argument("--credit-budget", type=int, default=3000)
    parser.add_argument("--skip-harvest", action="store_true")
    args = parser.parse_args()

    if not args.skip_harvest:
        client = VybeClient(credit_budget=args.credit_budget)
        categories = args.categories.split(",") if args.categories else None
        wallets = load_cohort(args.cohort_csv, args.top_n, categories)
        print(f"Loaded {len(wallets)} wallets from cohort. Free tier = 4 req/min, ~{len(wallets) * 15 / 60:.1f} "
              f"minutes minimum for a fresh pass (trade-history calls cost more credits per call than top-traders).")
        harvest_trades(wallets, client, args.trades_per_wallet)

    build_discovery_table(args.min_wallets)


if __name__ == "__main__":
    main()
