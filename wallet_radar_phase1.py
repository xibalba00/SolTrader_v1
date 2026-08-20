"""
Phase 1 of the smart-money-cohort experiment: turn a seed list of tokens
into a scored wallet cohort.

    seed tokens (from DexScreener/GeckoTerminal, or a manual list)
        |
        v
    for each token: Vybe top-pnl-traders  ->  logs/wallet_appearances.csv
        |
        v
    aggregate per wallet across all tokens seen
        |
        v
    score + write logs/wallet_cohort.csv

Design constraint: Vybe's free tier is 4 req/min / 12,000 credits/month.
50 seed tokens = 50 calls = ~12.5 minutes minimum. This script is built to
be interrupted (Ctrl+C, SSH drop, whatever) and resumed without re-spending
credits on tokens it already processed — that's the whole point of the
checkpoint file below, not just a nicety.

Usage:
    python wallet_radar_phase1.py --seed-file seed_tokens.txt
    python wallet_radar_phase1.py --seed-file seed_tokens.txt --credit-budget 3000

seed_tokens.txt: one Solana token mint address per line.
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

from bot.vybe_client import VybeClient

APPEARANCES_CSV = "logs/wallet_appearances.csv"
COHORT_CSV = "logs/wallet_cohort.csv"
CHECKPOINT_FILE = "logs/phase1_processed_tokens.txt"

APPEARANCE_FIELDS = [
    "wallet_address", "token_mint", "realized_pnl_usd", "volume_usd",
    "trade_count", "win_rate_pct", "harvested_at_utc",
]

COHORT_FIELDS = [
    "wallet_address", "tokens_seen", "total_trades", "total_volume_usd",
    "total_realized_pnl_usd", "avg_win_rate_pct",
    "score_activity", "score_volume", "score_pnl", "score_diversity",
    "trader_score", "category",
]


def load_seed_tokens(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def mark_processed(token_mint: str) -> None:
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(token_mint + "\n")


def append_appearances(rows: list[dict]) -> None:
    needs_header = not os.path.exists(APPEARANCES_CSV)
    os.makedirs(os.path.dirname(APPEARANCES_CSV), exist_ok=True)
    with open(APPEARANCES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=APPEARANCE_FIELDS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def harvest(seed_tokens: list[str], client: VybeClient, top_n_per_token: int) -> None:
    already_done = load_checkpoint()
    todo = [t for t in seed_tokens if t not in already_done]
    print(f"{len(already_done)} tokens already harvested (skipping), {len(todo)} remaining this run.")

    for i, mint in enumerate(todo, 1):
        try:
            traders = client.get_top_pnl_traders(mint, limit=top_n_per_token)
        except Exception as e:
            # A single bad/delisted mint shouldn't kill an hour-long harvest run.
            print(f"[{i}/{len(todo)}] {mint}: skipped ({e})")
            continue

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows = [{
            "wallet_address": t.wallet_address,
            "token_mint": t.token_mint,
            "realized_pnl_usd": t.realized_pnl_usd,
            "volume_usd": t.volume_usd,
            "trade_count": t.trade_count,
            "win_rate_pct": t.win_rate_pct if t.win_rate_pct is not None else "",
            "harvested_at_utc": now,
        } for t in traders if t.wallet_address]
        append_appearances(rows)
        mark_processed(mint)
        print(f"[{i}/{len(todo)}] {mint}: {len(rows)} traders (credits spent so far: {client.limiter.credits_spent})")


@dataclass
class WalletAgg:
    tokens: set = field(default_factory=set)
    total_trades: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    win_rates: list = field(default_factory=list)


def aggregate_and_score(top_n_per_category: int) -> None:
    if not os.path.exists(APPEARANCES_CSV):
        print(f"No {APPEARANCES_CSV} yet — run the harvest step first.")
        return

    agg: dict[str, WalletAgg] = defaultdict(WalletAgg)
    with open(APPEARANCES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            w = agg[row["wallet_address"]]
            w.tokens.add(row["token_mint"])
            w.total_trades += int(row["trade_count"] or 0)
            w.total_volume += float(row["volume_usd"] or 0)
            w.total_pnl += float(row["realized_pnl_usd"] or 0)
            if row["win_rate_pct"]:
                w.win_rates.append(float(row["win_rate_pct"]))

    if not agg:
        print("No wallet appearances recorded yet.")
        return

    # Min-max normalize each raw metric across the whole cohort before
    # weighting — otherwise volume_usd (which can span orders of magnitude)
    # would dominate trade_count and PnL purely on scale, not on what it
    # actually signals.
    def normalize(values: list[float]) -> dict:
        lo, hi = min(values), max(values)
        if hi == lo:
            return {v: 0.5 for v in values}
        return {v: (v - lo) / (hi - lo) for v in values}

    trades_list = [w.total_trades for w in agg.values()]
    volume_list = [w.total_volume for w in agg.values()]
    pnl_list = [w.total_pnl for w in agg.values()]
    diversity_list = [len(w.tokens) for w in agg.values()]

    n_trades = normalize(trades_list)
    n_volume = normalize(volume_list)
    n_pnl = normalize(pnl_list)
    n_diversity = normalize(diversity_list)

    scored = []
    for wallet, w in agg.items():
        s_activity = n_trades[w.total_trades]
        s_volume = n_volume[w.total_volume]
        s_pnl = n_pnl[w.total_pnl]
        s_diversity = n_diversity[len(w.tokens)]
        # Weights from the original proposal — treat as a starting point,
        # not a validated constant. Worth re-deriving once you have enough
        # backtest results to see which component actually predicts
        # forward performance.
        trader_score = (0.25 * s_activity) + (0.30 * s_volume) + (0.30 * s_pnl) + (0.15 * s_diversity)
        scored.append({
            "wallet_address": wallet,
            "tokens_seen": len(w.tokens),
            "total_trades": w.total_trades,
            "total_volume_usd": round(w.total_volume, 2),
            "total_realized_pnl_usd": round(w.total_pnl, 2),
            "avg_win_rate_pct": round(sum(w.win_rates) / len(w.win_rates), 2) if w.win_rates else "",
            "score_activity": round(s_activity, 4),
            "score_volume": round(s_volume, 4),
            "score_pnl": round(s_pnl, 4),
            "score_diversity": round(s_diversity, 4),
            "trader_score": round(trader_score, 4),
        })

    # Category tags: independent rankings by the four raw dimensions, mirroring
    # the "several classes, overlap allowed" idea rather than one blended sort.
    # A wallet can (and often should) land in more than one category.
    def top_by(key: str) -> set:
        return {r["wallet_address"] for r in sorted(scored, key=lambda r: r[key], reverse=True)[:top_n_per_category]}

    frequent = top_by("total_trades")
    high_volume = top_by("total_volume_usd")
    high_pnl = top_by("total_realized_pnl_usd")
    diversified = top_by("tokens_seen")

    for r in scored:
        cats = []
        if r["wallet_address"] in frequent:
            cats.append("high_frequency")
        if r["wallet_address"] in high_volume:
            cats.append("high_volume")
        if r["wallet_address"] in high_pnl:
            cats.append("high_pnl")
        if r["wallet_address"] in diversified:
            cats.append("diversified")
        r["category"] = "|".join(cats) if cats else ""

    scored.sort(key=lambda r: r["trader_score"], reverse=True)

    with open(COHORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COHORT_FIELDS)
        writer.writeheader()
        for r in scored:
            writer.writerow(r)

    in_any_category = sum(1 for r in scored if r["category"])
    print(f"Wrote {len(scored)} wallets to {COHORT_CSV} ({in_any_category} landed in at least one category).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-file", type=str, required=True, help="Text file, one token mint address per line")
    parser.add_argument("--top-n-per-token", type=int, default=25, help="How many top traders to pull per seed token")
    parser.add_argument("--top-n-per-category", type=int, default=200, help="Cohort size per scoring category (overlap allowed)")
    parser.add_argument("--credit-budget", type=int, default=3000, help="Hard stop on credits spent this run")
    parser.add_argument("--skip-harvest", action="store_true", help="Only re-run scoring on already-harvested data")
    args = parser.parse_args()

    if not args.skip_harvest:
        client = VybeClient(credit_budget=args.credit_budget)
        seed_tokens = load_seed_tokens(args.seed_file)
        print(f"Loaded {len(seed_tokens)} seed tokens. Free tier = 4 req/min, so a full pass takes "
              f"~{len(seed_tokens) * 15 / 60:.1f} minutes minimum if starting fresh.")
        harvest(seed_tokens, client, args.top_n_per_token)

    aggregate_and_score(args.top_n_per_category)


if __name__ == "__main__":
    main()
