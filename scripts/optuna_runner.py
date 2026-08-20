"""
Optuna-based parameter search runner for SolTrader_v1

Usage:
    python scripts/optuna_runner.py --trials 3000 --pages 1 --timeframe minute --aggregate 15
    python scripts/optuna_runner.py --trials 3000 --pages 1 --timeframe hour   --aggregate 1

FIXED-DATASET BEHAVIOR (important — this changed):
 - Pool discovery (trending+top pools) and OHLCV candle fetching now happen
   ONCE, at the start of main(), for the requested --timeframe/--aggregate.
 - Every trial afterward replays that SAME cached dataset in memory — zero
   API calls happen inside a trial. This makes trials directly comparable
   to each other (no candidate-universe drift mid-run) and makes 1000s of
   trials fast, since only the one-time prefetch phase touches the network.
 - Re-running this script later (e.g. a week later) re-fetches a fresh
   dataset — that's your out-of-sample comparison, not something to avoid.

PER-TIMEFRAME OUTPUT ISOLATION:
 - Every --timeframe/--aggregate combo writes to its own folder and files:
   backtest_results_<timeframe>_<aggregate>/  and
   logs/backtest_trades_<timeframe>_<aggregate>.csv
 - Running --timeframe minute --aggregate 1, then --aggregate 5, then
   --aggregate 15, then --timeframe hour --aggregate 1 never overwrites or
   merges with each other — four independent output sets.

Notes:
 - This imports the repo's bot.config.CONFIG and mutates fields for each trial.
 - It runs the BacktestEngine programmatically (no subprocess/config file edits).
 - Requires optuna: pip install optuna
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
import math

# Ensure the project root (parent of this script's /scripts folder) is on
# sys.path, so `bot` resolves regardless of the current working directory
# or how the script is invoked (e.g. `python3 scripts/optuna_runner.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import optuna
except Exception as e:
    raise SystemExit("optuna is required. Install with: pip install optuna")

from bot.config import CONFIG
from bot.geckoterminal_client import GeckoTerminalClient
from bot.backtest_engine import BacktestEngine, prefetch_candle_cache
from bot.trade_logger import stats_from_rows

# These are placeholders — main() overwrites them with timeframe-suffixed
# paths (e.g. backtest_results_minute_15/, master_results_minute_15.csv)
# before ensure_results_dir()/objective_factory() run, so every timeframe
# you run writes to its own folder/files and nothing gets merged together.
RESULTS_DIR = "backtest_results"
MASTER_CSV = os.path.join(RESULTS_DIR, "master_results.csv")
PER_RUN_PREFIX = os.path.join(RESULTS_DIR, "trades_")
TRADES_LOG_PATH = "logs/backtest_trades.csv"


# Parameter bounds (hardcoded min/max per your request — uniform sampling within these)
PARAM_BOUNDS = {
    # min_liquidity_usd: 10k - 250k
    "min_liquidity_usd": (10_000, 250_000),
    # min_volume_24h_usd: 30k - 300k
    "min_volume_24h_usd": (30_000, 300_000),
    # min_age_minutes: 2h - 48h -> minutes
    "min_age_minutes": (2 * 60, 48 * 60),
    # max_age_hours: 48h - 720h
    "max_age_hours": (48, 720),
    # min_mcap_usd: 40k - 2_000_000
    "min_mcap_usd": (40_000, 2_000_000),
    # max_mcap_usd: 1_000_000 - 20_000_000
    "max_mcap_usd": (1_000_000, 20_000_000),
    # take_profit_pct: 5 - 50
    "take_profit_pct": (5.0, 50.0),
    # stop_loss_pct magnitude: 5 - 25 (we'll store negative in config)
    "stop_loss_mag": (5.0, 25.0),
    # trailing_stop_pct: 5 - 20 (can be None, but we'll sample numeric)
    "trailing_stop_pct": (5.0, 20.0),
    # trailing_stop_activation_pct: 5 - 15
    "trailing_stop_activation_pct": (5.0, 15.0),
    # position_size_pct: 1.5 - 7.5
    "position_size_pct": (1.5, 7.5),
    # blacklist_cooldown_hours: 6 - 48
    "blacklist_cooldown_hours": (6, 48),
    # circuit_breaker_multiplier: 1.5 - 5.0
    "circuit_breaker_multiplier": (1.5, 5.0),
}

# Defaults for runner behavior (user requested)
DEFAULT_INTER_TRIAL_DELAY = 10
DEFAULT_MAX_RETRIES = 8


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def append_master_row(row: dict):
    needs_header = not os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def write_per_run_trades(session_rows: list[dict], session_id: str):
    path = f"{PER_RUN_PREFIX}{session_id}.csv"
    if not session_rows:
        # write empty header to keep structure
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["no_trades_for_session"])
        return path

    fieldnames = list(session_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in session_rows:
            writer.writerow(r)
    return path


def compute_metrics(session_rows: list[dict], starting_balance: float):
    # Returns dict with: trade_count, total_profit_usd, sharpe, max_drawdown_pct
    if not session_rows:
        return {
            "trade_count": 0,
            "total_profit_usd": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
        }

    # profit_usd series in chronological order (sort by exit_time_utc or timestamp_utc)
    def ts_key(r):
        return r.get("exit_time_utc") or r.get("timestamp_utc") or r.get("entry_time_utc")

    session_rows_sorted = sorted(session_rows, key=lambda r: ts_key(r))
    profits_usd = [float(r.get("profit_usd", 0.0)) for r in session_rows_sorted]
    profits_pct = [float(r.get("profit_pct", 0.0)) for r in session_rows_sorted]

    total_profit = sum(profits_usd)

    # Sharpe (trade-level): mean/std of profit_pct (as fraction). If std==0 -> 0
    sharpe = 0.0
    if len(profits_pct) >= 2:
        import statistics

        returns = [p / 100.0 for p in profits_pct]
        mean_r = statistics.mean(returns)
        stdev_r = statistics.pstdev(returns) if len(returns) >= 2 else 0.0
        if stdev_r > 0:
            sharpe = mean_r / stdev_r
        else:
            sharpe = 0.0

    # Equity curve and max drawdown
    equity = []
    bal = starting_balance
    for p in profits_usd:
        bal += p
        equity.append(bal)
    peak = -math.inf
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    max_drawdown_pct = max_dd * 100.0

    return {
        "trade_count": len(session_rows_sorted),
        "total_profit_usd": round(total_profit, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
    }


def objective_factory(
    args,
    unique_pools: list[dict],
    candle_cache: dict,
    inter_trial_delay: int = DEFAULT_INTER_TRIAL_DELAY,
):
    """unique_pools and candle_cache are fetched ONCE in main() before Optuna
    starts (see prefetch_candle_cache in bot/backtest_engine.py) — every
    trial below replays that same fixed dataset. No API calls happen inside
    objective() anymore, so trials run at in-memory speed, and every trial
    is judged against an identical, non-drifting candidate universe."""
    # Return an objective function that closes over CLI args
    def objective(trial: optuna.trial.Trial):
        # Sample parameters (use Optuna suggest_* API)
        p = {}
        p["min_liquidity_usd"] = trial.suggest_float("min_liquidity_usd", *PARAM_BOUNDS["min_liquidity_usd"])
        p["min_volume_24h_usd"] = trial.suggest_float("min_volume_24h_usd", *PARAM_BOUNDS["min_volume_24h_usd"])

        # min_age_minutes sampled in minutes (integer)
        p["min_age_minutes"] = trial.suggest_int(
            "min_age_minutes",
            int(PARAM_BOUNDS["min_age_minutes"][0]),
            int(PARAM_BOUNDS["min_age_minutes"][1]),
        )
        # max_age_hours must be >= min_age (converted to hours)
        min_age_hours = p["min_age_minutes"] / 60.0
        max_age_low = max(min_age_hours, PARAM_BOUNDS["max_age_hours"][0])
        max_age_high = PARAM_BOUNDS["max_age_hours"][1]
        p["max_age_hours"] = trial.suggest_float("max_age_hours", float(max_age_low), float(max_age_high))

        # market caps: ensure max_mcap >= min_mcap
        p["min_mcap_usd"] = trial.suggest_float("min_mcap_usd", *PARAM_BOUNDS["min_mcap_usd"])
        max_mcap_low = max(p["min_mcap_usd"], PARAM_BOUNDS["max_mcap_usd"][0])
        p["max_mcap_usd"] = trial.suggest_float(
            "max_mcap_usd", float(max_mcap_low), float(PARAM_BOUNDS["max_mcap_usd"][1])
        )

        p["take_profit_pct"] = trial.suggest_float("take_profit_pct", *PARAM_BOUNDS["take_profit_pct"])
        # sample stop-loss magnitude (positive) then store negative in CONFIG
        stop_loss_mag = trial.suggest_float("stop_loss_mag", *PARAM_BOUNDS["stop_loss_mag"])
        p["stop_loss_pct"] = -abs(stop_loss_mag)
        p["trailing_stop_pct"] = trial.suggest_float("trailing_stop_pct", *PARAM_BOUNDS["trailing_stop_pct"])
        p["trailing_stop_activation_pct"] = trial.suggest_float(
            "trailing_stop_activation_pct", *PARAM_BOUNDS["trailing_stop_activation_pct"]
        )

        p["position_size_pct"] = trial.suggest_float("position_size_pct", *PARAM_BOUNDS["position_size_pct"])
        p["blacklist_cooldown_hours"] = trial.suggest_int(
            "blacklist_cooldown_hours",
            int(PARAM_BOUNDS["blacklist_cooldown_hours"][0]),
            int(PARAM_BOUNDS["blacklist_cooldown_hours"][1]),
        )
        p["circuit_breaker_multiplier"] = trial.suggest_float(
            "circuit_breaker_multiplier", *PARAM_BOUNDS["circuit_breaker_multiplier"]
        )

        # Apply sampled params to CONFIG (mutate in place)
        # Buy params
        CONFIG.buy.min_liquidity_usd = float(p["min_liquidity_usd"])
        CONFIG.buy.min_volume_24h_usd = float(p["min_volume_24h_usd"])
        CONFIG.buy.min_age_minutes = int(round(p["min_age_minutes"]))
        CONFIG.buy.max_age_hours = float(p["max_age_hours"])
        CONFIG.buy.min_mcap_usd = float(p["min_mcap_usd"])
        CONFIG.buy.max_mcap_usd = float(p["max_mcap_usd"])

        # Sell params
        CONFIG.sell.take_profit_pct = float(p["take_profit_pct"])
        CONFIG.sell.stop_loss_pct = float(p["stop_loss_pct"])  # negative
        CONFIG.sell.trailing_stop_pct = float(p["trailing_stop_pct"])
        CONFIG.sell.trailing_stop_activation_pct = float(p["trailing_stop_activation_pct"])

        # Risk params
        CONFIG.risk.position_size_pct = float(p["position_size_pct"])
        CONFIG.risk.blacklist_cooldown_hours = float(p["blacklist_cooldown_hours"])
        CONFIG.risk.circuit_breaker_multiplier = float(p["circuit_breaker_multiplier"])

        # unique_pools + candle_cache come from main()'s ONE-TIME prefetch —
        # this is the fixed dataset every trial in this run is scored
        # against. No API call happens in this function at all.

        tf_suffix = f"{args.timeframe}_{args.aggregate}"
        session_id = datetime.now(timezone.utc).strftime(f"optuna_trial_{tf_suffix}_%Y%m%d_%H%M%S_{trial.number}")
        engine = BacktestEngine(session_id=session_id, log_path=TRADES_LOG_PATH)

        rows_before = len(engine.logger.read_all())
        # Run against the FIXED, pre-fetched dataset — candle_cache means
        # zero API calls happen inside this trial.
        stats = engine.run(unique_pools, timeframe=args.timeframe, aggregate=args.aggregate, candle_cache=candle_cache)
        all_rows = engine.logger.read_all()
        session_rows = all_rows[rows_before:]

        metrics = compute_metrics(session_rows, starting_balance=CONFIG.risk.starting_balance_usd)

        # write per-run trades
        trades_path = write_per_run_trades(session_rows, session_id)

        # assemble master row (parameters + metrics)
        master_row = {
            "run_id": session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            # parameters (rounded)
            "min_liquidity_usd": round(CONFIG.buy.min_liquidity_usd, 2),
            "min_volume_24h_usd": round(CONFIG.buy.min_volume_24h_usd, 2),
            "min_age_minutes": CONFIG.buy.min_age_minutes,
            "max_age_hours": round(CONFIG.buy.max_age_hours, 2),
            "min_mcap_usd": round(CONFIG.buy.min_mcap_usd, 2),
            "max_mcap_usd": round(CONFIG.buy.max_mcap_usd, 2),
            "take_profit_pct": round(CONFIG.sell.take_profit_pct, 3),
            "stop_loss_pct": round(CONFIG.sell.stop_loss_pct, 3),
            "trailing_stop_pct": round(CONFIG.sell.trailing_stop_pct, 3),
            "trailing_stop_activation_pct": round(CONFIG.sell.trailing_stop_activation_pct, 3),
            "position_size_pct": round(CONFIG.risk.position_size_pct, 3),
            "blacklist_cooldown_hours": round(CONFIG.risk.blacklist_cooldown_hours, 3),
            "circuit_breaker_multiplier": round(CONFIG.risk.circuit_breaker_multiplier, 3),
            # metrics
            "trade_count": metrics["trade_count"],
            "total_profit_usd": metrics["total_profit_usd"],
            "sharpe": metrics["sharpe"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "trades_csv": trades_path,
        }

        append_master_row(master_row)

        # Compose a simple scalar for optuna to maximize
        # score = sharpe + (total_profit / starting_balance) - (max_drawdown_pct / 100)
        score = master_row["sharpe"] + (master_row["total_profit_usd"] / CONFIG.risk.starting_balance_usd) - (master_row["max_drawdown_pct"] / 100.0)

        # Report intermediate values to Optuna
        trial.set_user_attr("master_row", master_row)

        # No API calls happen in this function anymore (fixed dataset), so
        # there's nothing to rate-limit here. inter_trial_delay is kept as
        # an optional throttle only — 0 by default now, see main().
        if inter_trial_delay:
            time.sleep(inter_trial_delay)

        return score

    return objective


def postprocess_master_and_write_gps(top_k: int = 10):
    # Load master CSV
    if not os.path.exists(MASTER_CSV):
        print("No master_results.csv found; skipping GPS postprocessing")
        return

    rows = []
    with open(MASTER_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert numeric fields
            for k in ["min_liquidity_usd","min_volume_24h_usd","min_age_minutes","max_age_hours",
                      "min_mcap_usd","max_mcap_usd","take_profit_pct","stop_loss_pct",
                      "trailing_stop_pct","trailing_stop_activation_pct","position_size_pct",
                      "blacklist_cooldown_hours","circuit_breaker_multiplier","trade_count",
                      "total_profit_usd","sharpe","max_drawdown_pct"]:
                if r.get(k) is None or r.get(k) == "":
                    r[k] = None
                    continue
                try:
                    if k in ("min_age_minutes", "trade_count"):
                        r[k] = int(float(r[k]))
                    else:
                        r[k] = float(r[k])
                except Exception:
                    r[k] = None
            rows.append(r)

    if not rows:
        return

    # For GPS we need min/max of sharpe, total_profit_usd, max_drawdown_pct
    sharpe_vals = [r["sharpe"] for r in rows if r.get("sharpe") is not None]
    profit_vals = [r["total_profit_usd"] for r in rows if r.get("total_profit_usd") is not None]
    dd_vals = [r["max_drawdown_pct"] for r in rows if r.get("max_drawdown_pct") is not None]

    min_sh, max_sh = (min(sharpe_vals), max(sharpe_vals)) if sharpe_vals else (0, 0)
    min_pf, max_pf = (min(profit_vals), max(profit_vals)) if profit_vals else (0, 0)
    min_dd, max_dd = (min(dd_vals), max(dd_vals)) if dd_vals else (0, 0)

    # Avoid zero-division
    def norm(x, lo, hi):
        if lo == hi:
            return 0.5
        return (x - lo) / (hi - lo)

    for r in rows:
        ns = norm(r.get("sharpe") or 0.0, min_sh, max_sh)
        npf = norm(r.get("total_profit_usd") or 0.0, min_pf, max_pf)
        ndd = norm(r.get("max_drawdown_pct") or 0.0, min_dd, max_dd)
        # drawdown smaller is better -> use (1 - normalized_dd)
        gps = (ns + npf + (1 - ndd)) / 3.0
        r["GPS"] = round(gps, 6)

    # Write a new master file with GPS appended
    master_with_gps = os.path.join(RESULTS_DIR, "master_results_with_gps.csv")
    fieldnames = list(rows[0].keys())
    with open(master_with_gps, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write top-k
    sorted_rows = sorted(rows, key=lambda r: r["GPS"], reverse=True)
    topk_path = os.path.join(RESULTS_DIR, "top_k.csv")
    if sorted_rows:
        with open(topk_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sorted_rows[0].keys()))
            writer.writeheader()
            for r in sorted_rows[:top_k]:
                writer.writerow(r)
    print(f"Wrote master_results_with_gps.csv and top_k.csv (top {top_k}) in {RESULTS_DIR}")


def main():
    global RESULTS_DIR, MASTER_CSV, PER_RUN_PREFIX, TRADES_LOG_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--timeframe", type=str, default="hour", choices=["day", "hour", "minute"])
    parser.add_argument("--aggregate", type=int, default=1, help="day=1, hour=1/4/12, minute=1/5/15")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--inter_trial_delay", type=int, default=0,
                         help="Optional throttle between trials. Default 0 — trials run against a "
                              "pre-fetched fixed dataset and make no API calls, so there's nothing "
                              "to rate-limit here anymore.")
    parser.add_argument("--candle_limit", type=int, default=1000, help="Max candles per pool (GeckoTerminal caps at 1000/call)")
    args = parser.parse_args()

    # Every timeframe/aggregate combo gets its OWN results folder + master
    # CSV + trade log — nothing from a 1min run ever lands in the same file
    # as a 1h run. This is what "not merged into a frankenstein" means here.
    tf_suffix = f"{args.timeframe}_{args.aggregate}"
    RESULTS_DIR = f"backtest_results_{tf_suffix}"
    MASTER_CSV = os.path.join(RESULTS_DIR, "master_results.csv")
    PER_RUN_PREFIX = os.path.join(RESULTS_DIR, "trades_")
    TRADES_LOG_PATH = f"logs/backtest_trades_{tf_suffix}.csv"

    ensure_results_dir()

    # ---- ONE-TIME fixed-dataset prefetch (this is the only phase that hits the API) ----
    print(f"[{tf_suffix}] Discovering candidate pools (trending + top volume, pages={args.pages})...")
    gecko = GeckoTerminalClient()
    pools = gecko.get_trending_pools(pages=args.pages) + gecko.get_top_pools(pages=args.pages)
    seen = set()
    unique_pools = []
    for ppool in pools:
        addr = ppool.get("attributes", {}).get("address")
        if addr and addr not in seen:
            seen.add(addr)
            unique_pools.append(ppool)
    print(f"[{tf_suffix}] {len(unique_pools)} unique pools. Prefetching {args.timeframe}/{args.aggregate} OHLCV for each (one API call per pool)...")

    candle_cache = prefetch_candle_cache(gecko, unique_pools, args.timeframe, args.aggregate, candle_limit=args.candle_limit)
    print(f"[{tf_suffix}] Prefetch done: {len(candle_cache)}/{len(unique_pools)} pools cached. "
          f"Starting {args.trials} trials against this FIXED dataset — no further API calls.")
    # ---- End of API-touching phase. Everything below is pure in-memory compute. ----

    study = optuna.create_study(direction="maximize", study_name=f"soltrader_param_search_{tf_suffix}")
    objective = objective_factory(args, unique_pools, candle_cache, inter_trial_delay=args.inter_trial_delay)

    try:
        study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        print("Optimization interrupted by user — will postprocess available results.")

    print(f"[{tf_suffix}] Finished {len(study.trials)} trials. Best trial:")
    if study.best_trial:
        print(study.best_trial.params)

    postprocess_master_and_write_gps(top_k=args.top_k)
    print(f"[{tf_suffix}] Results: {RESULTS_DIR}/  |  Trade log: {TRADES_LOG_PATH}")


if __name__ == "__main__":
    main()
