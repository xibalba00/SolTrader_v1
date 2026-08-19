"""
Logs one row per closed trade, and can also emit a running stats summary
(win rate, avg gain, avg loss, expectancy) computed from real logged
trades — never fabricated.
"""

import csv
import os
import time
import dataclasses
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

LOG_FIELDS = [
    "session_id",            # links this trade back to its exact rules in session_rules.csv
    "timestamp_utc",
    "mode",                # paper | live | backtest
    "strategy_name",
    "strategy_version",
    "token_symbol",
    "token_address",
    "entry_time_utc",       # actual calendar time of the buy (real historical time in backtest mode)
    "exit_time_utc",        # actual calendar time of the sell
    "entry_reason",          # short string of why this candidate passed the buy filter
    # --- candidate snapshot AT THE MOMENT OF ENTRY — same API response
    # already being fetched, just parsing more of it. Lets you eventually
    # ask "did trades with X liquidity / Y volume-acceleration / Z recent
    # momentum perform differently" instead of only seeing the outcome.
    "entry_liquidity_usd",
    "entry_mcap_usd",
    "entry_volume_5m_usd",
    "entry_volume_1h_usd",
    "entry_volume_6h_usd",
    "entry_volume_24h_usd",
    "entry_price_change_5m_pct",
    "entry_price_change_1h_pct",
    "entry_price_change_6h_pct",
    "entry_price_change_24h_pct",
    "entry_age_hours",
    "wallet_balance_before_usd",
    "position_size_pct_wallet",
    "position_size_usd",
    "buy_price_usd",         # quoted/market price before slippage
    "buy_slippage_pct",
    "effective_buy_price_usd",   # what you actually paid, after slippage
    "tp_target_pct",
    "sl_target_pct",
    "sell_reason",          # take_profit | stop_loss | circuit_breaker | max_hold_time | trailing_stop | gas_topup
    "sell_price_usd",        # quoted/market price before slippage
    "sell_slippage_pct",
    "effective_sell_price_usd",  # what you actually received, after slippage
    "gas_cost_usd",
    "hold_duration_minutes",
    # --- MFE/MAE: how far the position moved in your favor / against you
    # before it closed, regardless of where it actually exited. Answers
    # "is my TP cutting off trades that routinely run further?" and
    # "does my SL match where winners vs losers actually diverge?"
    "mfe_pct",               # maximum favorable excursion — best unrealized gain seen, ever
    "mae_pct",               # maximum adverse excursion — worst unrealized loss seen, ever
    "profit_pct",             # net of slippage AND gas
    "profit_usd",              # net of slippage AND gas
]

STRATEGY_NAME = "trend_follow_v1"
STRATEGY_VERSION = "2026.08.19"  # bump this string whenever entry/exit logic changes meaningfully


@dataclass
class TradeRecord:
    session_id: str
    timestamp_utc: str
    mode: str
    strategy_name: str
    strategy_version: str
    token_symbol: str
    token_address: str
    entry_time_utc: str
    exit_time_utc: str
    entry_reason: str
    entry_liquidity_usd: float
    entry_mcap_usd: float
    entry_volume_5m_usd: float
    entry_volume_1h_usd: float
    entry_volume_6h_usd: float
    entry_volume_24h_usd: float
    entry_price_change_5m_pct: float
    entry_price_change_1h_pct: float
    entry_price_change_6h_pct: float
    entry_price_change_24h_pct: float
    entry_age_hours: float
    wallet_balance_before_usd: float
    position_size_pct_wallet: float
    position_size_usd: float
    buy_price_usd: float
    buy_slippage_pct: float
    effective_buy_price_usd: float
    tp_target_pct: float
    sl_target_pct: float
    sell_reason: str
    sell_price_usd: float
    sell_slippage_pct: float
    effective_sell_price_usd: float
    gas_cost_usd: float
    hold_duration_minutes: float
    mfe_pct: float
    mae_pct: float
    profit_pct: float
    profit_usd: float


class TradeLogger:
    def __init__(self, log_path: str = "logs/trades.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            self._write_header()
            return

        # Guard against schema drift: if this file was created by an older
        # version of this code (fewer/different columns), keep appending
        # to it silently corrupts the file — new rows won't line up with
        # the old header, and reading it back later throws confusing
        # KeyErrors far from the actual cause. Detect and auto-migrate
        # instead of failing mysteriously.
        with open(log_path, "r", newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != LOG_FIELDS:
            backup_path = log_path.replace(".csv", f"_old_schema_{int(time.time())}.csv")
            os.rename(log_path, backup_path)
            print(f"[trade_logger] '{log_path}' had an outdated column schema — "
                  f"archived it to '{backup_path}' and starting a fresh log with the current schema.")
            self._write_header()

    def _write_header(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()

    def log_trade(self, record: TradeRecord) -> None:
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writerow(asdict(record))

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", newline="") as f:
            return list(csv.DictReader(f))

    def summary_stats(self) -> dict:
        """Real stats computed from whatever has actually been logged so far.
        Returns zeros / None on an empty log rather than inventing numbers."""
        return stats_from_rows(self.read_all())


def stats_from_rows(rows: list[dict]) -> dict:
    """Same stats calc as TradeLogger.summary_stats(), but usable on any
    subset of rows (e.g. filtered to specific calendar dates) — not just
    the whole log file. This is what makes multi-date backtest summaries
    possible without a separate stats implementation."""
    if not rows:
        return {
            "trade_count": 0,
            "win_rate_pct": None,
            "avg_gain_pct": None,
            "avg_loss_pct": None,
            "expectancy_pct": None,
            "total_profit_usd": 0.0,
            "total_gas_usd": 0.0,
        }

    profits_pct = [float(r["profit_pct"]) for r in rows]
    profits_usd = [float(r["profit_usd"]) for r in rows]
    gas_usd = [float(r.get("gas_cost_usd") or 0) for r in rows]
    wins = [p for p in profits_pct if p > 0]
    losses = [p for p in profits_pct if p <= 0]

    win_rate = (len(wins) / len(rows)) * 100
    avg_gain = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = (win_rate / 100 * avg_gain) + ((1 - win_rate / 100) * avg_loss)

    return {
        "trade_count": len(rows),
        "win_rate_pct": round(win_rate, 2),
        "avg_gain_pct": round(avg_gain, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expectancy_pct": round(expectancy, 2),
        "total_profit_usd": round(sum(profits_usd), 2),
        "total_gas_usd": round(sum(gas_usd), 2),
    }


def make_record(
    mode: str,
    token_symbol: str,
    token_address: str,
    wallet_balance_before_usd: float,
    position_size_usd: float,
    buy_price_usd: float,
    buy_slippage_pct: float,
    tp_target_pct: float,
    sl_target_pct: float,
    sell_reason: str,
    sell_price_usd: float,
    sell_slippage_pct: float,
    hold_duration_minutes: float,
    session_id: str = "unknown_session",
    entry_time_utc: Optional[str] = None,
    exit_time_utc: Optional[str] = None,
    gas_cost_usd: float = 0.0,
    strategy_name: str = STRATEGY_NAME,
    strategy_version: str = STRATEGY_VERSION,
    entry_reason: str = "",
    entry_liquidity_usd: float = 0.0,
    entry_mcap_usd: float = 0.0,
    entry_volume_5m_usd: float = 0.0,
    entry_volume_1h_usd: float = 0.0,
    entry_volume_6h_usd: float = 0.0,
    entry_volume_24h_usd: float = 0.0,
    entry_price_change_5m_pct: float = 0.0,
    entry_price_change_1h_pct: float = 0.0,
    entry_price_change_6h_pct: float = 0.0,
    entry_price_change_24h_pct: float = 0.0,
    entry_age_hours: float = 0.0,
    mfe_pct: float = 0.0,
    mae_pct: float = 0.0,
) -> TradeRecord:
    position_size_pct_wallet = (
        (position_size_usd / wallet_balance_before_usd) * 100
        if wallet_balance_before_usd else 0.0
    )

    # This is the actual fix for "slippage isn't reflected in P&L": buying
    # costs you MORE than the quoted price (slippage works against you on
    # entry), and selling gets you LESS than quoted (slippage works against
    # you on exit too). Both must move the effective price in the direction
    # that hurts you, not just get logged as a column nobody reads.
    effective_buy_price = buy_price_usd * (1 + buy_slippage_pct / 100)
    effective_sell_price = sell_price_usd * (1 - sell_slippage_pct / 100)

    profit_pct = (
        ((effective_sell_price - effective_buy_price) / effective_buy_price) * 100
        if effective_buy_price else 0.0
    )
    profit_usd = position_size_usd * (profit_pct / 100) - gas_cost_usd

    now = datetime.now(timezone.utc)
    if exit_time_utc is None:
        exit_time_utc = now.isoformat()
    if entry_time_utc is None:
        # fall back to "now minus hold duration" for paper/live mode where
        # we don't separately track entry time at the call site
        from datetime import timedelta
        entry_time_utc = (now - timedelta(minutes=hold_duration_minutes)).isoformat()

    return TradeRecord(
        session_id=session_id,
        timestamp_utc=now.isoformat(),
        mode=mode,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        token_symbol=token_symbol,
        token_address=token_address,
        entry_time_utc=entry_time_utc,
        exit_time_utc=exit_time_utc,
        entry_reason=entry_reason,
        entry_liquidity_usd=round(entry_liquidity_usd, 2),
        entry_mcap_usd=round(entry_mcap_usd, 2),
        entry_volume_5m_usd=round(entry_volume_5m_usd, 2),
        entry_volume_1h_usd=round(entry_volume_1h_usd, 2),
        entry_volume_6h_usd=round(entry_volume_6h_usd, 2),
        entry_volume_24h_usd=round(entry_volume_24h_usd, 2),
        entry_price_change_5m_pct=round(entry_price_change_5m_pct, 2),
        entry_price_change_1h_pct=round(entry_price_change_1h_pct, 2),
        entry_price_change_6h_pct=round(entry_price_change_6h_pct, 2),
        entry_price_change_24h_pct=round(entry_price_change_24h_pct, 2),
        entry_age_hours=round(entry_age_hours, 2),
        wallet_balance_before_usd=round(wallet_balance_before_usd, 2),
        position_size_pct_wallet=round(position_size_pct_wallet, 2),
        position_size_usd=round(position_size_usd, 2),
        buy_price_usd=buy_price_usd,
        buy_slippage_pct=round(buy_slippage_pct, 3),
        effective_buy_price_usd=effective_buy_price,
        tp_target_pct=tp_target_pct,
        sl_target_pct=sl_target_pct,
        sell_reason=sell_reason,
        sell_price_usd=sell_price_usd,
        sell_slippage_pct=round(sell_slippage_pct, 3),
        effective_sell_price_usd=effective_sell_price,
        gas_cost_usd=round(gas_cost_usd, 4),
        hold_duration_minutes=round(hold_duration_minutes, 2),
        mfe_pct=round(mfe_pct, 2),
        mae_pct=round(mae_pct, 2),
        profit_pct=round(profit_pct, 2),
        profit_usd=round(profit_usd, 2),
    )


def log_session_rules(
    buy_params,
    sell_params,
    risk_params,
    session_id: str,
    rules_log_path: str = "logs/session_rules.csv",
) -> None:
    """
    Records the EXACT parameters a session actually ran with — sourced
    directly from the live config.py dataclasses via dataclasses.asdict(),
    never hand-typed — so this can never drift out of sync with reality
    the way a manually-copied note could. One row per session start,
    appended to a companion file, kept SEPARATE from trades.csv/
    backtest_trades.csv so it can't break the trade-stats parsing (mixing
    rule rows into the trade log crashes stats_from_rows(), since it
    isn't a real trade row).

    `session_id` must match what's passed to TradeLogger/make_record for
    that same session — that's the join key that lets compare_sessions.py
    (or any manual analysis) line rules up against results without
    merging the files together.
    """
    buy_dict = {f"buy_{k}": v for k, v in dataclasses.asdict(buy_params).items()}
    sell_dict = {f"sell_{k}": v for k, v in dataclasses.asdict(sell_params).items()}
    risk_dict = {f"risk_{k}": v for k, v in dataclasses.asdict(risk_params).items()}
    row = {"session_id": session_id, "session_start_utc": datetime.now(timezone.utc).isoformat(), **buy_dict, **sell_dict, **risk_dict}
    fieldnames = list(row.keys())

    os.makedirs(os.path.dirname(rules_log_path), exist_ok=True)

    # Same schema-drift guard as TradeLogger: if config.py gains/loses a
    # field later, don't silently corrupt the file — archive and restart
    # with a fresh header instead.
    needs_fresh_header = True
    if os.path.exists(rules_log_path):
        with open(rules_log_path, "r", newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header == fieldnames:
            needs_fresh_header = False
        else:
            backup_path = rules_log_path.replace(".csv", f"_old_schema_{int(time.time())}.csv")
            os.rename(rules_log_path, backup_path)
            print(f"[trade_logger] '{rules_log_path}' had an outdated column schema — "
                  f"archived it to '{backup_path}'.")

    with open(rules_log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_fresh_header:
            writer.writeheader()
        writer.writerow(row)


CANDIDATE_FUNNEL_FIELDS = [
    "session_id",
    "cycle_time_utc",
    "total_candidates_seen",
    "failed_no_age_data",
    "failed_blacklist",
    "failed_stablecoin",
    "failed_liquidity",
    "failed_volume",
    "failed_age_min",
    "failed_age_max",
    "failed_mcap",
    "passed_all_buy_filters",
    "rejected_max_concurrent",
    "rejected_already_held",
    "rejected_size_invalid",
    "actually_opened",
]


def log_candidate_funnel(
    funnel: dict,
    session_id: str,
    funnel_log_path: str = "logs/candidate_funnel.csv",
) -> None:
    """
    One row per discovery cycle (every poll_interval_seconds, same cadence
    that already runs — no extra API calls, this just counts WHY each
    already-fetched candidate passed or failed instead of only recording
    the ones that became trades).

    Answers "why only 4-5 concurrent positions out of 45-50 raw
    candidates" directly: most candidates are failing at a specific,
    now-visible stage (liquidity, volume, age, blacklist, etc.) rather
    than a mystery. `funnel` is expected to already contain all
    CANDIDATE_FUNNEL_FIELDS keys except session_id/cycle_time_utc, which
    this function fills in.
    """
    row = {
        "session_id": session_id,
        "cycle_time_utc": datetime.now(timezone.utc).isoformat(),
        **{k: funnel.get(k, 0) for k in CANDIDATE_FUNNEL_FIELDS if k not in ("session_id", "cycle_time_utc")},
    }

    os.makedirs(os.path.dirname(funnel_log_path), exist_ok=True)

    needs_fresh_header = True
    if os.path.exists(funnel_log_path):
        with open(funnel_log_path, "r", newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header == CANDIDATE_FUNNEL_FIELDS:
            needs_fresh_header = False
        else:
            backup_path = funnel_log_path.replace(".csv", f"_old_schema_{int(time.time())}.csv")
            os.rename(funnel_log_path, backup_path)
            print(f"[trade_logger] '{funnel_log_path}' had an outdated column schema — "
                  f"archived it to '{backup_path}'.")

    with open(funnel_log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FUNNEL_FIELDS)
        if needs_fresh_header:
            writer.writeheader()
        writer.writerow(row)
