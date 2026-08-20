"""
Backtest engine: same buy/sell rules as PaperEngine, but instead of
polling live prices, it walks forward through REAL historical OHLCV
candles pulled from GeckoTerminal.

Honest limitations (read before trusting the output):
  1. Liquidity and market-cap filters use TODAY's snapshot value applied
     across the whole historical window — GeckoTerminal's free tier
     doesn't expose a historical liquidity/mcap time series. This means
     a token that had thin liquidity a week ago but is well-liquidated
     today will pass the filter throughout, which the live paper/gone-
     live bot would NOT have allowed at the time. Volume, by contrast,
     IS computed from real historical candles (rolling 24h sum), so
     that filter is accurate.
  2. TP/SL are checked against each candle's high/low (a real intra-
     candle touch), which is more realistic than only checking closes,
     but still an approximation of exact execution timing/price.
  3. This only backtests pools that exist and are trending/high-volume
     TODAY — it can't discover pools that existed and died before now,
     which biases the sample toward "survivors." Real, worth knowing.
  4. Slippage is estimated from a simple price-impact approximation
     (trade size vs. today's pool liquidity snapshot), NOT a real
     historical order book — treat it as a reasonable estimate, not a
     precise figure. Gas is a flat assumed cost from config, not a real
     historical fee. Both are still far better than pretending they're
     zero, which the previous version of this engine did.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from .config import CONFIG
from .geckoterminal_client import GeckoTerminalClient, extract_pool_metrics
from .trade_logger import TradeLogger, make_record
from .jupiter_client import estimate_price_impact_pct  # shared with paper_engine.py — one source of truth


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_candles(raw: list[list[float]]) -> list[Candle]:
    candles = [Candle(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in raw]
    return sorted(candles, key=lambda c: c.timestamp)  # GeckoTerminal returns newest-first


def rolling_24h_volume(candles: list[Candle], index: int, candle_hours: int = 1) -> float:
    window = max(1, int(24 / candle_hours))
    start = max(0, index - window + 1)
    return sum(c.volume for c in candles[start:index + 1])


class BacktestEngine:
    def __init__(self, session_id: str, starting_balance_usd: float | None = None, log_path: str = "logs/backtest_trades.csv"):
        self.session_id = session_id
        self.balance_usd = starting_balance_usd or CONFIG.risk.starting_balance_usd
        self.gecko = GeckoTerminalClient()
        self.logger = TradeLogger(log_path=log_path)

    def run_pool(
        self, pool: dict, timeframe: str = "hour", aggregate: int = 1, candle_limit: int = 1000,
        cached_raw_ohlcv: list | None = None,
    ) -> int:
        """Backtest a single pool's history. Returns number of trades logged.

        If cached_raw_ohlcv is provided (a pre-fetched OHLCV list, same shape
        GeckoTerminalClient.get_ohlcv() returns), NO API call is made — the
        cached candles are used directly. This is what makes a genuine
        fixed-dataset run possible: pull once via prefetch_candle_cache(),
        then every trial replays the same candles instead of re-hitting the
        API. Passing nothing preserves the old live-fetch-every-call
        behavior, so backtest.py / main.py are unaffected."""
        m = extract_pool_metrics(pool)
        pool_address = m["pool_address"]
        if not pool_address or not m["pool_created_at"]:
            return 0

        created_at = datetime.fromisoformat(m["pool_created_at"].replace("Z", "+00:00"))
        if cached_raw_ohlcv is not None:
            raw = cached_raw_ohlcv
        else:
            raw = self.gecko.get_ohlcv(pool_address, timeframe=timeframe, aggregate=aggregate, limit=candle_limit)
        candles = parse_candles(raw)
        if len(candles) < 5:
            return 0

        bp, sp = CONFIG.buy, CONFIG.sell
        trades_logged = 0
        i = 0
        while i < len(candles):
            candle = candles[i]
            candle_time = datetime.fromtimestamp(candle.timestamp, tz=timezone.utc)
            age_hours = (candle_time - created_at).total_seconds() / 3600
            vol_24h = rolling_24h_volume(candles, i, candle_hours=aggregate if timeframe == "hour" else 24)

            # Apply buy filters using snapshot metrics (liquidity/mcap) and historical volume/age
            passes_buy = (
                (m.get("reserve_in_usd", 0.0) >= bp.min_liquidity_usd) and       # snapshot approximation, see caveat
                (bp.min_mcap_usd <= m.get("fdv_usd", 0.0) <= bp.max_mcap_usd) and  # snapshot approximation, see caveat
                (vol_24h >= bp.min_volume_24h_usd) and                              # real historical value
                ((age_hours * 60) >= bp.min_age_minutes) and
                (age_hours <= bp.max_age_hours)
            )

            if not passes_buy:
                i += 1
                continue

            entry_price = candle.close
            entry_index = i
            entry_time = candle_time
            high_water = entry_price
            low_water = entry_price

            j = i + 1
            closed = False
            while j < len(candles):
                c = candles[j]
                high_water = max(high_water, c.high)
                low_water = min(low_water, c.low)
                hold_hours = (c.timestamp - candle.timestamp) / 3600

                gain_at_high = ((c.high - entry_price) / entry_price) * 100 if entry_price else 0.0
                loss_at_low = ((c.low - entry_price) / entry_price) * 100 if entry_price else 0.0
                trail_dd = ((c.low - high_water) / high_water) * 100 if (sp.trailing_stop_pct and high_water) else 0

                reason, exit_price = None, None
                if gain_at_high >= sp.take_profit_pct:
                    reason = "take_profit"
                    exit_price = entry_price * (1 + sp.take_profit_pct / 100)
                elif loss_at_low <= sp.stop_loss_pct:
                    reason = "stop_loss"
                    exit_price = entry_price * (1 + sp.stop_loss_pct / 100)
                elif sp.trailing_stop_pct is not None and trail_dd <= -sp.trailing_stop_pct:
                    reason = "trailing_stop"
                    exit_price = high_water * (1 - sp.trailing_stop_pct / 100)
                elif hold_hours >= sp.max_hold_hours:
                    reason = "max_hold_time"
                    exit_price = c.close

                if reason:
                    exit_time = datetime.fromtimestamp(c.timestamp, tz=timezone.utc)
                    # compute mfe/mae from high_water/low_water relative to entry
                    mfe_pct = ((high_water - entry_price) / entry_price) * 100 if entry_price else 0.0
                    mae_pct = ((low_water - entry_price) / entry_price) * 100 if entry_price else 0.0
                    self._log_trade(
                        m,
                        entry_price,
                        exit_price,
                        reason,
                        hold_hours,
                        entry_time,
                        exit_time,
                        entry_liquidity_usd=m.get("reserve_in_usd", 0.0),
                        entry_mcap_usd=m.get("fdv_usd", 0.0),
                        entry_volume_24h_usd=vol_24h,
                        entry_age_hours=age_hours,
                        mfe_pct=mfe_pct,
                        mae_pct=mae_pct,
                    )
                    trades_logged += 1
                    closed = True
                    i = j + 1
                    break
                j += 1

            if not closed:
                i = len(candles)  # ran off the end still holding — stop scanning this pool

        return trades_logged

    def _log_trade(
        self,
        pool_metrics: dict,
        entry_price: float,
        exit_price: float,
        reason: str,
        hold_hours: float,
        entry_time: datetime,
        exit_time: datetime,
        entry_liquidity_usd: float = 0.0,
        entry_mcap_usd: float = 0.0,
        entry_volume_24h_usd: float = 0.0,
        entry_age_hours: float = 0.0,
        mfe_pct: float = 0.0,
        mae_pct: float = 0.0,
    ) -> None:
        size_usd = self.balance_usd * (CONFIG.risk.position_size_pct / 100)
        slippage_pct = estimate_price_impact_pct(size_usd, pool_metrics.get("reserve_in_usd", 0.0))

        record = make_record(
            mode="backtest",
            session_id=self.session_id,
            token_symbol=pool_metrics.get("name") or "UNKNOWN",
            token_address=pool_metrics.get("pool_address"),
            wallet_balance_before_usd=self.balance_usd,
            position_size_usd=size_usd,
            buy_price_usd=entry_price,
            buy_slippage_pct=slippage_pct,       # now a real estimate, not hardcoded 0
            tp_target_pct=CONFIG.sell.take_profit_pct,
            sl_target_pct=CONFIG.sell.stop_loss_pct,
            sell_reason=reason,
            sell_price_usd=exit_price,
            sell_slippage_pct=slippage_pct,      # now a real estimate, not hardcoded 0
            hold_duration_minutes=hold_hours * 60,
            entry_time_utc=entry_time.isoformat(),
            exit_time_utc=exit_time.isoformat(),
            gas_cost_usd=CONFIG.risk.assumed_gas_cost_usd_per_trade,
            # extra fields so backtest CSVs include the same entry snapshot info as paper_engine
            entry_liquidity_usd=entry_liquidity_usd,
            entry_mcap_usd=entry_mcap_usd,
            entry_volume_24h_usd=entry_volume_24h_usd,
            entry_age_hours=entry_age_hours,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
        )
        # balance update must reflect the SAME slippage+gas-adjusted profit
        # that got logged, not the pre-slippage price difference
        self.balance_usd += record.profit_usd
        self.logger.log_trade(record)

    def run(self, pools: list[dict], timeframe: str = "hour", aggregate: int = 1, candle_cache: dict | None = None) -> dict:
        """If candle_cache is provided (pool_address -> raw OHLCV list, from
        prefetch_candle_cache()), every pool in this run replays cached
        candles — zero API calls made here. Leave candle_cache=None for the
        old live-fetch behavior (unchanged for backtest.py / main.py)."""
        for pool in pools:
            try:
                addr = extract_pool_metrics(pool)["pool_address"]
                cached = (candle_cache or {}).get(addr)
                self.run_pool(pool, timeframe=timeframe, aggregate=aggregate, cached_raw_ohlcv=cached)
            except Exception as e:
                print(f"  skipped a pool due to error: {e}")
        return self.logger.summary_stats()


def prefetch_candle_cache(
    gecko: GeckoTerminalClient, pools: list[dict], timeframe: str, aggregate: int, candle_limit: int = 1000,
) -> dict:
    """Pull OHLCV ONCE per pool for the given timeframe/aggregate and return
    a {pool_address: raw_ohlcv_list} cache. Call this once before running
    N trials/iterations against a FIXED dataset — every trial afterward
    replays this cache via BacktestEngine.run(..., candle_cache=...) with
    no further API calls. GeckoTerminalClient already rate-limits/retries
    internally (see geckoterminal_client.py), so no extra throttling is
    needed here."""
    cache: dict = {}
    for idx, pool in enumerate(pools, 1):
        m = extract_pool_metrics(pool)
        addr = m["pool_address"]
        if not addr:
            continue
        try:
            raw = gecko.get_ohlcv(addr, timeframe=timeframe, aggregate=aggregate, limit=candle_limit)
            cache[addr] = raw
            print(f"  [prefetch {idx}/{len(pools)}] {addr}: {len(raw)} candles cached")
        except Exception as e:
            print(f"  [prefetch {idx}/{len(pools)}] {addr}: FAILED ({e}) — skipped, this pool yields 0 trades")
    return cache
