"""
Paper trading engine.

Tracks a simulated wallet balance and open positions in memory, pulls
REAL price data from DexScreener, and (optionally) real quote/slippage
data from Jupiter, so the numbers this produces are grounded in actual
market conditions — not synthetic randomness.

Nothing here signs or sends a transaction. Safe to run continuously.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from .config import CONFIG
from .dexscreener_client import DexScreenerClient, extract_pair_metrics, pair_age_hours
from .trade_logger import TradeLogger, make_record, STRATEGY_NAME, STRATEGY_VERSION
from .jupiter_client import estimate_price_impact_pct  # shared with backtest_engine.py — one source of truth
from .token_filters import is_stablecoin_or_blacklisted


@dataclass
class OpenPosition:
    token_symbol: str
    token_address: str
    pair_address: str
    entry_price_usd: float
    buy_slippage_pct: float
    size_usd: float
    opened_at: datetime
    high_water_price: float  # for trailing stop AND mfe tracking
    low_water_price: float   # for mae tracking
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


class PaperEngine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.balance_usd = CONFIG.risk.starting_balance_usd
        self.positions: dict[str, OpenPosition] = {}  # keyed by pair_address
        self.dex = DexScreenerClient(chain=CONFIG.dexscreener_chain)
        self.logger = TradeLogger()
        self.blacklist: dict[str, datetime] = {}  # token_address -> blacklisted until this time

    def is_blacklisted(self, token_address: str) -> bool:
        expiry = self.blacklist.get(token_address)
        if expiry is None:
            return False
        if datetime.now(timezone.utc) >= expiry:
            del self.blacklist[token_address]  # cooldown expired, clean up
            return False
        return True

    # ---------- BUY SIDE ----------

    def evaluate_candidate_verbose(self, pair: dict) -> tuple[bool, str]:
        """Same filter logic as evaluate_candidate(), but also reports
        WHICH stage a candidate failed at (or 'passed_all_buy_filters'),
        for the candidate funnel log. evaluate_candidate() below is now a
        thin wrapper over this — no behavior change to actual trading."""
        m = extract_pair_metrics(pair)
        p = CONFIG.buy

        if self.is_blacklisted(m["base_address"] or ""):
            return False, "failed_blacklist"

        # Stablecoins occasionally pass the liquidity/volume/mcap filters
        # (found in real data: a USDC trade slipped through) but they're
        # not a memecoin play at all — price barely moves, so TP/SL logic
        # doesn't apply meaningfully. Skip them explicitly.
        if is_stablecoin_or_blacklisted(
            symbol=m.get("base_token"),
            address=m.get("base_address"),
            price_usd=m.get("price_usd"),
        ):
            return False, "failed_stablecoin"

        age_h = pair_age_hours(m["pair_created_at_ms"])
        if age_h is None:
            return False, "failed_no_age_data"

        if m["liquidity_usd"] < p.min_liquidity_usd:
            return False, "failed_liquidity"
        if m["volume_24h_usd"] < p.min_volume_24h_usd:
            return False, "failed_volume"
        if age_h < (p.min_age_minutes / 60):
            return False, "failed_age_min"
        if age_h > p.max_age_hours:
            return False, "failed_age_max"
        if not (p.min_mcap_usd <= m["mcap_usd"] <= p.max_mcap_usd):
            return False, "failed_mcap"

        return True, "passed_all_buy_filters"

    def evaluate_candidate(self, pair: dict) -> bool:
        """Return True if this pair passes all buy-side filters."""
        passed, _ = self.evaluate_candidate_verbose(pair)
        return passed

    def open_position_verbose(self, pair: dict) -> str:
        """Same logic as open_position(), but reports why a position
        wasn't opened (or 'actually_opened') for the funnel log.
        open_position() below is a thin wrapper — no behavior change."""
        m = extract_pair_metrics(pair)

        if len(self.positions) >= CONFIG.risk.max_concurrent_positions:
            return "rejected_max_concurrent"
        if m["pair_address"] in self.positions:
            return "rejected_already_held"

        size_usd = self.balance_usd * (CONFIG.risk.position_size_pct / 100)
        if size_usd <= 0 or size_usd > self.balance_usd:
            return "rejected_size_invalid"

        self._do_open_position(pair, m, size_usd)
        return "actually_opened"

    def open_position(self, pair: dict) -> None:
        self.open_position_verbose(pair)

    def _do_open_position(self, pair: dict, m: dict, size_usd: float) -> None:

        # Real slippage estimate from trade size vs this pair's actual
        # liquidity — was previously a flat hardcoded 0.5% regardless of
        # size or liquidity, which isn't real slippage, just a placeholder
        # number. Same model used in backtest_engine.py now.
        slippage_pct = estimate_price_impact_pct(size_usd, m["liquidity_usd"])
        fill_price = m["price_usd"] * (1 + slippage_pct / 100)
        age_h = pair_age_hours(m["pair_created_at_ms"]) or 0.0

        entry_reason = (
            f"liq=${m['liquidity_usd']:.0f} vol24h=${m['volume_24h_usd']:.0f} "
            f"age={age_h:.1f}h mcap=${m['mcap_usd']:.0f}"
        )

        self.positions[m["pair_address"]] = OpenPosition(
            token_symbol=m["base_token"] or "UNKNOWN",
            token_address=m["base_address"] or "",
            pair_address=m["pair_address"],
            entry_price_usd=fill_price,
            buy_slippage_pct=slippage_pct,
            size_usd=size_usd,
            opened_at=datetime.now(timezone.utc),
            high_water_price=fill_price,
            low_water_price=fill_price,
            entry_reason=entry_reason,
            entry_liquidity_usd=m["liquidity_usd"],
            entry_mcap_usd=m["mcap_usd"],
            entry_volume_5m_usd=m["volume_5m_usd"],
            entry_volume_1h_usd=m["volume_1h_usd"],
            entry_volume_6h_usd=m["volume_6h_usd"],
            entry_volume_24h_usd=m["volume_24h_usd"],
            entry_price_change_5m_pct=m["price_change_5m_pct"],
            entry_price_change_1h_pct=m["price_change_1h_pct"],
            entry_price_change_6h_pct=m["price_change_6h_pct"],
            entry_price_change_24h_pct=m["price_change_24h_pct"],
            entry_age_hours=age_h,
        )
        self.balance_usd -= size_usd

    # ---------- SELL SIDE ----------

    def check_exits(self) -> None:
        """Poll current prices for every open position and close any that
        hit TP, SL, trailing stop, or max hold time."""
        sp = CONFIG.sell
        to_close = []

        for pair_address, pos in self.positions.items():
            pair = self.dex.get_pair(pair_address)
            if not pair:
                continue
            m = extract_pair_metrics(pair)
            current_price = m["price_usd"]
            if current_price <= 0:
                continue

            pos.high_water_price = max(pos.high_water_price, current_price)
            pos.low_water_price = min(pos.low_water_price, current_price)
            change_pct = ((current_price - pos.entry_price_usd) / pos.entry_price_usd) * 100
            hold_minutes = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 60

            reason = None
            if change_pct >= sp.take_profit_pct:
                reason = "take_profit"
            elif change_pct <= sp.stop_loss_pct:
                # Distinguish a normal stop-loss from a gap-through crash:
                # if the drop is well beyond the configured SL (meaning
                # price moved past the stop entirely between two polls,
                # not a clean trigger at the threshold), tag it separately
                # so it's visible in the log rather than hidden inside
                # ordinary "stop_loss" numbers. Sold either way — this is
                # a labeling/analysis distinction, not a different action.
                if change_pct <= sp.stop_loss_pct * CONFIG.risk.circuit_breaker_multiplier:
                    reason = "circuit_breaker"
                else:
                    reason = "stop_loss"
            elif sp.trailing_stop_pct is not None:
                # Trailing only engages once the position has actually
                # shown a real gain (high-water mark at or above the
                # activation threshold above entry) — otherwise a plain
                # dip from entry would trigger this before ever reaching
                # the real stop-loss, which is the bug this fixes.
                high_water_gain_pct = ((pos.high_water_price - pos.entry_price_usd) / pos.entry_price_usd) * 100
                if high_water_gain_pct >= sp.trailing_stop_activation_pct:
                    drawdown_from_high = ((current_price - pos.high_water_price) / pos.high_water_price) * 100
                    if drawdown_from_high <= -sp.trailing_stop_pct:
                        reason = "trailing_stop"
            if reason is None and hold_minutes >= sp.max_hold_hours * 60:
                reason = "max_hold_time"

            if reason:
                # Real slippage estimate at exit time too, from CURRENT
                # liquidity (not reused from entry — liquidity can shift).
                sell_slippage_pct = estimate_price_impact_pct(pos.size_usd, m["liquidity_usd"])
                to_close.append((pair_address, current_price, reason, hold_minutes, sell_slippage_pct))

        for pair_address, price, reason, hold_minutes, sell_slippage_pct in to_close:
            self._close_position(pair_address, price, reason, hold_minutes, sell_slippage_pct)

    def _close_position(
        self, pair_address: str, sell_price: float, reason: str,
        hold_minutes: float, sell_slippage_pct: float,
    ) -> None:
        pos = self.positions.pop(pair_address)
        fill_price = sell_price * (1 - sell_slippage_pct / 100)

        proceeds_usd = pos.size_usd * (fill_price / pos.entry_price_usd)
        self.balance_usd += proceeds_usd

        # The actual fix for the repeat-buyback problem: block re-entry
        # into this specific token for a cooldown period after any
        # stop-loss (including circuit_breaker) exit. Found from real
        # paper-trading data where 5 tokens alone produced 14 catastrophic
        # trades because nothing stopped the bot re-buying a token that
        # had just crashed.
        if CONFIG.risk.blacklist_after_stop_loss and reason in ("stop_loss", "circuit_breaker") and pos.token_address:
            self.blacklist[pos.token_address] = datetime.now(timezone.utc) + timedelta(hours=CONFIG.risk.blacklist_cooldown_hours)

        record = make_record(
            mode="paper",
            session_id=self.session_id,
            strategy_name=STRATEGY_NAME,
            strategy_version=STRATEGY_VERSION,
            token_symbol=pos.token_symbol,
            token_address=pos.token_address,
            entry_reason=pos.entry_reason,
            entry_liquidity_usd=pos.entry_liquidity_usd,
            entry_mcap_usd=pos.entry_mcap_usd,
            entry_volume_5m_usd=pos.entry_volume_5m_usd,
            entry_volume_1h_usd=pos.entry_volume_1h_usd,
            entry_volume_6h_usd=pos.entry_volume_6h_usd,
            entry_volume_24h_usd=pos.entry_volume_24h_usd,
            entry_price_change_5m_pct=pos.entry_price_change_5m_pct,
            entry_price_change_1h_pct=pos.entry_price_change_1h_pct,
            entry_price_change_6h_pct=pos.entry_price_change_6h_pct,
            entry_price_change_24h_pct=pos.entry_price_change_24h_pct,
            entry_age_hours=pos.entry_age_hours,
            wallet_balance_before_usd=self.balance_usd - proceeds_usd + pos.size_usd,
            position_size_usd=pos.size_usd,
            buy_price_usd=pos.entry_price_usd,
            buy_slippage_pct=pos.buy_slippage_pct,
            tp_target_pct=CONFIG.sell.take_profit_pct,
            sl_target_pct=CONFIG.sell.stop_loss_pct,
            sell_reason=reason,
            sell_price_usd=fill_price,
            sell_slippage_pct=sell_slippage_pct,
            hold_duration_minutes=hold_minutes,
            gas_cost_usd=CONFIG.risk.assumed_gas_cost_usd_per_trade,
            mfe_pct=((pos.high_water_price - pos.entry_price_usd) / pos.entry_price_usd) * 100,
            mae_pct=((pos.low_water_price - pos.entry_price_usd) / pos.entry_price_usd) * 100,
        )
        self.logger.log_trade(record)

    # ---------- LOOP ----------

    def run_once(self, candidate_pairs: list[dict]) -> None:
        for pair in candidate_pairs:
            if self.evaluate_candidate(pair):
                self.open_position(pair)
        self.check_exits()

    def run_discovery_cycle(self, candidate_pairs: list[dict]) -> dict:
        """Same effect as evaluating + opening each candidate in run_once(),
        but also returns a funnel dict of exactly why each candidate did
        or didn't become a position — no extra API calls, this only counts
        outcomes from data already being fetched this cycle. Does NOT call
        check_exits() — that stays on the fast loop in main.py, separate
        from this (slower) discovery cycle."""
        funnel = {k: 0 for k in [
            "total_candidates_seen", "failed_no_age_data", "failed_blacklist",
            "failed_stablecoin", "failed_liquidity", "failed_volume",
            "failed_age_min", "failed_age_max", "failed_mcap",
            "passed_all_buy_filters", "rejected_max_concurrent",
            "rejected_already_held", "rejected_size_invalid", "actually_opened",
        ]}

        for pair in candidate_pairs:
            funnel["total_candidates_seen"] += 1
            passed, stage = self.evaluate_candidate_verbose(pair)
            funnel[stage] += 1
            if passed:
                open_outcome = self.open_position_verbose(pair)
                funnel[open_outcome] += 1

        return funnel

    def status(self) -> dict:
        return {
            "balance_usd": round(self.balance_usd, 2),
            "open_positions": len(self.positions),
            "blacklisted_tokens": len(self.blacklist),
            **self.logger.summary_stats(),
        }
