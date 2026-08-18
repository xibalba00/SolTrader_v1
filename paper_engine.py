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
from datetime import datetime, timezone

from .config import CONFIG
from .dexscreener_client import DexScreenerClient, extract_pair_metrics, pair_age_hours
from .trade_logger import TradeLogger, make_record
from .jupiter_client import estimate_price_impact_pct  # shared with backtest_engine.py — one source of truth


@dataclass
class OpenPosition:
    token_symbol: str
    token_address: str
    pair_address: str
    entry_price_usd: float
    buy_slippage_pct: float
    size_usd: float
    opened_at: datetime
    high_water_price: float  # for trailing stop tracking


class PaperEngine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.balance_usd = CONFIG.risk.starting_balance_usd
        self.positions: dict[str, OpenPosition] = {}  # keyed by pair_address
        self.dex = DexScreenerClient(chain=CONFIG.dexscreener_chain)
        self.logger = TradeLogger()

    # ---------- BUY SIDE ----------

    def evaluate_candidate(self, pair: dict) -> bool:
        """Return True if this pair passes all buy-side filters."""
        m = extract_pair_metrics(pair)
        p = CONFIG.buy

        age_h = pair_age_hours(m["pair_created_at_ms"])
        if age_h is None:
            return False

        checks = [
            m["liquidity_usd"] >= p.min_liquidity_usd,
            m["volume_24h_usd"] >= p.min_volume_24h_usd,
            age_h >= (p.min_age_minutes / 60),
            age_h <= p.max_age_hours,
            p.min_mcap_usd <= m["mcap_usd"] <= p.max_mcap_usd,
        ]
        return all(checks)

    def open_position(self, pair: dict) -> None:
        m = extract_pair_metrics(pair)

        if len(self.positions) >= CONFIG.risk.max_concurrent_positions:
            return
        if m["pair_address"] in self.positions:
            return  # already holding this one

        size_usd = self.balance_usd * (CONFIG.risk.position_size_pct / 100)
        if size_usd <= 0 or size_usd > self.balance_usd:
            return

        # Real slippage estimate from trade size vs this pair's actual
        # liquidity — was previously a flat hardcoded 0.5% regardless of
        # size or liquidity, which isn't real slippage, just a placeholder
        # number. Same model used in backtest_engine.py now.
        slippage_pct = estimate_price_impact_pct(size_usd, m["liquidity_usd"])
        fill_price = m["price_usd"] * (1 + slippage_pct / 100)

        self.positions[m["pair_address"]] = OpenPosition(
            token_symbol=m["base_token"] or "UNKNOWN",
            token_address=m["base_address"] or "",
            pair_address=m["pair_address"],
            entry_price_usd=fill_price,
            buy_slippage_pct=slippage_pct,
            size_usd=size_usd,
            opened_at=datetime.now(timezone.utc),
            high_water_price=fill_price,
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
            change_pct = ((current_price - pos.entry_price_usd) / pos.entry_price_usd) * 100
            hold_minutes = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 60

            reason = None
            if change_pct >= sp.take_profit_pct:
                reason = "take_profit"
            elif change_pct <= sp.stop_loss_pct:
                reason = "stop_loss"
            elif sp.trailing_stop_pct is not None:
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

        record = make_record(
            mode="paper",
            session_id=self.session_id,
            token_symbol=pos.token_symbol,
            token_address=pos.token_address,
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
            gas_cost_usd=CONFIG.risk.assumed_gas_cost_usd_per_trade,  # was missing entirely — always logged as $0
        )
        self.logger.log_trade(record)

    # ---------- LOOP ----------

    def run_once(self, candidate_pairs: list[dict]) -> None:
        for pair in candidate_pairs:
            if self.evaluate_candidate(pair):
                self.open_position(pair)
        self.check_exits()

    def status(self) -> dict:
        return {
            "balance_usd": round(self.balance_usd, 2),
            "open_positions": len(self.positions),
            **self.logger.summary_stats(),
        }
