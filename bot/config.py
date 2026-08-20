"""
Central configuration for the trading bot.

Nothing here is a validated 'winning' parameter set — these are starting
defaults for paper trading. The whole point of the backtest/paper phase
is to replace guesses with numbers computed from real data.
"""

from dataclasses import dataclass, field


@dataclass
class BuyParams:
    # Your hand-tuned values (restored after I accidentally clobbered
    # this file with my stale local copy during the max_concurrent_positions
    # edit — my mistake, not yours).
    min_liquidity_usd: float = 20_000
    min_volume_24h_usd: float = 70_000
    min_age_minutes: int = 360             # 4 hours
    max_age_hours: int = 72
    min_mcap_usd: float = 50_000           # confirmed per your latest request
    max_mcap_usd: float = 5_000_000
    max_price_impact_pct: float = 3.0      # abort buy if Jupiter quote impact exceeds this


@dataclass
class SellParams:
    # Your hand-tuned values (restored — see note above). TP/SL updated
    # per your latest explicit request.
    take_profit_pct: float = 25.0          # per your request
    stop_loss_pct: float = -12.0           # per your request
    trailing_stop_pct: float | None = 10.0
    trailing_stop_activation_pct: float = 8.0  # trailing only ENGAGES once unrealized gain
                                                # crosses this. Fixes a real bug found in 381
                                                # real trades: trailing (10%) was tighter than
                                                # SL (-12%), so any trade that dipped straight
                                                # from entry without ever going positive hit
                                                # "trailing_stop" before it could ever reach the
                                                # real stop-loss — acting as an unintended,
                                                # tighter stop rather than protecting real gains.
                                                # Now trailing stays off until the position is
                                                # actually up 8%+, then protects from there.
    max_hold_hours: float = 6.0
    max_price_impact_pct: float = 2.0      # abort sell if quote impact exceeds this


@dataclass
class RiskParams:
    starting_balance_usd: float = 100.0    # paper balance only, edit freely
    position_size_pct: float = 2.5         # % of current bankroll per trade
    max_concurrent_positions: int = 10     # updated from 5 — see note above
    gas_reserve_sol: float = 0.05          # never let live SOL drop below this
    gas_topup_sell_pct: float = 25.0       # sell this % of a holding to top up gas
    assumed_gas_cost_usd_per_trade: float = 0.02  # realistic round-trip (buy+sell) under normal (non-congested) conditions; real Solana swaps typically run $0.001-$0.01 each way

    # Added after real paper-trading data (Aug 2026, 381 trades) showed 14
    # trades losing -98% to -99% — far past the -12% SL — because 30-second
    # polling can't catch a token collapsing between two checks. Those 14
    # trades alone caused MORE than the entire session's net loss; the
    # other 367 trades combined were roughly breakeven. Root causes: (1)
    # no cooldown stopped the bot re-buying the same already-crashing
    # tokens repeatedly, (2) no faster check loop for open positions to
    # shrink the gap window. Both addressed below.
    blacklist_after_stop_loss: bool = True
    blacklist_cooldown_hours: float = 24.0   # how long a token is off-limits after a stop-loss
    circuit_breaker_multiplier: float = 2.0  # a single-poll drop beyond (SL% * this) gets tagged
                                              # 'circuit_breaker' instead of 'stop_loss' for analysis —
                                              # it's still sold immediately either way; this just labels
                                              # "normal stop" vs "something went badly wrong" separately
    position_check_interval_seconds: int = 10  # how often OPEN positions are re-checked (fast loop);
                                                # separate from poll_interval_seconds, which is the
                                                # slower full-discovery cycle. Shrinks the gap window
                                                # a real crash can hide inside, without hammering the
                                                # discovery API every few seconds too.


@dataclass
class BotConfig:
    buy: BuyParams = field(default_factory=BuyParams)
    sell: SellParams = field(default_factory=SellParams)
    risk: RiskParams = field(default_factory=RiskParams)
    mode: str = "paper"                    # "paper" or "live" — live not wired up yet
    poll_interval_seconds: int = 30
    dexscreener_chain: str = "solana"


CONFIG = BotConfig()
