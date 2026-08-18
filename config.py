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
    min_liquidity_usd: float = 15_000
    min_volume_24h_usd: float = 50_000
    min_age_minutes: int = 240             # 4 hours
    max_age_hours: int = 48
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
    max_hold_hours: float = 6.0
    max_price_impact_pct: float = 2.0      # abort sell if quote impact exceeds this


@dataclass
class RiskParams:
    starting_balance_usd: float = 100.0    # paper balance only, edit freely
    position_size_pct: float = 2.0         # % of current bankroll per trade
    max_concurrent_positions: int = 10     # updated from 5 — see note above
    gas_reserve_sol: float = 0.05          # never let live SOL drop below this
    gas_topup_sell_pct: float = 25.0       # sell this % of a holding to top up gas
    assumed_gas_cost_usd_per_trade: float = 0.02  # realistic round-trip (buy+sell) under normal (non-congested) conditions; real Solana swaps typically run $0.001-$0.01 each way


@dataclass
class BotConfig:
    buy: BuyParams = field(default_factory=BuyParams)
    sell: SellParams = field(default_factory=SellParams)
    risk: RiskParams = field(default_factory=RiskParams)
    mode: str = "paper"                    # "paper" or "live" — live not wired up yet
    poll_interval_seconds: int = 30
    dexscreener_chain: str = "solana"


CONFIG = BotConfig()
