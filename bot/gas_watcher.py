"""
Gas-reserve watcher for the LIVE wallet (not paper mode).

Purpose: your existing SPL token holdings should never block the bot
from paying transaction fees. This module checks live SOL balance and,
if it's under the configured reserve, sells a slice of an existing
holding at market via Jupiter to top it back up.

This does NOT implement any TP/SL logic on those holdings — per your
instruction, those tokens are sold at market, only when gas is needed,
nothing more elaborate than that.

NOTE: this module is a template. It requires a funded keypair and is
NOT wired to execute real transactions yet — the `_send_transaction`
stub is where that would go, deliberately left unimplemented so this
can't accidentally fire against a real wallet until you explicitly
build and review that piece.
"""

from dataclasses import dataclass

from .config import CONFIG
from .jupiter_client import JupiterClient, SOL_MINT


@dataclass
class Holding:
    token_symbol: str
    token_mint: str
    amount_base_units: int
    decimals: int


class GasWatcher:
    def __init__(self, wallet_pubkey: str):
        self.wallet_pubkey = wallet_pubkey
        self.jupiter = JupiterClient()

    def get_sol_balance(self) -> float:
        """
        Placeholder — real implementation calls getBalance via your RPC
        client (solana-py) against self.wallet_pubkey and divides by
        1e9 for lamports -> SOL. Left unimplemented here since it
        needs your actual RPC endpoint configured.
        """
        raise NotImplementedError("Wire up solana-py getBalance() here with your RPC client.")

    def needs_topup(self, current_sol_balance: float) -> bool:
        return current_sol_balance < CONFIG.risk.gas_reserve_sol

    def pick_holding_to_sell(self, holdings: list[Holding]) -> Holding | None:
        """Simple default: sell from the largest-value holding first.
        Swap this for whatever selection rule you prefer."""
        if not holdings:
            return None
        return max(holdings, key=lambda h: h.amount_base_units)

    def build_topup_quote(self, holding: Holding) -> dict:
        sell_amount = int(holding.amount_base_units * (CONFIG.risk.gas_topup_sell_pct / 100))
        return self.jupiter.get_quote(
            input_mint=holding.token_mint,
            output_mint=SOL_MINT,
            amount_lamports_or_base_units=sell_amount,
            slippage_bps=300,  # gas top-ups are urgency-driven; a bit looser than normal trades
        )

    def _send_transaction(self, quote: dict):
        raise NotImplementedError(
            "Live execution intentionally not implemented in this scaffold. "
            "This is where you'd call Jupiter's /swap endpoint with the "
            "quote, sign the returned transaction with your keypair via "
            "solana-py, and submit it via your RPC client."
        )
