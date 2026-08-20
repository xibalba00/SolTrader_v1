"""
Wrapper around Jupiter's Quote API (Solana swap aggregator).

Docs: https://dev.jup.ag/docs/api/quote-api

This client only ever fetches QUOTES — it does not sign or send
transactions. That keeps it safe to use in paper mode (so your simulated
fills reflect real market depth and price impact) without any risk of
accidentally touching a live wallet. Live execution is a separate,
explicit module (see live_executor.py, not yet wired up).
"""

import requests

QUOTE_URL = "https://lite-api.jup.ag/swap/v1/quote"
SOL_MINT = "So11111111111111111111111111111111111111112"


class JupiterClient:
    def __init__(self, request_timeout: int = 10):
        self.timeout = request_timeout
        self.session = requests.Session()

    def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount_lamports_or_base_units: int,
        slippage_bps: int = 100,
    ) -> dict:
        """
        Fetch a swap quote. amount is in the input token's smallest unit
        (lamports for SOL, or the token's raw base units).

        Returns the raw Jupiter quote response, which includes
        'priceImpactPct' and 'outAmount' — the two fields the strategy
        cares about most.
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount_lamports_or_base_units,
            "slippageBps": slippage_bps,
        }
        resp = self.session.get(QUOTE_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def quote_price_impact_pct(self, quote: dict) -> float:
        raw = quote.get("priceImpactPct")
        try:
            return float(raw) * 100 if raw is not None and abs(float(raw)) < 1 else float(raw or 0)
        except (TypeError, ValueError):
            return 0.0


def estimate_price_impact_pct(trade_size_usd: float, liquidity_usd: float, max_pct: float = 25.0) -> float:
    """
    Shared price-impact approximation used by BOTH the backtest engine and
    the paper-trading engine, so they can't drift out of sync with each
    other again. Rough constant-product-style scaling: impact grows with
    trade size relative to available liquidity. Not a precise AMM
    calculation — a reasonable estimate, clipped at max_pct so a
    tiny/broken liquidity value can't produce a nonsense result.
    """
    if liquidity_usd <= 0:
        return max_pct
    impact = (trade_size_usd / liquidity_usd) * 50  # empirical-ish scaling factor
    return min(impact, max_pct)
