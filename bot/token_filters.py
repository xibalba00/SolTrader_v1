# file: token_filters.py
from datetime import datetime, timezone
from typing import Optional, Set

def is_stablecoin_or_blacklisted(
    symbol: Optional[str],
    address: Optional[str],
    price_usd: Optional[float],
    known_stable_symbols: Optional[Set[str]] = None,
    known_stable_addresses: Optional[Set[str]] = None,
    price_tolerance_pct: float = 2.5,
) -> bool:
    if known_stable_symbols is None:
        known_stable_symbols = {"USDC", "USDT", "DAI", "BUSD", "USDP"}
    if known_stable_addresses is None:
        known_stable_addresses = set()

    if symbol:
        if symbol.strip().upper() in known_stable_symbols:
            return True
    if address and address in known_stable_addresses:
        return True
    if price_usd is not None:
        tol = price_tolerance_pct / 100.0
        if abs(price_usd - 1.0) <= tol:
            return True
    return False