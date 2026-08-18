"""
Client for GeckoTerminal's free public API — used ONLY for backtesting,
because it's the one free, no-key source of real historical OHLCV
candles for Solana pools. DexScreener (used elsewhere in this bot) does
not expose historical candles via its free API, only current snapshots.

Docs: https://apiguide.geckoterminal.com/
Rate limit on the free tier: ~30 requests/minute. This client sleeps
between calls to stay well under that.
"""

import time
import requests

BASE_URL = "https://api.geckoterminal.com/api/v2"


class GeckoTerminalClient:
    def __init__(self, network: str = "solana", request_timeout: int = 15, request_delay_seconds: float = 6.5):
        self.network = network
        self.timeout = request_timeout
        self.request_delay = request_delay_seconds  # ~9 req/min, conservative under the free tier's real limit
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None, max_retries: int = 3) -> dict:
        url = f"{BASE_URL}{path}"
        for attempt in range(max_retries):
            resp = self.session.get(url, params=params or {}, timeout=self.timeout)
            if resp.status_code == 429:
                wait = self.request_delay * (attempt + 2)  # back off harder each retry
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(self.request_delay)
            return resp.json()
        resp.raise_for_status()  # exhausted retries, raise the last 429

    def get_trending_pools(self, pages: int = 1) -> list[dict]:
        """Currently-trending pools on the network — used as the candidate
        list for backtesting (we then pull each one's real history)."""
        results = []
        for page in range(1, pages + 1):
            data = self._get(f"/networks/{self.network}/trending_pools", params={"page": page})
            results.extend(data.get("data", []))
        return results

    def get_top_pools(self, pages: int = 1) -> list[dict]:
        """Highest-volume pools on the network — a second, independent
        candidate source alongside trending_pools."""
        results = []
        for page in range(1, pages + 1):
            data = self._get(f"/networks/{self.network}/pools", params={"page": page})
            results.extend(data.get("data", []))
        return results

    def get_ohlcv(
        self,
        pool_address: str,
        timeframe: str = "hour",
        aggregate: int = 1,
        limit: int = 1000,
        before_timestamp: int | None = None,
        currency: str = "usd",
    ) -> list[list[float]]:
        """
        Real historical candles: [timestamp, open, high, low, close, volume].
        timeframe: 'day' | 'hour' | 'minute'. aggregate: e.g. 1, 4, 15 depending
        on timeframe. Free tier caps each call's range at ~6 months back;
        use before_timestamp to page further back if you ever need more.
        """
        params = {"aggregate": aggregate, "limit": limit, "currency": currency}
        if before_timestamp:
            params["before_timestamp"] = before_timestamp
        data = self._get(f"/networks/{self.network}/pools/{pool_address}/ohlcv/{timeframe}", params=params)
        return data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])


def extract_pool_metrics(pool: dict) -> dict:
    """Normalize a GeckoTerminal pool object's current-snapshot fields.
    NOTE: liquidity/fdv here are TODAY's values, not a historical series —
    see module docstring caveat."""
    attrs = pool.get("attributes", {})
    return {
        "pool_address": attrs.get("address"),
        "name": attrs.get("name"),
        "pool_created_at": attrs.get("pool_created_at"),
        "reserve_in_usd": float(attrs.get("reserve_in_usd") or 0),  # liquidity proxy, current snapshot only
        "fdv_usd": float(attrs.get("fdv_usd") or 0),                # mcap proxy, current snapshot only
        "base_token_price_usd": float(attrs.get("base_token_price_usd") or 0),
    }
