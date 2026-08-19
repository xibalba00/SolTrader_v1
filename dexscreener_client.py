"""
Thin wrapper around the DexScreener public API.

Docs: https://docs.dexscreener.com/api/reference
No API key required. Rate limits are per-IP and unofficial — be polite,
don't hammer it in a tight loop.
"""

import time
import requests
from typing import Optional

BASE_URL = "https://api.dexscreener.com"


class DexScreenerClient:
    def __init__(self, chain: str = "solana", request_timeout: int = 10):
        self.chain = chain
        self.timeout = request_timeout
        self.session = requests.Session()

    def get_token_pairs(self, token_address: str) -> list[dict]:
        """Return all trading pairs for a given token mint address."""
        url = f"{BASE_URL}/token-pairs/v1/{self.chain}/{token_address}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def search_pairs(self, query: str) -> list[dict]:
        """Search pairs by token name/symbol/address."""
        url = f"{BASE_URL}/latest/dex/search"
        resp = self.session.get(url, params={"q": query}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("pairs", []) or []

    def get_pair(self, pair_address: str) -> Optional[dict]:
        """Get live data for a single pair by its pair (pool) address."""
        url = f"{BASE_URL}/latest/dex/pairs/{self.chain}/{pair_address}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        pairs = resp.json().get("pairs") or []
        return pairs[0] if pairs else None

    # ---------- Discovery endpoints (token-profiles / token-boosts) ----------
    # These are the real DexScreener discovery endpoints. They return
    # {chainId, tokenAddress, ...} — no liquidity/volume/price yet, so
    # discover_candidate_pairs() below chains them into get_token_pairs()
    # to get full tradeable-pair metrics.
    #
    # Rate limit on all three: 60 requests/minute (DexScreener's published
    # limit for profile/boost endpoints). Pair-lookup endpoints tolerate
    # 300/minute, which is why the fan-out step below is the cheaper part.

    def get_latest_token_profiles(self) -> list[dict]:
        url = f"{BASE_URL}/token-profiles/latest/v1"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else (data.get("data") or [])

    def get_latest_boosted_tokens(self) -> list[dict]:
        url = f"{BASE_URL}/token-boosts/latest/v1"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else (data.get("data") or [])

    def get_top_boosted_tokens(self) -> list[dict]:
        url = f"{BASE_URL}/token-boosts/top/v1"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else (data.get("data") or [])

    def discover_candidate_pairs(
        self,
        sources: tuple[str, ...] = ("profiles", "boosts_latest", "boosts_top"),
        request_delay_seconds: float = 0.3,
    ) -> list[dict]:
        """
        Real discovery pipeline:
          1. Pull token addresses from the requested discovery endpoints
          2. Filter to this client's chain (default: solana)
          3. Deduplicate addresses
          4. Fan out to get_token_pairs() per address for full metrics
          5. Return the highest-liquidity pair per token

        This replaces keyword search as the actual "find new/trending
        tokens" method. It's still bounded by DexScreener's own indexing
        lag (seconds to minutes behind raw on-chain events) and by the
        60 req/min discovery-endpoint limit — it is NOT a sub-second
        sniping feed. For that you'd need Helius/Birdeye webhooks.
        """
        source_fns = {
            "profiles": self.get_latest_token_profiles,
            "boosts_latest": self.get_latest_boosted_tokens,
            "boosts_top": self.get_top_boosted_tokens,
        }

        addresses: set[str] = set()
        for name in sources:
            fn = source_fns.get(name)
            if not fn:
                continue
            try:
                entries = fn()
            except requests.exceptions.HTTPError:
                continue  # one source failing shouldn't kill discovery entirely
            for entry in entries:
                if entry.get("chainId") == self.chain and entry.get("tokenAddress"):
                    addresses.add(entry["tokenAddress"])
            time.sleep(request_delay_seconds)

        candidate_pairs: list[dict] = []
        for address in addresses:
            try:
                pairs = self.get_token_pairs(address)
            except requests.exceptions.HTTPError:
                continue
            if not pairs:
                continue
            best_pair = max(
                pairs,
                key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0),
            )
            candidate_pairs.append(best_pair)
            time.sleep(request_delay_seconds)

        return candidate_pairs


def extract_pair_metrics(pair: dict) -> dict:
    """Normalize the fields the strategy actually needs from a raw pair dict.

    Extended to also pull the multi-timeframe volume/price-change fields
    DexScreener already includes in every pair response — these were
    always present in the API data, just never parsed out before. No
    extra API calls needed for any of this; it's the same response the
    bot already fetches every poll."""
    volume = pair.get("volume") or {}
    price_change = pair.get("priceChange") or {}
    return {
        "pair_address": pair.get("pairAddress"),
        "base_token": pair.get("baseToken", {}).get("symbol"),
        "base_address": pair.get("baseToken", {}).get("address"),
        "price_usd": float(pair.get("priceUsd") or 0),
        "liquidity_usd": float((pair.get("liquidity") or {}).get("usd") or 0),
        "volume_24h_usd": float(volume.get("h24") or 0),
        "volume_5m_usd": float(volume.get("m5") or 0),
        "volume_1h_usd": float(volume.get("h1") or 0),
        "volume_6h_usd": float(volume.get("h6") or 0),
        "price_change_5m_pct": float(price_change.get("m5") or 0),
        "price_change_1h_pct": float(price_change.get("h1") or 0),
        "price_change_6h_pct": float(price_change.get("h6") or 0),
        "price_change_24h_pct": float(price_change.get("h24") or 0),
        "mcap_usd": float(pair.get("marketCap") or pair.get("fdv") or 0),
        "pair_created_at_ms": pair.get("pairCreatedAt"),
        "dex_id": pair.get("dexId"),
    }


def pair_age_hours(pair_created_at_ms: Optional[int]) -> Optional[float]:
    if not pair_created_at_ms:
        return None
    age_seconds = time.time() - (pair_created_at_ms / 1000)
    return age_seconds / 3600
