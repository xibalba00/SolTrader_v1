"""
Thin wrapper around the Vybe Network API (https://docs.vybenetwork.com).

Vybe is the data source for the "smart-money cohort" experiment: it exposes
per-token top-trader leaderboards (ranked by realized PnL, win rate, volume)
and per-wallet trade/PnL history, which is the primitive this whole idea
needs and that GeckoTerminal/DexScreener don't have.

IMPORTANT — verify before running:
Endpoint paths below are assembled from Vybe's public docs/reference pages
and partner examples (docs.vybenetwork.com/reference), not from a live test
against your own key (this sandbox can't reach vybenetwork.com — it's not on
the allowed domain list). Before running this against your real key on the
VPS, hit each endpoint once manually (curl or Postman) and confirm the exact
path/params/response shape match what's assumed here. The endpoints most
likely to need adjustment are get_wallet_pnl and get_wallet_trades — those
were less consistently documented across sources than the top-traders and
top-holders endpoints.

FREE TIER (confirmed as of Aug 2026): 4 requests/minute, 12,000 credits/month.
That is the binding constraint on this whole pipeline — see the rate limiter
below and the phase scripts for how they budget around it.
"""

import os
import time
import requests
from dataclasses import dataclass
from typing import Optional

BASE_URL = "https://api.vybenetwork.xyz"  # VERIFY: confirm exact base host with your API key/docs page
FREE_TIER_REQUESTS_PER_MINUTE = 4


@dataclass
class TopTraderRow:
    wallet_address: str
    token_mint: str
    realized_pnl_usd: float
    volume_usd: float
    trade_count: int
    win_rate_pct: Optional[float]


class VybeRateLimiter:
    """Enforces the free-tier 4 req/min ceiling with a simple sliding delay,
    plus a hard stop on total credits spent this run so a bug can't silently
    burn the whole monthly allowance in one session."""

    def __init__(self, requests_per_minute: int = FREE_TIER_REQUESTS_PER_MINUTE, credit_budget: int = 10_000):
        self.min_interval = 60.0 / requests_per_minute
        self._last_call = 0.0
        self.credit_budget = credit_budget
        self.credits_spent = 0  # best-effort counter; Vybe doesn't echo credit cost per-call in all responses

    def wait(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()

    def check_budget(self, assumed_cost: int = 1) -> None:
        if self.credits_spent + assumed_cost > self.credit_budget:
            raise RuntimeError(
                f"Stopping before exceeding the configured credit_budget "
                f"({self.credits_spent}/{self.credit_budget} spent). Raise "
                f"credit_budget explicitly if you meant to keep going — this "
                f"guard exists so a bug can't burn the whole monthly quota."
            )
        self.credits_spent += assumed_cost


class VybeClient:
    def __init__(self, api_key: Optional[str] = None, credit_budget: int = 10_000, request_timeout: int = 15):
        self.api_key = api_key or os.environ.get("VYBE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set VYBE_API_KEY (env var) or pass api_key= explicitly.")
        self.session = requests.Session()
        self.session.headers.update({"X-API-KEY": self.api_key})  # VERIFY: confirm header name in your docs
        self.timeout = request_timeout
        self.limiter = VybeRateLimiter(credit_budget=credit_budget)

    def _get(self, path: str, params: Optional[dict] = None, assumed_credit_cost: int = 1) -> dict:
        self.limiter.check_budget(assumed_credit_cost)
        self.limiter.wait()
        url = f"{BASE_URL}{path}"
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        if resp.status_code == 429:
            # Free-tier burst protection: back off hard and retry once.
            time.sleep(20)
            resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ---------- Token-scoped: who trades this token, and how well ----------

    def get_top_pnl_traders(self, mint_address: str, limit: int = 50, resolution_days: int = 30) -> list[TopTraderRow]:
        """GET /v4/tokens/{mint}/top-pnl-traders — top wallets by realized
        PnL on this specific token. This is the core Phase-1 primitive:
        run it over a seed list of tokens to build the wallet cohort."""
        data = self._get(f"/v4/tokens/{mint_address}/top-pnl-traders", params={"limit": limit, "days": resolution_days})
        rows = []
        for r in data.get("data", data.get("traders", [])):
            rows.append(TopTraderRow(
                wallet_address=r.get("ownerAddress") or r.get("walletAddress") or r.get("address"),
                token_mint=mint_address,
                realized_pnl_usd=float(r.get("realizedPnlUsd") or r.get("pnlUsd") or 0),
                volume_usd=float(r.get("volumeUsd") or 0),
                trade_count=int(r.get("tradeCount") or r.get("trades") or 0),
                win_rate_pct=float(r["winRate"]) if r.get("winRate") is not None else None,
            ))
        return rows

    def get_top_holders(self, mint_address: str, limit: int = 100) -> list[dict]:
        """GET /v4/tokens/{mint}/top-holders — supplementary signal, not the
        primary Phase-1 source (holders != active traders, but cheap to pull
        alongside top-pnl-traders for the same token)."""
        data = self._get(f"/v4/tokens/{mint_address}/top-holders", params={"limit": limit})
        return data.get("data", [])

    # ---------- Wallet-scoped: what has this wallet actually done ----------

    def get_wallet_pnl(self, wallet_address: str, resolution_days: int = 30) -> dict:
        """GET /v4/wallets/{address}/pnl (path VERIFY) — realized/unrealized
        PnL, win rate, volume for one wallet. Used in Phase 1 scoring to
        confirm a wallet that shows up across many tokens is actually good,
        not just active."""
        data = self._get(f"/v4/wallets/{wallet_address}/pnl", params={"days": resolution_days})
        return data

    def get_wallet_trades(self, wallet_address: str, limit: int = 200, before_ts: Optional[int] = None) -> list[dict]:
        """GET /v4/trades filtered by wallet (path/params VERIFY) — this is
        the Phase-2 primitive: pull a monitored wallet's actual buy events
        with real timestamps, to discover which tokens/pools it touched and
        when. Paginate with before_ts on repeat calls once you exceed one
        page of history for an active wallet."""
        params = {"walletAddress": wallet_address, "limit": limit}
        if before_ts:
            params["beforeTimestamp"] = before_ts
        data = self._get("/v4/trades", params=params, assumed_credit_cost=2)  # trade-history calls likely cost more
        return data.get("data", data.get("trades", []))
