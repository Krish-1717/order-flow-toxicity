"""
src/vpin.py -- Volume-Synchronised Probability of Informed Trading (VPIN)
order-flow-toxicity Day 1 Commit 1
Reference: Easley, Lopez de Prado, O'Hara (2012)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Trade:
    price: float
    volume: float
    side: Optional[str] = None   # 'buy', 'sell', or None (classify by tick rule)


@dataclass
class VPINResult:
    vpin: float            # VPIN estimate â [0, 1]
    buy_volume: float
    sell_volume: float
    total_volume: float
    n_buckets: int
    bucket_imbalances: List[float] = field(default_factory=list)

    def is_toxic(self, threshold: float = 0.5) -> bool:
        return self.vpin >= threshold

    def summary(self) -> str:
        flag = "TOXIC" if self.is_toxic() else "normal"
        return (f"{flag}  VPIN={self.vpin:.4f}  "
                f"buy={self.buy_volume:,.0f}  sell={self.sell_volume:,.0f}  "
                f"buckets={self.n_buckets}")


def _classify_tick(trades: List[Trade]) -> List[Trade]:
    """Apply tick rule: compare to previous price to classify buy/sell."""
    result = []
    last_price = None
    for t in trades:
        if t.side is not None:
            result.append(t)
            last_price = t.price
            continue
        if last_price is None:
            side = "buy"
        elif t.price > last_price:
            side = "buy"
        elif t.price < last_price:
            side = "sell"
        else:
            side = result[-1].side if result else "buy"
        result.append(Trade(t.price, t.volume, side))
        last_price = t.price
    return result


def _bulk_classify(trades: List[Trade], sigma_dv: float) -> List[Tuple[float, float]]:
    """Bulk volume classification using normal CDF of price change."""
    result = []
    prices = [t.price for t in trades]
    volumes = [t.volume for t in trades]
    n = len(prices)
    for i in range(n):
        if trades[i].side is not None:
            bv = volumes[i] if trades[i].side == "buy" else 0.0
            sv = volumes[i] - bv
        else:
            if i == 0:
                dp = 0.0
            else:
                dp = prices[i] - prices[i-1]
            z = dp / sigma_dv if sigma_dv > 0 else 0.0
            buy_frac = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
            bv = volumes[i] * buy_frac
            sv = volumes[i] * (1 - buy_frac)
        result.append((bv, sv))
    return result


def compute_vpin(trades: List[Trade], bucket_size: Optional[float] = None,
                 n_window: int = 50, use_bulk: bool = True) -> VPINResult:
    """
    Compute VPIN over a list of trades.

    Args:
        trades: list of Trade objects
        bucket_size: volume per bucket (default: total_volume / 50)
        n_window: number of buckets in rolling window
        use_bulk: if True, use bulk volume classification; else tick rule
    """
    if not trades:
        return VPINResult(vpin=0.0, buy_volume=0.0, sell_volume=0.0,
                          total_volume=0.0, n_buckets=0)

    total_vol = sum(t.volume for t in trades)
    if bucket_size is None:
        bucket_size = total_vol / max(n_window, 1)

    if use_bulk:
        prices = [t.price for t in trades]
        returns = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        sigma_dv = (sum(r**2 for r in returns) / len(returns))**0.5 if returns else 1.0
        classified = _bulk_classify(trades, sigma_dv)
    else:
        classified_trades = _classify_tick(trades)
        classified = [(t.volume if t.side == "buy" else 0.0,
                       t.volume if t.side == "sell" else 0.0)
                      for t in classified_trades]

    # Fill volume buckets
    buckets: List[Tuple[float, float]] = []
    bv_accum, sv_accum, vol_accum = 0.0, 0.0, 0.0

    for bv, sv in classified:
        remaining_vol = bv + sv
        remaining_bv = bv
        remaining_sv = sv
        while remaining_vol > 0:
            space = bucket_size - vol_accum
            take = min(space, remaining_vol)
            frac = take / (remaining_vol + 1e-12)
            bv_accum += remaining_bv * frac
            sv_accum += remaining_sv * frac
            vol_accum += take
            remaining_vol -= take
            remaining_bv *= (1 - frac)
            remaining_sv *= (1 - frac)
            if vol_accum >= bucket_size - 1e-9:
                buckets.append((bv_accum, sv_accum))
                bv_accum, sv_accum, vol_accum = 0.0, 0.0, 0.0

    if not buckets:
        return VPINResult(vpin=0.0, buy_volume=bv_accum, sell_volume=sv_accum,
                          total_volume=total_vol, n_buckets=0)

    # Rolling VPIN
    window = buckets[-n_window:]
    imbalances = [abs(b - s) / (b + s + 1e-12) for b, s in window]
    vpin = sum(imbalances) / len(imbalances)

    total_buy = sum(b for b, _ in buckets)
    total_sell = sum(s for _, s in buckets)

    return VPINResult(
        vpin=vpin,
        buy_volume=total_buy,
        sell_volume=total_sell,
        total_volume=total_vol,
        n_buckets=len(buckets),
        bucket_imbalances=imbalances,
    )
