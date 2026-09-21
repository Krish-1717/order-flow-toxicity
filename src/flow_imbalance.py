"""
src/flow_imbalance.py -- Order Flow Imbalance (OFI) signals for order-flow-toxicity
Day 11: tick-level buy/sell pressure, OFI ratio, cumulative delta, and toxicity score.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Tick

@dataclass
class Tick:
    """Single market data tick."""
    timestamp:   float   # UNIX seconds
    price:       float
    volume:      float
    bid:         float   # best bid at trade time
    ask:         float   # best ask at trade time
    side:        str     # 'buy' | 'sell' | 'unknown'

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def classify_side(self) -> str:
        """Lee-Ready tick-test fallback if side is unknown."""
        if self.side in ('buy', 'sell'):
            return self.side
        if self.price >= self.ask:
            return 'buy'
        if self.price <= self.bid:
            return 'sell'
        return 'unknown'


# OFI Window

@dataclass
class OFIWindow:
    """Order Flow Imbalance computed over a rolling window of ticks."""
    buy_volume:    float
    sell_volume:   float
    total_volume:  float
    n_buy:         int
    n_sell:        int
    n_ticks:       int
    cum_delta:     float   # buy_vol - sell_vol
    ofi_ratio:     float   # (buy - sell) / total
    vwap_buy:      float
    vwap_sell:     float
    toxicity:      float   # normalised [0, 1] toxicity estimate

    @property
    def pressure(self) -> str:
        if self.ofi_ratio > 0.15:
            return 'bullish'
        if self.ofi_ratio < -0.15:
            return 'bearish'
        return 'neutral'

    def __str__(self) -> str:
        return (
            f"OFI={self.ofi_ratio:+.4f} | delta={self.cum_delta:+.1f} | "
            f"buy%={self.buy_volume/max(self.total_volume,1e-9)*100:.1f} | "
            f"toxicity={self.toxicity:.4f} | {self.pressure}"
        )


# Flow Imbalance Calculator

class FlowImbalanceCalc:
    """
    Computes Order Flow Imbalance (OFI) over a sliding window.
    Toxicity is estimated as: |ofi_ratio| * spread_factor,
    where spread_factor = mean_spread / mean_price (normalised cost of adverse selection).
    """

    def __init__(self, window: int = 50):
        self.window = window
        self._ticks: List[Tick] = []

    # public API

    def add_tick(self, tick: Tick) -> None:
        self._ticks.append(tick)
        if len(self._ticks) > self.window:
            self._ticks.pop(0)

    def compute(self) -> Optional[OFIWindow]:
        if not self._ticks:
            return None
        return self._compute_window(self._ticks)

    def compute_rolling(self, ticks: List[Tick]) -> List[OFIWindow]:
        """Compute OFI for every tick position once window is full."""
        results: List[OFIWindow] = []
        buf: List[Tick] = []
        for t in ticks:
            buf.append(t)
            if len(buf) > self.window:
                buf.pop(0)
            if len(buf) == self.window:
                results.append(self._compute_window(buf))
        return results

    # internal

    @staticmethod
    def _compute_window(ticks: List[Tick]) -> OFIWindow:
        buy_vol = sell_vol = 0.0
        buy_pv  = sell_pv  = 0.0
        n_buy = n_sell = 0
        spreads: List[float] = []
        prices:  List[float] = []

        for t in ticks:
            side = t.classify_side()
            spreads.append(t.spread)
            prices.append(t.price)
            if side == 'buy':
                buy_vol += t.volume
                buy_pv  += t.price * t.volume
                n_buy   += 1
            elif side == 'sell':
                sell_vol += t.volume
                sell_pv  += t.price * t.volume
                n_sell   += 1

        total_vol = buy_vol + sell_vol
        cum_delta = buy_vol - sell_vol
        ofi_ratio = cum_delta / total_vol if total_vol > 0 else 0.0

        vwap_buy  = buy_pv  / buy_vol  if buy_vol  > 0 else 0.0
        vwap_sell = sell_pv / sell_vol if sell_vol > 0 else 0.0

        mean_spread = sum(spreads) / len(spreads) if spreads else 0.0
        mean_price  = sum(prices)  / len(prices)  if prices  else 1.0
        spread_factor = mean_spread / mean_price if mean_price > 0 else 0.0

        # toxicity: absolute flow imbalance amplified by proportional spread
        raw_toxicity = abs(ofi_ratio) * (1.0 + 50.0 * spread_factor)
        toxicity = min(raw_toxicity, 1.0)

        return OFIWindow(
            buy_volume=buy_vol, sell_volume=sell_vol, total_volume=total_vol,
            n_buy=n_buy, n_sell=n_sell, n_ticks=len(ticks),
            cum_delta=cum_delta, ofi_ratio=ofi_ratio,
            vwap_buy=vwap_buy, vwap_sell=vwap_sell,
            toxicity=toxicity,
        )


# Regime Detector

@dataclass
class FlowRegime:
    label:         str    # 'toxic', 'pressure', 'balanced'
    toxicity:      float
    ofi_ratio:     float
    n_windows:     int
    pct_toxic:     float  # fraction of windows above toxicity threshold

def detect_regime(windows: List[OFIWindow], toxicity_thresh: float = 0.4) -> FlowRegime:
    if not windows:
        return FlowRegime('balanced', 0.0, 0.0, 0, 0.0)
    n = len(windows)
    mean_tox = sum(w.toxicity for w in windows) / n
    mean_ofi = sum(w.ofi_ratio for w in windows) / n
    pct_toxic = sum(1 for w in windows if w.toxicity > toxicity_thresh) / n
    if pct_toxic > 0.3 or mean_tox > toxicity_thresh:
        label = 'toxic'
    elif abs(mean_ofi) > 0.15:
        label = 'pressure'
    else:
        label = 'balanced'
    return FlowRegime(label=label, toxicity=mean_tox, ofi_ratio=mean_ofi,
                      n_windows=n, pct_toxic=pct_toxic)


# CLI demo

if __name__ == '__main__':
    import random
    rng = random.Random(42)

    def _sim_ticks(n: int, base_price: float = 100.0,
                   buy_bias: float = 0.5) -> List[Tick]:
        ticks = []
        price = base_price
        t = 0.0
        for _ in range(n):
            price += rng.gauss(0, 0.05)
            spread = abs(rng.gauss(0.02, 0.005))
            bid = price - spread / 2
            ask = price + spread / 2
            side = 'buy' if rng.random() < buy_bias else 'sell'
            vol = max(1.0, rng.gauss(100, 30))
            ticks.append(Tick(t, price, vol, bid, ask, side))
            t += rng.expovariate(10)
        return ticks

    print('=== Balanced flow ===')
    balanced = _sim_ticks(500, buy_bias=0.50)
    calc = FlowImbalanceCalc(window=50)
    wins = calc.compute_rolling(balanced)
    regime = detect_regime(wins)
    print(f'Regime: {regime.label}  toxicity={regime.toxicity:.4f}  ofi={regime.ofi_ratio:+.4f}  pct_toxic={regime.pct_toxic:.2%}')
    if wins:
        print('Last window:', wins[-1])

    print()
    print('=== Buy-pressure flow ===')
    bullish = _sim_ticks(500, buy_bias=0.72)
    wins2 = calc.compute_rolling(bullish)
    regime2 = detect_regime(wins2)
    print(f'Regime: {regime2.label}  toxicity={regime2.toxicity:.4f}  ofi={regime2.ofi_ratio:+.4f}  pct_toxic={regime2.pct_toxic:.2%}')
    if wins2:
        print('Last window:', wins2[-1])

    print()
    print('=== Wide-spread toxic flow ===')
    # simulate wide spreads = higher toxicity
    toxic_ticks = []
    for i in range(500):
        price = 100 + rng.gauss(0, 0.1)
        spread = abs(rng.gauss(0.15, 0.03))   # 5x wider spread
        bid, ask = price - spread/2, price + spread/2
        side = 'buy' if rng.random() < 0.65 else 'sell'
        vol = max(1.0, rng.gauss(80, 20))
        toxic_ticks.append(Tick(float(i)*0.1, price, vol, bid, ask, side))
    wins3 = calc.compute_rolling(toxic_ticks)
    regime3 = detect_regime(wins3)
    print(f'Regime: {regime3.label}  toxicity={regime3.toxicity:.4f}  ofi={regime3.ofi_ratio:+.4f}  pct_toxic={regime3.pct_toxic:.2%}')
    if wins3:
        print('Last window:', wins3[-1])
