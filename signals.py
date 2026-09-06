"""Signal adapter: QuantMind forward-vol forecast -> position sizing intent.

The validated edge is VOLATILITY (R2 0.48), not direction. So the signal is a
RISK-SIZING overlay on a base long-equities exposure, not a direction bet:
  predicted vol high  -> de-risk (target a smaller position)
  predicted vol low   -> full base exposure

This is vol-targeting: keep portfolio risk roughly constant by sizing in
inverse proportion to predicted vol. It monetizes the vol forecast without
needing a direction edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

UNIVERSE = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]  # liquid vega names


@dataclass
class Intent:
    symbol: str
    target_weight: float   # fraction of equity to hold
    pred_vol: float
    reason: str


@dataclass
class SignalSet:
    as_of: str
    base_exposure: float          # total equity fraction to deploy
    intents: list[Intent] = field(default_factory=list)


def vol_target_weights(pred_vol: pd.Series, base_exposure: float = 0.95) -> pd.Series:
    """Inverse-vol sizing normalized so weights sum to base_exposure."""
    inv = 1.0 / pred_vol.clip(lower=1e-6)
    w = inv / inv.sum()
    return w * base_exposure


def generate(pred_vols: dict[str, float], base_exposure: float = 0.95,
             as_of: str = "") -> SignalSet:
    pv = pd.Series(pred_vols)
    w = vol_target_weights(pv, base_exposure)
    intents = [
        Intent(sym, float(w[sym]), float(pv[sym]),
               "inverse-vol target weight")
        for sym in w.index
    ]
    return SignalSet(as_of=as_of, base_exposure=base_exposure, intents=intents)


def demo_signal() -> SignalSet:
    """Deterministic demo signal from realistic vol levels (offline testing)."""
    demo = {"AAPL": 0.0018, "MSFT": 0.0016, "NVDA": 0.0031, "JPM": 0.0014, "XOM": 0.0012}
    return generate(demo, as_of="demo")


if __name__ == "__main__":
    s = demo_signal()
    for i in s.intents:
        print(f"  {i.symbol}: weight {i.target_weight:.3f} (pred_vol {i.pred_vol:.4f})")
