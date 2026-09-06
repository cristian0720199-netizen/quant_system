"""Live QuantMind forecast -> guarded paper-trade intents, end to end.

Trains QuantBrain (LSTM) on the vega panel, produces a per-symbol forward-vol
forecast for the most recent bar, converts to inverse-vol weights, and runs
them through the Guards enforcement layer. Purely offline (no Alpaca needed to
generate the signal); the executor only submits when you pass --execute.

Usage:
  python live_forecast.py            # forecast + intents + guard check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from quantmind import QuantDataEngine, FeatureIntelligenceAgents, build_brain, make_sequences, LEAK, NON_FEATURE
from signals import generate
from guards import Guards, Order

LOOKBACK = 16
TOP_N = 15


def forecast_latest() -> dict[str, float]:
    """Train on the full panel and forecast forward vol for the latest bar per symbol."""
    engine = QuantDataEngine("_prepped_stk")
    df, feats = engine.build()
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # feature agents select on the whole history
    agents = FeatureIntelligenceAgents(TOP_N)
    sel, _ = agents.select(df[feats], df["forward_realized_vol"].to_numpy())

    from sklearn.preprocessing import StandardScaler
    import keras

    fsc = StandardScaler().fit(df[sel])
    y = df["forward_realized_vol"].to_numpy()
    y_mu, y_sd = y.mean(), y.std()

    # build per-symbol sequences (respect symbol boundaries)
    Xs, ys = [], []
    last_seq_by_symbol: dict[str, np.ndarray] = {}
    for sym, g in df.groupby("symbol", sort=False):
        Xg = fsc.transform(g[sel])
        yg = ((g["forward_realized_vol"].to_numpy() - y_mu) / y_sd).clip(-2.5, 2.5) / 2.5
        Xs_g, ys_g = make_sequences(Xg, yg, LOOKBACK)
        Xs.append(Xs_g); ys.append(ys_g)
        last_seq_by_symbol[sym] = Xg[-LOOKBACK:]  # most recent window
    X = np.concatenate(Xs); Y = np.concatenate(ys)

    model = build_brain(len(sel), LOOKBACK)
    cbs = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]
    model.fit(X, Y, validation_split=0.15, epochs=40, batch_size=128, callbacks=cbs, verbose=0)

    # forecast each symbol's latest window -> back to vol units
    symbols = list(last_seq_by_symbol)
    seqs = np.stack([last_seq_by_symbol[s] for s in symbols]).astype("float32")
    pred = model.predict(seqs, verbose=0).ravel()
    pred_vol = np.clip(pred * 2.5 * y_sd + y_mu, 1e-6, None)
    return dict(zip(symbols, pred_vol)), model


def main():
    print("training QuantBrain on full panel and forecasting latest vol per symbol...\n")
    pred_vols, model = forecast_latest()
    model.save(Path(__file__).parent / "models" / "brain_live.keras")

    sig = generate(pred_vols, as_of="latest")
    print(f"forward-vol forecast -> inverse-vol weights (base exposure {sig.base_exposure:.0%}):")
    for it in sorted(sig.intents, key=lambda x: x.pred_vol):
        print(f"  {it.symbol:6} pred_vol {it.pred_vol:.4f}  ->  weight {it.target_weight:.3f}")

    # run through enforcement (demo equity/prices for the guard check)
    guards = Guards(allowed_symbols=list(pred_vols), max_position_notional=25_000,
                    max_total_notional=100_000, min_cash_buffer=500)
    equity, cash = 100_000.0, 100_000.0
    print("\nguard check @ $100k equity (no prices -> sizing only):")
    for it in sig.intents:
        notional = it.target_weight * equity
        o = Order(it.symbol, qty=0 if notional <= 0 else 1, side="buy",
                  notional=notional, price=1.0)
        d = guards.check(o, portfolio_notional=0, cash=cash)
        status = "ALLOW" if d.allowed else "BLOCK " + "; ".join(d.violations)
        print(f"  {it.symbol:6} target ${notional:>8,.0f} -> {status}")


if __name__ == "__main__":
    main()
