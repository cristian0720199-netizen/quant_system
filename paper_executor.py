"""Alpaca paper executor — vol-targeted sizing, guarded by the enforcement layer.

Safety model (fails CLOSED):
  * paper=True always (never a live endpoint)
  * without ALPACA_PAPER_* keys -> refuses to initialize
  * --dry-run (default) computes and validates orders but submits nothing
  * every order goes through Guards; blocked orders are logged, never sent

Usage:
  python paper_executor.py --demo            # offline dry-run, demo signal
  python paper_executor.py --dry-run         # real signal + account, no submit
  python paper_executor.py --execute         # real paper orders (needs keys)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from guards import Guards, Order
from signals import demo_signal, generate, UNIVERSE


def get_client():
    from alpaca.trading.client import TradingClient
    key = os.getenv("ALPACA_PAPER_API_KEY")
    secret = os.getenv("ALPACA_PAPER_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET_KEY not set — "
                         "executor refuses to run without paper credentials.")
    return TradingClient(key, secret, paper=True)


def latest_prices(client, symbols):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    import os
    dc = StockHistoricalDataClient(os.getenv("ALPACA_PAPER_API_KEY"),
                                   os.getenv("ALPACA_PAPER_SECRET_KEY"))
    trades = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbols))
    return {s: float(trades[s].price) for s in symbols}


def real_signal(client):
    """Vol-target signal from recent realized vol of each symbol (daily bars)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from datetime import datetime, timedelta, timezone
    import numpy as np
    dc = StockHistoricalDataClient(os.getenv("ALPACA_PAPER_API_KEY"),
                                   os.getenv("ALPACA_PAPER_SECRET_KEY"))
    req = StockBarsRequest(symbol_or_symbols=UNIVERSE, timeframe=TimeFrame.Day,
                           start=datetime.now(timezone.utc) - timedelta(days=40))
    bars = dc.get_stock_bars(req).df
    pv = {}
    for sym in UNIVERSE:
        try:
            closes = bars.loc[sym]["close"].astype(float)
            rets = closes.pct_change().dropna()
            pv[sym] = float(rets.std())          # realized daily vol proxy
        except Exception:
            pass
    return generate(pv, as_of=str(datetime.now(timezone.utc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="offline demo signal, no Alpaca")
    ap.add_argument("--dry-run", action="store_true", help="validate but submit nothing (default)")
    ap.add_argument("--execute", action="store_true", help="submit paper orders (else dry-run)")
    args = ap.parse_args()

    dry_run = not args.execute
    guards = Guards(allowed_symbols=UNIVERSE, max_position_notional=25_000,
                    max_total_notional=100_000, min_cash_buffer=500)

    if args.demo:
        sig = demo_signal()
        equity, cash, port_notional = 100_000.0, 100_000.0, 0.0
        prices = {"AAPL": 230.0, "MSFT": 420.0, "NVDA": 180.0, "JPM": 240.0, "XOM": 120.0}
        print("DEMO mode (offline)\n")
    else:
        client = get_client()
        acct = client.get_account()
        equity, cash = float(acct.equity), float(acct.cash)
        sig = real_signal(client)
        prices = latest_prices(client, UNIVERSE)
        positions = client.get_all_positions()
        port_notional = sum(float(p.market_value) for p in positions)
        mode = "DRY-RUN" if dry_run else "EXECUTE (paper)"
        print(f"{mode} | equity ${equity:,.0f} | cash ${cash:,.0f}\n")

    print(f"signal as of {sig.as_of} | base exposure {sig.base_exposure:.0%}")
    placed = 0
    for it in sig.intents:
        px = prices.get(it.symbol)
        if not px or px <= 0:
            print(f"  {it.symbol}: no price, skip")
            continue
        target_notional = it.target_weight * equity
        qty = max(0, int(target_notional // px))
        order = Order(it.symbol, qty, "buy", qty * px, px)
        d = guards.check(order, portfolio_notional=port_notional, cash=cash)
        if not d.allowed:
            print(f"  {it.symbol}: BLOCK ({'; '.join(d.violations)})")
            continue
        print(f"  {it.symbol}: {qty} sh @ ${px:.2f} (~${order.notional:,.0f}, "
              f"w={it.target_weight:.2f}) -> {'SUBMIT' if not dry_run else 'dry-run ok'}")
        if not dry_run and not args.demo and qty > 0:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            client.submit_order(MarketOrderRequest(
                symbol=it.symbol, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY))
            placed += 1
    print(f"\n{'submitted' if not dry_run else 'would submit'} {placed} orders "
          f"(see quant_system/logs/enforcement_audit.jsonl)")


if __name__ == "__main__":
    main()
