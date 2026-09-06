"""Mechanical enforcement layer: hooks + rules. Every order passes through here.

Hooks  = binary pre-trade checks (invalid trade, allowed symbol, market open)
Rules  = numeric/risk checks (max position, max notional, min cash buffer)
Audit  = every decision (allow/block) is appended to a JSONL audit log.

The executor NEVER places an order that guards.reject. Enforcement is code,
not prose — the AI cannot silently disable it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG = Path(__file__).parent / "logs" / "enforcement_audit.jsonl"
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Order:
    symbol: str
    qty: float
    side: str            # "buy" | "sell"
    notional: float      # qty * price
    price: float = 0.0


@dataclass
class Decision:
    allowed: bool
    violations: list[str] = field(default_factory=list)
    order: dict = field(default_factory=dict)


class Guards:
    def __init__(self, *, max_position_notional=10_000, max_total_notional=50_000,
                 min_cash_buffer=1_000, allowed_symbols=None,
                 market_hours_only=False, is_market_open=None):
        self.cfg = dict(
            max_position_notional=max_position_notional,
            max_total_notional=max_total_notional,
            min_cash_buffer=min_cash_buffer,
            allowed_symbols=set(allowed_symbols or []),
            market_hours_only=market_hours_only,
        )
        self._is_market_open = is_market_open or (lambda: True)

    # ---- hooks (binary) ----
    def _hook_valid(self, o: Order) -> str | None:
        if o.qty <= 0:
            return "qty<=0"
        if o.price < 0:
            return "negative price"
        if o.side not in ("buy", "sell"):
            return "bad side"
        if self.cfg["allowed_symbols"] and o.symbol not in self.cfg["allowed_symbols"]:
            return f"{o.symbol} not in allowed universe"
        return None

    # ---- rules (numeric) ----
    def _rule_max_position(self, o: Order) -> str | None:
        if o.notional > self.cfg["max_position_notional"]:
            return f"position notional {o.notional:.0f} > max {self.cfg['max_position_notional']}"
        return None

    def _rule_total_risk(self, o: Order, portfolio_notional: float) -> str | None:
        if o.side == "buy" and portfolio_notional + o.notional > self.cfg["max_total_notional"]:
            return f"total notional would exceed {self.cfg['max_total_notional']}"
        return None

    def _rule_cash(self, o: Order, cash: float) -> str | None:
        if o.side == "buy" and cash - o.notional < self.cfg["min_cash_buffer"]:
            return f"buy would breach min cash buffer {self.cfg['min_cash_buffer']}"
        return None

    def check(self, o: Order, *, portfolio_notional=0.0, cash=0.0) -> Decision:
        violations = []
        v = self._hook_valid(o)
        if v:
            violations.append(f"hook:valid:{v}")
        if self.cfg["market_hours_only"] and not self._is_market_open():
            violations.append("hook:market_closed")
        for r in (self._rule_max_position(o),
                  self._rule_total_risk(o, portfolio_notional),
                  self._rule_cash(o, cash)):
            if r:
                violations.append(f"rule:{r}")
        d = Decision(allowed=not violations, violations=violations, order=asdict(o))
        self._audit(d)
        return d

    def _audit(self, d: Decision):
        rec = {"ts": datetime.now(timezone.utc).isoformat(),
               "allowed": d.allowed, "violations": d.violations, "order": d.order}
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    g = Guards(allowed_symbols=["AAPL", "MSFT"], max_position_notional=1000)
    tests = [
        Order("AAPL", 5, "buy", 500, 100),     # valid
        Order("AAPL", 0, "buy", 0, 100),       # bad qty
        Order("TSLA", 5, "buy", 500, 100),     # not allowed
        Order("MSFT", 50, "buy", 5000, 100),   # exceeds max position
    ]
    for t in tests:
        d = g.check(t, portfolio_notional=0, cash=10_000)
        print(f"{t.symbol} qty={t.qty} notional={t.notional}: "
              f"{'ALLOW' if d.allowed else 'BLOCK ' + '; '.join(d.violations)}")
