from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Protocol


class HasRiskContext(Protocol):
    def get_cash(self) -> float:
        ...
    def get_position(self, ts_code: str) -> float:
        ...
    def log(self, msg: str) -> None:
        ...


@dataclass
class RiskConfig:
    max_order_value_pct_of_cash: float = 0.2
    max_orders_per_day: int = 50
    banned_ts_codes: List[str] = field(default_factory=list)


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    ts_code: Optional[str] = None
    side: Optional[str] = None
    price: Optional[float] = None
    qty: Optional[int] = None
    dt: Optional[datetime] = None


class RiskManager:
    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        self._daily_order_count: Dict[date, int] = {}

    def check_order(self, ts_code: str, side: str, price: float, qty: int, trade_dt: datetime, ctx: HasRiskContext) -> RiskDecision:
        side = side.lower()
        qty = int(qty)
        price = float(price)
        trade_date = trade_dt.date()
        if ts_code in self.config.banned_ts_codes:
            return RiskDecision(allowed=False, reason="ts_code in banned list", ts_code=ts_code, side=side, price=price, qty=qty, dt=trade_dt)
        cnt = self._daily_order_count.get(trade_date, 0)
        if cnt >= self.config.max_orders_per_day:
            return RiskDecision(allowed=False, reason="daily order count limit reached", ts_code=ts_code, side=side, price=price, qty=qty, dt=trade_dt)
        if side == "buy":
            cash = ctx.get_cash()
            max_notional = cash * self.config.max_order_value_pct_of_cash
            notional = price * qty
            if notional > max_notional:
                return RiskDecision(allowed=False, reason=f"order notional {notional:.2f} > max {max_notional:.2f}", ts_code=ts_code, side=side, price=price, qty=qty, dt=trade_dt)
        self._daily_order_count[trade_date] = cnt + 1
        return RiskDecision(allowed=True, reason="ok", ts_code=ts_code, side=side, price=price, qty=qty, dt=trade_dt)


if __name__ == "__main__":
    class DummyCtx:
        def __init__(self, cash: float) -> None:
            self._cash = cash
            self._pos: Dict[str, float] = {}
        def get_cash(self) -> float:
            return self._cash
        def get_position(self, ts_code: str) -> float:
            return self._pos.get(ts_code, 0.0)
        def log(self, msg: str) -> None:
            print("[DummyCtx]", msg)
    from datetime import datetime as _dt
    cfg = RiskConfig(max_order_value_pct_of_cash=0.1, max_orders_per_day=3, banned_ts_codes=["000003.SZ"]) 
    rm = RiskManager(cfg)
    ctx = DummyCtx(cash=100_000)
    dt = _dt.now()
    for i in range(5):
        dec = rm.check_order("000001.SZ", "buy", price=10.0, qty=1000, trade_dt=dt, ctx=ctx)
        print(f"[selftest] i={i}, decision=", dec)
    dec2 = rm.check_order("000003.SZ", "buy", 10.0, 1000, trade_dt=dt, ctx=ctx)
    print("[selftest] banned ts decision=", dec2)

