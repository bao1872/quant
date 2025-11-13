from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .qmt_client import QmtClient


@dataclass
class LiveTradeRecord:
    ts_code: str
    dt: datetime
    side: str
    price: float
    qty: int
    reason: str
    meta: Dict[str, Any]
    order_id: str
    status: str


class LiveContext:
    def __init__(self, client: QmtClient, ts_code: Optional[str] = None, max_slippage_pct: float = 0.002) -> None:
        self.client = client
        self.ts_code = ts_code
        self.max_slippage_pct = float(max_slippage_pct)
        self.cash: float = 0.0
        self.equity: float = 0.0
        self.trades: List[LiveTradeRecord] = []
        self.refresh_account()

    def refresh_account(self) -> None:
        info = self.client.get_account_info()
        self.cash = float(info.get("cash", 0.0))
        self.equity = float(info.get("equity", self.cash))

    def send_order(self, ts_code: str, side: str, price: float, qty: int, reason: str, meta: Dict[str, Any]) -> None:
        if self.ts_code is not None and ts_code != self.ts_code:
            print(f"[LiveContext] ignore order for {ts_code}, context bound to {self.ts_code}")
            return
        qty = int(qty)
        price = float(price)
        dt = meta.get("dt") or datetime.now()
        remark = f"{reason}"
        order_id = self.client.place_order(ts_code=ts_code, side=side, price=price, qty=qty, order_type="limit", remark=remark)
        self.refresh_account()
        status = "unknown"
        for od in self.client.get_orders():
            if od.order_id == order_id:
                status = od.status
                break
        rec = LiveTradeRecord(ts_code=ts_code, dt=dt, side=side, price=price, qty=qty, reason=reason, meta=meta, order_id=order_id, status=status)
        self.trades.append(rec)
        print(f"[LiveContext] order sent: {rec}")

    def trades_to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=["ts_code", "dt", "side", "price", "qty", "reason", "order_id", "status"])
        rows = []
        for t in self.trades:
            rows.append({"ts_code": t.ts_code, "dt": t.dt, "side": t.side, "price": t.price, "qty": t.qty, "reason": t.reason, "order_id": t.order_id, "status": t.status})
        return pd.DataFrame(rows)

if __name__ == "__main__":
    client = QmtClient(dry_run=True, extra_config={"initial_cash": 100_000})
    client.connect()
    ctx = LiveContext(client=client, ts_code="000001.SZ")
    print("Initial account:", ctx.cash, ctx.equity)
    ctx.send_order(ts_code="000001.SZ", side="buy", price=10.0, qty=1000, reason="test_entry", meta={})
    ctx.send_order(ts_code="000001.SZ", side="sell", price=10.5, qty=1000, reason="test_exit", meta={})
    print("Account after:", ctx.cash, ctx.equity)
    print(ctx.trades_to_dataframe())
    client.disconnect()

