from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import threading
import uuid


@dataclass
class QmtOrder:
    order_id: str
    ts_code: str
    side: str
    price: float
    qty: int
    status: str
    remark: str


class QmtClient:
    def __init__(
        self,
        account_id: Optional[str] = None,
        dry_run: bool = True,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.account_id = account_id or "default"
        self.dry_run = dry_run
        self.extra_config = extra_config or {}
        self._connected = False
        self._lock = threading.Lock()
        self._orders: List[QmtOrder] = []

        self._cash = float(self.extra_config.get("initial_cash", 1_000_000.0))
        self._equity = self._cash
        self._positions: Dict[str, Dict[str, Any]] = {}

    def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        print(f"[QmtClient] connected (dry_run={self.dry_run})")

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        print("[QmtClient] disconnected")

    def get_account_info(self) -> Dict[str, Any]:
        if not self._connected:
            raise RuntimeError("QmtClient not connected")
        return {
            "account_id": self.account_id,
            "cash": self._cash,
            "equity": self._equity,
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("QmtClient not connected")
        res: List[Dict[str, Any]] = []
        for ts_code, pos in self._positions.items():
            res.append({"ts_code": ts_code, "qty": pos["qty"], "cost": pos["cost"]})
        return res

    def place_order(
        self,
        ts_code: str,
        side: str,
        price: float,
        qty: int,
        order_type: str = "limit",
        remark: str = "",
    ) -> str:
        if not self._connected:
            raise RuntimeError("QmtClient not connected")
        price = float(price)
        qty = int(qty)
        order_id = uuid.uuid4().hex
        with self._lock:
            order = QmtOrder(order_id=order_id, ts_code=ts_code, side=side, price=price, qty=qty, status="new", remark=remark)
            self._orders.append(order)
            self._simulate_fill(order)
        print(f"[QmtClient] order placed: {order}")
        return order_id

    def _simulate_fill(self, order: QmtOrder) -> None:
        if order.status == "filled":
            return
        ts_code = order.ts_code
        price = order.price
        qty = order.qty
        if order.side == "buy":
            cost = price * qty
            if cost > self._cash:
                order.status = "rejected"
                order.remark += " | insufficient cash"
                return
            self._cash -= cost
            pos = self._positions.get(ts_code, {"qty": 0, "cost": 0.0})
            total_qty = pos["qty"] + qty
            if total_qty > 0:
                pos["cost"] = (pos["cost"] * pos["qty"] + cost) / total_qty
            pos["qty"] = total_qty
            self._positions[ts_code] = pos
            order.status = "filled"
        elif order.side == "sell":
            pos = self._positions.get(ts_code, {"qty": 0, "cost": 0.0})
            if qty > pos["qty"]:
                order.status = "rejected"
                order.remark += " | insufficient position"
                return
            proceeds = price * qty
            self._cash += proceeds
            pos["qty"] -= qty
            if pos["qty"] <= 0:
                self._positions.pop(ts_code, None)
            else:
                self._positions[ts_code] = pos
            order.status = "filled"
        self._equity = self._cash

    def get_orders(self) -> List[QmtOrder]:
        with self._lock:
            return list(self._orders)

if __name__ == "__main__":
    client = QmtClient(dry_run=True, extra_config={"initial_cash": 100_000})
    client.connect()
    print("Account:", client.get_account_info())
    oid1 = client.place_order("000001.SZ", "buy", 10.0, 1000, remark="test_buy")
    oid2 = client.place_order("000001.SZ", "sell", 10.5, 500, remark="test_sell")
    print("Orders:", client.get_orders())
    print("Positions:", client.get_positions())
    print("Account after trades:", client.get_account_info())
    client.disconnect()
