from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import threading
import uuid


@dataclass
class BrokerOrder:
    order_id: str
    ts_code: str
    side: str
    price: float
    qty: int
    status: str
    remark: str
    created_at: datetime
    filled_at: Optional[datetime] = None


class BrokerBase(ABC):
    @abstractmethod
    def place_order(self, ts_code: str, side: str, price: float, qty: int, order_type: str = "limit", remark: str = "") -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_orders(self) -> List[BrokerOrder]:
        raise NotImplementedError


class DummyBroker(BrokerBase):
    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self._cash = float(initial_cash)
        self._equity = float(initial_cash)
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._orders: List[BrokerOrder] = []
        self._lock = threading.Lock()

    def place_order(self, ts_code: str, side: str, price: float, qty: int, order_type: str = "limit", remark: str = "") -> Optional[str]:
        dt = datetime.now()
        side = side.lower()
        qty = int(qty)
        price = float(price)
        if qty <= 0 or price <= 0:
            print(f"[DummyBroker] invalid order qty/price: {qty}, {price}")
            return None
        order_id = uuid.uuid4().hex
        order = BrokerOrder(order_id=order_id, ts_code=ts_code, side=side, price=price, qty=qty, status="new", remark=remark, created_at=dt)
        with self._lock:
            if side == "buy":
                notional = price * qty
                if notional > self._cash:
                    order.status = "rejected"
                    order.remark += " | insufficient cash"
                else:
                    self._cash -= notional
                    pos = self._positions.get(ts_code, {"qty": 0, "cost": 0.0})
                    new_qty = pos["qty"] + qty
                    if new_qty > 0:
                        pos["cost"] = (pos["cost"] * pos["qty"] + notional) / new_qty
                    pos["qty"] = new_qty
                    self._positions[ts_code] = pos
                    order.status = "filled"
                    order.filled_at = dt
            elif side == "sell":
                pos = self._positions.get(ts_code, {"qty": 0, "cost": 0.0})
                if qty > pos["qty"]:
                    order.status = "rejected"
                    order.remark += " | insufficient position"
                else:
                    proceeds = price * qty
                    self._cash += proceeds
                    pos["qty"] -= qty
                    if pos["qty"] <= 0:
                        self._positions.pop(ts_code, None)
                    else:
                        self._positions[ts_code] = pos
                    order.status = "filled"
                    order.filled_at = dt
            else:
                order.status = "rejected"
                order.remark += " | unknown side"
            self._orders.append(order)
            self._equity = self._cash
        print(f"[DummyBroker] order: {order}")
        return order_id if order.status != "rejected" else None

    def cancel_order(self, order_id: str) -> None:
        with self._lock:
            for od in self._orders:
                if od.order_id == order_id and od.status == "new":
                    od.status = "canceled"
                    print(f"[DummyBroker] cancel_order: {order_id}")
                    break

    def get_account_info(self) -> Dict[str, Any]:
        with self._lock:
            return {"cash": self._cash, "equity": self._equity}

    def get_positions(self) -> List[Dict[str, Any]]:
        with self._lock:
            res: List[Dict[str, Any]] = []
            for ts_code, pos in self._positions.items():
                res.append({"ts_code": ts_code, "qty": pos["qty"], "cost": pos["cost"]})
            return res

    def list_orders(self) -> List[BrokerOrder]:
        with self._lock:
            return list(self._orders)


if __name__ == "__main__":
    broker = DummyBroker(initial_cash=100_000)
    print("[selftest] init account:", broker.get_account_info())
    oid1 = broker.place_order("000001.SZ", "buy", price=10.0, qty=1000, remark="test_buy")
    oid2 = broker.place_order("000001.SZ", "sell", price=10.5, qty=500, remark="test_sell")
    oid3 = broker.place_order("000002.SZ", "sell", price=20.0, qty=100, remark="invalid_sell")
    print("[selftest] orders:")
    for od in broker.list_orders():
        print(od)
    print("[selftest] positions:", broker.get_positions())
    print("[selftest] account:", broker.get_account_info())

