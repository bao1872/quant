from __future__ import annotations

from typing import Any, Dict, List, Optional

from .broker_base import BrokerBase, BrokerOrder, DummyBroker


class QmtBroker(BrokerBase):
    def __init__(self, dry_run: bool = True, config: Optional[Dict[str, Any]] = None) -> None:
        self.dry_run = dry_run
        self.config = config or {}
        self._dummy = DummyBroker(initial_cash=float(self.config.get("initial_cash", 1_000_000.0)))
        self._api = None
        if not self.dry_run:
            raise NotImplementedError("QmtBroker non-dry_run is not implemented")

    def place_order(self, ts_code: str, side: str, price: float, qty: int, order_type: str = "limit", remark: str = "") -> Optional[str]:
        print(
            f"[QmtBroker][dry_run] place_order ts_code={ts_code} side={side} price={price} qty={qty} order_type={order_type} remark={remark}"
        )
        return self._dummy.place_order(ts_code=ts_code, side=side, price=float(price), qty=int(qty), order_type=order_type, remark=remark)

    def cancel_order(self, order_id: str) -> None:
        print(f"[QmtBroker][dry_run] cancel_order order_id={order_id}")
        self._dummy.cancel_order(order_id)

    def get_account_info(self) -> Dict[str, Any]:
        return self._dummy.get_account_info()

    def get_positions(self) -> List[Dict[str, Any]]:
        return self._dummy.get_positions()

    def list_orders(self) -> List[BrokerOrder]:
        return self._dummy.list_orders()


if __name__ == "__main__":
    broker = QmtBroker(dry_run=True, config={"initial_cash": 100_000})
    print("[selftest] account:", broker.get_account_info())
    oid1 = broker.place_order("000001.SZ", "buy", 10.0, 1000, remark="test_buy")
    oid2 = broker.place_order("000001.SZ", "sell", 10.5, 500, remark="test_sell")
    print("[selftest] positions:", broker.get_positions())
    print("[selftest] orders:", broker.list_orders())

