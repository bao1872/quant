from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Protocol

import pandas as pd

from data.tick_store import TickStore
from strategy.abu_level_strategy import AbuKeyLevelStrategy
from strategy.base import StrategyContext
from strategy.risk_manager import RiskManager
from live.broker_base import BrokerBase, DummyBroker


class HasSendOrderContext(StrategyContext, Protocol):
    def send_order(self, ts_code: str, side: str, price: float, qty: int, reason: str, meta: Dict[str, Any]) -> None:
        ...


@dataclass
class LiveOrderLog:
    ts_code: str
    dt: datetime
    side: str
    price: float
    qty: int
    reason: str
    meta: Dict[str, Any]
    broker_order_id: Optional[str]


class LiveStrategyContext:
    def __init__(self, broker: BrokerBase, risk_manager: Optional[RiskManager] = None, logger: Optional[Any] = None) -> None:
        self.broker = broker
        self.risk_manager = risk_manager or RiskManager()
        self.logger = logger or (lambda msg: print(f"[LiveContext] {msg}"))
        self._vars: Dict[str, Any] = {}
        self._orders: List[LiveOrderLog] = []
        info = self.broker.get_account_info()
        self.cash: float = float(info.get("cash", 0.0))
        self.equity: float = float(info.get("equity", self.cash))

    def get_position(self, ts_code: str) -> float:
        pos_list = self.broker.get_positions()
        qty = 0.0
        for pos in pos_list:
            if pos.get("ts_code") == ts_code:
                qty += float(pos.get("qty", 0.0))
        return qty

    def get_cash(self) -> float:
        return float(self.cash)

    def log(self, msg: str) -> None:
        self.logger(msg)

    def set_var(self, key: str, value: Any) -> None:
        self._vars[key] = value

    def get_var(self, key: str, default: Any = None) -> Any:
        return self._vars.get(key, default)

    def get_config(self) -> Dict[str, Any]:
        return {}

    def send_order(self, ts_code: str, side: str, price: float, qty: int, reason: str, meta: Dict[str, Any]) -> None:
        dt: datetime = meta.get("dt") or datetime.now()
        side = side.lower()
        decision = self.risk_manager.check_order(ts_code=ts_code, side=side, price=float(price), qty=int(qty), trade_dt=dt, ctx=self)
        if not decision.allowed:
            self.log(f"[Risk] reject order: {decision.reason}")
            return
        order_id = self.broker.place_order(ts_code=ts_code, side=side, price=float(price), qty=int(qty), order_type="limit", remark=reason)
        info = self.broker.get_account_info()
        self.cash = float(info.get("cash", 0.0))
        self.equity = float(info.get("equity", self.cash))
        self._orders.append(LiveOrderLog(ts_code=ts_code, dt=dt, side=side, price=float(price), qty=int(qty), reason=reason, meta=meta, broker_order_id=order_id))
        self.log(f"[Order] ts={ts_code} side={side} price={price} qty={qty} reason={reason} broker_order_id={order_id}")

    def orders_to_dataframe(self) -> pd.DataFrame:
        if not self._orders:
            return pd.DataFrame(columns=["ts_code", "dt", "side", "price", "qty", "reason", "broker_order_id"])
        rows = []
        for od in self._orders:
            rows.append({"ts_code": od.ts_code, "dt": od.dt, "side": od.side, "price": od.price, "qty": od.qty, "reason": od.reason, "broker_order_id": od.broker_order_id})
        return pd.DataFrame(rows)


class TickSource(Protocol):
    def iter_ticks(self, ts_code: str, trade_dates: Iterable[date]) -> Iterable[Dict[str, Any]]:
        ...


class TickStoreSource:
    def __init__(self, store: Optional[TickStore] = None) -> None:
        self.store = store or TickStore()

    def iter_ticks(self, ts_code: str, trade_dates: Iterable[date]) -> Iterable[Dict[str, Any]]:
        for d in trade_dates:
            df = self.store.load_ticks(ts_code, d)
            if df is None or df.empty:
                continue
            df = df.sort_values("datetime").reset_index(drop=True)
            for _, row in df.iterrows():
                dt = row["datetime"]
                if not isinstance(dt, datetime):
                    dt = pd.to_datetime(dt).to_pydatetime()
                yield {"ts_code": ts_code, "datetime": dt, "price": float(row["price"]), "volume": int(row.get("volume", 0)), "side": row.get("side", "N")}


class LiveEngine:
    def __init__(self, strategy: AbuKeyLevelStrategy, broker: Optional[BrokerBase] = None, risk_manager: Optional[RiskManager] = None, tick_source: Optional[TickSource] = None) -> None:
        self.strategy = strategy
        self.broker = broker or DummyBroker(initial_cash=1_000_000.0)
        self.risk_manager = risk_manager or RiskManager()
        self.tick_source = tick_source or TickStoreSource()
        self.ctx = LiveStrategyContext(broker=self.broker, risk_manager=self.risk_manager)

    def run_replay(self, ts_code: str, trade_dates: Iterable[date]) -> pd.DataFrame:
        for tick in self.tick_source.iter_ticks(ts_code, trade_dates):
            self.strategy.on_tick(tick, self.ctx)
        return self.ctx.orders_to_dataframe()


if __name__ == "__main__":
    from datetime import timedelta
    ts_code = "000001.SZ"
    today = date.today()
    trade_dates = [today - timedelta(days=1)]
    from factors import AbuPriceLevelProvider
    from microstructure import AbuMicrostructureAnalyzer
    pl_provider = AbuPriceLevelProvider()
    micro = AbuMicrostructureAnalyzer()
    strat = AbuKeyLevelStrategy(ts_code=ts_code, price_level_provider=pl_provider, micro_analyzer=micro, config=None)
    engine = LiveEngine(strategy=strat)
    df_orders = engine.run_replay(ts_code, trade_dates)
    print("[selftest] orders from replay:")
    print(df_orders)
