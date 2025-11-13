from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from data.tick_store import TickStore
from strategy.abu_level_strategy import AbuKeyLevelStrategy
from strategy.base import PriceLevelProvider
from factors import AbuPriceLevelProvider


@dataclass
class TradeRecord:
    ts_code: str
    open_dt: pd.Timestamp
    close_dt: pd.Timestamp
    side: str
    qty: int
    entry_price: float
    exit_price: float
    pnl: float
    reason: str


@dataclass
class BacktestResult:
    ts_code: str
    start_date: date
    end_date: date
    initial_cash: float
    final_equity: float
    trades: List[TradeRecord]

    @property
    def total_pnl(self) -> float:
        return self.final_equity - self.initial_cash

    @property
    def trade_count(self) -> int:
        return len(self.trades)


class BacktestContext:
    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.equity = float(initial_cash)
        self.position_side: Optional[str] = None
        self.position_qty: int = 0
        self.position_price: float = 0.0
        self.position_open_dt: Optional[pd.Timestamp] = None
        self.trades: List[TradeRecord] = []

    def send_order(self, ts_code: str, side: str, price: float, qty: int, reason: str, meta: Dict[str, Any]) -> None:
        price = float(price)
        qty = int(qty)
        dt = meta.get("dt", pd.Timestamp.now())
        if side == "buy":
            if self.position_side is None:
                cost = price * qty
                if cost > self.cash:
                    return
                self.cash -= cost
                self.position_side = "long"
                self.position_qty = qty
                self.position_price = price
                self.position_open_dt = pd.Timestamp(dt)
            elif self.position_side == "short":
                pnl = (self.position_price - price) * self.position_qty
                self.cash += self.position_qty * self.position_price + pnl
                self.trades.append(TradeRecord(ts_code=ts_code, open_dt=self.position_open_dt or pd.Timestamp(dt), close_dt=pd.Timestamp(dt), side="short", qty=self.position_qty, entry_price=self.position_price, exit_price=price, pnl=pnl, reason=reason))
                self.position_side = None
                self.position_qty = 0
                self.position_price = 0.0
                self.position_open_dt = None
        elif side == "sell":
            if self.position_side is None:
                proceeds = price * qty
                self.cash += proceeds
                self.position_side = "short"
                self.position_qty = qty
                self.position_price = price
                self.position_open_dt = pd.Timestamp(dt)
            elif self.position_side == "long":
                pnl = (price - self.position_price) * self.position_qty
                self.cash += self.position_qty * self.position_price + pnl
                self.trades.append(TradeRecord(ts_code=ts_code, open_dt=self.position_open_dt or pd.Timestamp(dt), close_dt=pd.Timestamp(dt), side="long", qty=self.position_qty, entry_price=self.position_price, exit_price=price, pnl=pnl, reason=reason))
                self.position_side = None
                self.position_qty = 0
                self.position_price = 0.0
                self.position_open_dt = None
        self._update_equity(current_price=price)

    def _update_equity(self, current_price: float) -> None:
        pos_value = 0.0
        if self.position_side is not None and self.position_qty > 0:
            if self.position_side == "long":
                pos_value = current_price * self.position_qty
            else:
                pos_value = self.position_price * self.position_qty
        self.equity = self.cash + pos_value


class TickBacktester:
    def __init__(self, ts_code: str, price_level_provider: Optional[PriceLevelProvider] = None, initial_cash: float = 100_000.0) -> None:
        self.ts_code = ts_code
        self.price_level_provider = price_level_provider or AbuPriceLevelProvider()
        self.initial_cash = float(initial_cash)
        self.tick_store = TickStore()

    def run(self, trade_dates: List[date], strategy_config: Optional[Dict[str, Any]] = None) -> BacktestResult:
        ctx = BacktestContext(initial_cash=self.initial_cash)
        strat = AbuKeyLevelStrategy(ts_code=self.ts_code, price_level_provider=self.price_level_provider, config=strategy_config)
        first_date = trade_dates[0]
        last_date = trade_dates[-1]
        for d in trade_dates:
            df_ticks = self.tick_store.load_ticks(self.ts_code, d)
            if df_ticks is None or df_ticks.empty:
                continue
            df_ticks = df_ticks.sort_values("datetime").reset_index(drop=True)
            for _, row in df_ticks.iterrows():
                tick = {"ts_code": self.ts_code, "datetime": row["datetime"], "price": float(row.get("price", 0.0)), "volume": int(row.get("volume", 0)), "side": row.get("side", "N")}
                strat.on_tick(tick, ctx)
        final_equity = ctx.equity
        trades = ctx.trades
        return BacktestResult(ts_code=self.ts_code, start_date=first_date, end_date=last_date, initial_cash=self.initial_cash, final_equity=final_equity, trades=trades)


if __name__ == "__main__":
    from datetime import timedelta
    ts = "000001.SZ"
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(3)]
    dates = sorted(dates)
    bt = TickBacktester(ts_code=ts, initial_cash=100_000.0)
    result = bt.run(dates, strategy_config={"min_signal_score": 60.0})
    print("Backtest result:")
    print("Total trades:", result.trade_count)
    print("Total PnL:", result.total_pnl)
    print("Final equity:", result.final_equity)
    for tr in result.trades:
        print(tr)

