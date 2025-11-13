from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import math
import pandas as pd

from strategy.base import PriceLevel, PriceLevelProvider
from microstructure import AbuMicrostructureAnalyzer, MicroSignal


DEFAULT_STRAT_CONFIG: Dict[str, Any] = {
    "risk_per_trade_pct": 0.01,
    "max_position_pct": 0.3,
    "stop_loss_pct": 0.005,
    "take_profit_R": 2.0,
    "level_tolerance_pct": 0.002,
    "micro_window_seconds": 60,
    "min_signal_score": 55.0,
    "lot_size": 100,
}


@dataclass
class PositionState:
    ts_code: str
    side: str
    qty: int
    entry_price: float
    entry_dt: datetime
    entry_level_price: float
    stop_loss: float
    take_profit: float
    meta: Dict[str, Any]


class AbuKeyLevelStrategy:
    def __init__(self, ts_code: str, price_level_provider: PriceLevelProvider, micro_analyzer: Optional[AbuMicrostructureAnalyzer] = None, config: Optional[Dict[str, Any]] = None, name: str = "abu_key_level") -> None:
        self.name = name
        self.ts_code = ts_code
        self.price_level_provider = price_level_provider
        self.micro_analyzer = micro_analyzer or AbuMicrostructureAnalyzer()
        self.config = DEFAULT_STRAT_CONFIG.copy()
        if config:
            self.config.update(config)
        self._current_trade_date: Optional[date] = None
        self._daily_levels: List[PriceLevel] = []
        self._position: Optional[PositionState] = None
        self._tick_buffer: List[Dict[str, Any]] = []

    def on_tick(self, tick: Dict[str, Any], ctx: Any) -> None:
        dt: datetime = tick["datetime"]
        trade_date = dt.date()
        price = float(tick["price"])
        volume = int(tick.get("volume", 0))
        side_flag = tick.get("side", "N")
        if self._current_trade_date != trade_date:
            self._on_new_day(trade_date)
        ts_code = tick.get("ts_code", self.ts_code)
        if ts_code != self.ts_code:
            return
        self._append_tick(dt, price, volume, side_flag)
        if self._position is not None:
            self._handle_existing_position(price=price, dt=dt, ctx=ctx)
        if self._position is None:
            self._maybe_open_position(price=price, dt=dt, ctx=ctx)

    def _on_new_day(self, trade_date: date) -> None:
        self._current_trade_date = trade_date
        self._tick_buffer = []
        self._position = None
        self._daily_levels = self.price_level_provider.get_levels(self.ts_code, trade_date)

    def _append_tick(self, dt: datetime, price: float, volume: int, side: str) -> None:
        self._tick_buffer.append({"datetime": dt, "price": price, "volume": volume, "side": side})
        window_seconds = int(self.config["micro_window_seconds"])
        cutoff = dt - timedelta(seconds=window_seconds * 2)
        self._tick_buffer = [t for t in self._tick_buffer if t["datetime"] >= cutoff]

    def _get_tick_window(self, dt: datetime) -> pd.DataFrame:
        window_seconds = int(self.config["micro_window_seconds"])
        start = dt - timedelta(seconds=window_seconds)
        rows = [t for t in self._tick_buffer if start <= t["datetime"] <= dt]
        if not rows:
            return pd.DataFrame(columns=["datetime", "price", "volume", "side"])
        return pd.DataFrame(rows)

    def _handle_existing_position(self, price: float, dt: datetime, ctx: Any) -> None:
        if self._position is None:
            return
        pos = self._position
        stop = pos.stop_loss
        tp = pos.take_profit
        if pos.side == "long":
            if price <= stop:
                self._close_position(price, dt, ctx, reason="stop_loss")
                return
            if price >= tp:
                self._close_position(price, dt, ctx, reason="take_profit")
                return
        elif pos.side == "short":
            if price >= stop:
                self._close_position(price, dt, ctx, reason="stop_loss")
                return
            if price <= tp:
                self._close_position(price, dt, ctx, reason="take_profit")
                return

    def _close_position(self, price: float, dt: datetime, ctx: Any, reason: str) -> None:
        if self._position is None:
            return
        pos = self._position
        side = "sell" if pos.side == "long" else "buy"
        ctx.send_order(ts_code=pos.ts_code, side=side, price=price, qty=pos.qty, reason=f"exit_{reason}", meta={"entry_price": pos.entry_price, "dt": dt})
        self._position = None

    def _maybe_open_position(self, price: float, dt: datetime, ctx: Any) -> None:
        if not self._daily_levels:
            return
        tol_pct = float(self.config["level_tolerance_pct"])
        min_score = float(self.config["min_signal_score"])
        candidate_levels: List[PriceLevel] = [lv for lv in self._daily_levels if lv.direction in ("support", "resistance")]
        if not candidate_levels:
            return
        candidate_levels.sort(key=lambda lv: abs(price - float(lv.level_price)) / max(float(lv.level_price), 1e-6))
        best_level = candidate_levels[0]
        level_price = float(best_level.level_price)
        dist_pct = abs(price - level_price) / max(level_price, 1e-6)
        if dist_pct > tol_pct:
            return
        ticks_win = self._get_tick_window(dt)
        if ticks_win.empty:
            return
        sig = self.micro_analyzer.analyze_near_level(ts_code=self.ts_code, trade_date=self._current_trade_date, ticks=ticks_win, level=best_level, now_dt=dt)
        if sig is None or sig.score < min_score:
            return
        if best_level.direction == "support" and sig.side != "long":
            return
        if best_level.direction == "resistance" and sig.side != "short":
            return
        self._open_position(price=price, dt=dt, level=best_level, micro_sig=sig, ctx=ctx)

    def _open_position(self, price: float, dt: datetime, level: PriceLevel, micro_sig: MicroSignal, ctx: Any) -> None:
        if self._position is not None:
            return
        cfg = self.config
        risk_pct = float(cfg["risk_per_trade_pct"])
        max_pos_pct = float(cfg["max_position_pct"])
        stop_pct = float(cfg["stop_loss_pct"])
        take_R = float(cfg["take_profit_R"])
        lot_size = int(cfg["lot_size"])
        direction = "long" if micro_sig.side == "long" else "short"
        if direction == "long":
            stop_price = price * (1.0 - stop_pct)
            tp_price = price + (price - stop_price) * take_R
            risk_per_share = price - stop_price
        else:
            stop_price = price * (1.0 + stop_pct)
            tp_price = price - (stop_price - price) * take_R
            risk_per_share = stop_price - price
        if risk_per_share <= 1e-6:
            return
        equity = float(getattr(ctx, "equity", getattr(ctx, "cash", 0.0)))
        max_risk_amount = equity * risk_pct
        raw_qty = max_risk_amount / risk_per_share
        lots = int(raw_qty // lot_size)
        if lots <= 0:
            return
        qty = lots * lot_size
        max_value = equity * max_pos_pct
        if qty * price > max_value:
            qty = int(max_value // (price * lot_size)) * lot_size
        if qty <= 0:
            return
        order_side = "buy" if direction == "long" else "sell"
        ctx.send_order(ts_code=self.ts_code, side=order_side, price=price, qty=qty, reason=f"entry_level_{level.direction}", meta={"level_price": level.level_price, "level_type": level.level_type, "micro_score": micro_sig.score, "micro_reason": micro_sig.reason, "dt": dt})
        self._position = PositionState(ts_code=self.ts_code, side=direction, qty=qty, entry_price=price, entry_dt=dt, entry_level_price=float(level.level_price), stop_loss=float(stop_price), take_profit=float(tp_price), meta={"level_type": level.level_type, "level_direction": level.direction, "micro_score": micro_sig.score})


if __name__ == "__main__":
    class DummyLevelProvider(PriceLevelProvider):
        def precompute(self, trade_date: date) -> None:
            pass
        def get_levels(self, ts_code: str, trade_date: date) -> List[PriceLevel]:
            return [PriceLevel(ts_code=ts_code, trade_date=trade_date, level_price=10.0, level_type="range_low", direction="support", strength=80, source_flags=["range"], meta={})]
    class DummyCtx:
        def __init__(self) -> None:
            self.cash = 100_000.0
            self.equity = 100_000.0
            self.orders: List[Dict[str, Any]] = []
        def send_order(self, ts_code: str, side: str, price: float, qty: int, reason: str, meta: Dict[str, Any]) -> None:
            print("ORDER:", ts_code, side, price, qty, reason, meta)
            self.orders.append({"ts_code": ts_code, "side": side, "price": price, "qty": qty, "reason": reason, "meta": meta})
    from datetime import timedelta
    prov = DummyLevelProvider()
    strat = AbuKeyLevelStrategy(ts_code="000001.SZ", price_level_provider=prov, config={"micro_window_seconds": 60, "min_signal_score": 0})
    ctx = DummyCtx()
    now = datetime.now()
    for i in range(60):
        dt = now - timedelta(seconds=60 - i)
        tick = {"ts_code": "000001.SZ", "datetime": dt, "price": 10.0 + (i - 50) * 0.001, "volume": 200, "side": "B" if i > 30 else "S"}
        strat.on_tick(tick, ctx)
    print("Total orders:", len(ctx.orders))

