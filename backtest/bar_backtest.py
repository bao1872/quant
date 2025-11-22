from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_index: int
    exit_index: int
    side: str
    entry_price: float
    exit_price: float
    qty: int
    stop_price: Optional[float] = None
    target_price: Optional[float] = None

    @property
    def pnl(self) -> float:
        d = (self.exit_price - self.entry_price) if self.side == "long" else (self.entry_price - self.exit_price)
        return d * float(self.qty)

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0 or self.qty <= 0:
            return 0.0
        cost = self.entry_price * float(self.qty)
        return self.pnl / cost


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    signal: pd.Series


def run_backtest_one_unit(
    bars: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 100_000.0,
    fee_rate: float = 0.0,
    slippage: float = 0.0,
) -> BacktestResult:
    if "close" not in bars.columns:
        raise ValueError("bars 缺少 close")
    close = pd.to_numeric(bars["close"], errors="coerce").fillna(0.0).values
    sig_series = signal.reindex(bars.index).fillna(0.0).astype(float)
    sig = sig_series.values
    n = len(close)
    pos = 0
    trades: List[Trade] = []
    equity = np.zeros(n, dtype=float)
    capital = float(initial_capital)
    entry_price: Optional[float] = None
    entry_idx: Optional[int] = None
    entry_side: Optional[str] = None
    qty: int = 0
    for i in range(n):
        target = int(np.sign(sig[i]))
        price = float(close[i])
        if target != pos:
            if pos != 0 and entry_price is not None and entry_idx is not None and entry_side is not None and qty > 0:
                exit_px_raw = price
                exit_px = exit_px_raw * (1.0 - slippage) if entry_side == "long" else exit_px_raw * (1.0 + slippage)
                t = Trade(entry_index=entry_idx, exit_index=i, side=entry_side, entry_price=float(entry_price), exit_price=float(exit_px), qty=int(qty))
                fee = fee_rate * ((abs(entry_price) * qty) + (abs(exit_px) * qty))
                capital += t.pnl - fee
                trades.append(t)
                qty = 0
            if target != 0:
                pos = target
                if target > 0:
                    entry_px = price * (1.0 + slippage)
                    entry_side = "long"
                else:
                    entry_px = price * (1.0 - slippage)
                    entry_side = "short"
                entry_price = float(entry_px)
                entry_idx = i
                qty = int(capital // entry_price // 100 * 100)
                if qty <= 0:
                    qty = 0
            else:
                pos = 0
                entry_price = None
                entry_idx = None
                entry_side = None
                qty = 0
        if pos == 0 or entry_price is None or qty <= 0:
            equity[i] = capital
        else:
            floating = ((price - entry_price) * qty) if pos > 0 else ((entry_price - price) * qty)
            equity[i] = capital + floating
    if pos != 0 and entry_price is not None and entry_idx is not None and entry_side is not None and qty > 0:
        exit_px_raw = close[-1]
        exit_px = exit_px_raw * (1.0 - slippage) if entry_side == "long" else exit_px_raw * (1.0 + slippage)
        t = Trade(entry_index=entry_idx, exit_index=n - 1, side=entry_side, entry_price=float(entry_price), exit_price=float(exit_px), qty=int(qty))
        fee = fee_rate * ((abs(entry_price) * qty) + (abs(exit_px) * qty))
        capital += t.pnl - fee
        trades.append(t)
        equity[-1] = capital
    equity_curve = pd.Series(equity, index=bars.index, name="equity")
    return BacktestResult(trades=trades, equity_curve=equity_curve, signal=sig_series)


if __name__ == "__main__":
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({"close": np.linspace(10, 12, 20)}, index=idx)
    sig = pd.Series([0]*5 + [1]*10 + [0]*5, index=idx)
    res = run_backtest_one_unit(df, sig, fee_rate=0.0005, slippage=0.0005)
    print(len(res.trades), float(res.equity_curve.iloc[-1]))