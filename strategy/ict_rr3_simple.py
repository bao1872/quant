from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def mark_bullish_reversal(df: pd.DataFrame) -> pd.Series:
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    body = (close - open_).abs()
    candle_range = (high - low).replace(0, np.nan)
    body_ratio = body / candle_range
    low_shadow = (np.minimum(open_, close) - low)
    high_shadow = (high - np.maximum(open_, close))
    is_bull = close > open_
    cond_body = body_ratio >= 0.2
    cond_low_shadow = (low_shadow >= (0.4 * candle_range)) & (low_shadow >= body)
    cond_high_shadow = high_shadow <= (0.5 * candle_range)
    flag = (is_bull & cond_body & cond_low_shadow & cond_high_shadow).astype(int)
    return pd.Series(flag.values, index=df.index, name="bull_reversal_flag")


@dataclass
class EntrySignal:
    index: int
    entry_price: float
    stop_price: float
    target_price: float
    rr: float


def generate_rr3_long_signals(
    df_15m: pd.DataFrame,
    target_price: float,
    rr_min: float = 3.0,
    ob_max_count: int = 5,
) -> list[EntrySignal]:
    df = df_15m.copy()
    df["bull_reversal_flag"] = mark_bullish_reversal(df)
    ob_mask = pd.to_numeric(df.get("ict_ob_flag", pd.Series(0, index=df.index)), errors="coerce").fillna(0) > 0
    ob_indices = list(df.index[ob_mask])[-ob_max_count:]
    ob_zones: list[tuple[float, float]] = []
    if ob_indices:
        ob_top = pd.to_numeric(df.get("ict_ob_top", pd.Series(index=df.index)), errors="coerce")
        ob_bottom = pd.to_numeric(df.get("ict_ob_bottom", pd.Series(index=df.index)), errors="coerce")
        for i in ob_indices:
            top = float(ob_top.at[i]) if pd.notna(ob_top.at[i]) else np.nan
            bottom = float(ob_bottom.at[i]) if pd.notna(ob_bottom.at[i]) else np.nan
            if not np.isnan(top) and not np.isnan(bottom):
                ob_zones.append((bottom, top))
    signals: list[EntrySignal] = []
    if not ob_zones:
        return signals
    low = pd.to_numeric(df["low"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    choch = pd.to_numeric(df.get("ict_choch_flag", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    tp = float(target_price)
    for i in df.index:
        px_low = float(low.at[i])
        px_high = float(high.at[i])
        px_close = float(close.at[i])
        hit_ob = None
        for (b, t) in ob_zones:
            if px_low <= t and px_high >= b:
                hit_ob = (b, t)
                break
        if hit_ob is None:
            continue
        is_bull_choch = float(choch.at[i]) > 0
        is_bull_rev = int(df.at[i, "bull_reversal_flag"]) == 1
        if not (is_bull_choch or is_bull_rev):
            continue
        ob_bottom, _ = hit_ob
        entry_price = float(px_close)
        stop_price = float(ob_bottom * 0.998)
        if stop_price <= 0 or entry_price <= stop_price:
            continue
        if tp <= entry_price:
            continue
        risk = entry_price - stop_price
        reward = tp - entry_price
        rr = reward / risk
        if rr < rr_min:
            continue
        signals.append(
            EntrySignal(
                index=int(i),
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=tp,
                rr=float(rr),
            )
        )
    return signals


@dataclass
class Trade:
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    rr: float
    pnl: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]


def backtest_fullsize_rr3(
    df_15m: pd.DataFrame,
    signals: list[EntrySignal],
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    if df_15m.empty:
        return BacktestResult(equity_curve=pd.Series(dtype=float), trades=[])
    equity = []
    idx_list = []
    capital = float(initial_capital)
    position_shares = 0
    entry_price = 0.0
    entry_idx = -1
    current_tp = 0.0
    current_sl = 0.0
    current_rr = 0.0
    sig_by_index = {s.index: s for s in signals}
    trades: list[Trade] = []
    for i in df_15m.index:
        bar = df_15m.loc[i]
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])
        if position_shares == 0 and i in sig_by_index:
            sig = sig_by_index[i]
            entry_price = sig.entry_price
            current_sl = sig.stop_price
            current_tp = sig.target_price
            current_rr = sig.rr
            entry_idx = i
            position_shares = int(capital // entry_price // 100 * 100)
            if position_shares <= 0:
                equity.append(capital)
                idx_list.append(i)
                continue
        if position_shares > 0:
            exit_idx = None
            exit_price = None
            if low <= current_sl:
                exit_price = current_sl
                exit_idx = i
            elif high >= current_tp:
                exit_price = current_tp
                exit_idx = i
            if exit_idx is not None and exit_price is not None:
                pnl = (exit_price - entry_price) * position_shares
                capital += pnl
                trades.append(
                    Trade(
                        entry_index=entry_idx,
                        exit_index=exit_idx,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stop_price=current_sl,
                        target_price=current_tp,
                        rr=current_rr,
                        pnl=pnl,
                    )
                )
                position_shares = 0
                entry_price = 0.0
                entry_idx = -1
                current_tp = 0.0
                current_sl = 0.0
                current_rr = 0.0
        if position_shares > 0:
            float_value = capital + (close - entry_price) * position_shares
            equity.append(float_value)
        else:
            equity.append(capital)
        idx_list.append(i)
    equity_series = pd.Series(equity, index=idx_list, name="equity")
    return BacktestResult(equity_curve=equity_series, trades=trades)


if __name__ == "__main__":
    n = 20
    df = pd.DataFrame({
        "open": np.linspace(10, 11, n),
        "high": np.linspace(10.1, 11.1, n),
        "low": np.linspace(9.9, 10.9, n),
        "close": np.linspace(10.05, 11.05, n),
        "ict_ob_flag": ([0]*15) + ([1]*5),
        "ict_ob_top": ([np.nan]*15) + ([10.8]*5),
        "ict_ob_bottom": ([np.nan]*15) + ([10.6]*5),
        "ict_choch_flag": ([0]*10) + ([1]*10),
    })
    sigs = generate_rr3_long_signals(df, target_price=11.2)
    res = backtest_fullsize_rr3(df, sigs)
    print(len(sigs), len(res.trades), float(res.equity_curve.iloc[-1]))