from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Literal

import numpy as np
import pandas as pd

from factors.ict_smc import ICTConfig, compute_ict_structures


class TrendState(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class IctMtfConfig:
    swing_length_daily: int = 20
    swing_length_exec: int = 5
    risk_per_trade_pct: float = 0.01
    min_rr: float = 3.0
    ob_tolerance_pct: float = 0.005
    lot_size: int = 100
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0005
    max_position_pct: float = 0.3


@dataclass
class IctTrade:
    side: Literal["long", "short"]
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    qty: int
    rr: float
    pnl: float
    pnl_after_fee: float


@dataclass
class IctBacktestResult:
    trades: List[IctTrade]
    equity: pd.Series
    config: IctMtfConfig
    debug: pd.DataFrame


def _is_bullish_reversal(
    prev_open: float,
    prev_close: float,
    cur_open: float,
    cur_close: float,
) -> bool:
    if any(np.isnan(x) for x in [prev_open, prev_close, cur_open, cur_close]):
        return False
    if prev_close >= prev_open:
        return False
    if cur_close <= cur_open:
        return False
    mid_prev = 0.5 * (prev_open + prev_close)
    return cur_close >= mid_prev


def _is_bearish_reversal(
    prev_open: float,
    prev_close: float,
    cur_open: float,
    cur_close: float,
) -> bool:
    if any(np.isnan(x) for x in [prev_open, prev_close, cur_open, cur_close]):
        return False
    if prev_close <= prev_open:
        return False
    if cur_close >= cur_open:
        return False
    mid_prev = 0.5 * (prev_open + prev_close)
    return cur_close <= mid_prev


def _compute_daily_trend(df_daily: pd.DataFrame, ict_cfg: Optional[ICTConfig] = None) -> pd.DataFrame:
    if ict_cfg is None:
        ict_cfg = ICTConfig()
    df = compute_ict_structures(df_daily, ict_cfg)
    trend_state: List[TrendState] = []
    last_state: TrendState = TrendState.FLAT
    long_ob_top: List[float] = []
    long_ob_bottom: List[float] = []
    short_ob_top: List[float] = []
    short_ob_bottom: List[float] = []
    cur_long_top = np.nan
    cur_long_bottom = np.nan
    cur_short_top = np.nan
    cur_short_bottom = np.nan
    for i in range(len(df)):
        bos = float(df.get("ict_bos_flag", pd.Series(index=df.index)).iloc[i]) if "ict_bos_flag" in df.columns else 0.0
        choch = float(df.get("ict_choch_flag", pd.Series(index=df.index)).iloc[i]) if "ict_choch_flag" in df.columns else 0.0
        ob_flag = float(df.get("ict_ob_flag", pd.Series(index=df.index)).iloc[i]) if "ict_ob_flag" in df.columns else 0.0
        state = last_state
        if bos > 0 or choch > 0:
            state = TrendState.LONG
        elif bos < 0 or choch < 0:
            state = TrendState.SHORT
        if last_state == TrendState.FLAT and state not in (TrendState.LONG, TrendState.SHORT):
            state = TrendState.FLAT
        if ob_flag > 0:
            cur_long_top = float(df.get("ict_ob_top").iloc[i])
            cur_long_bottom = float(df.get("ict_ob_bottom").iloc[i])
        elif ob_flag < 0:
            cur_short_top = float(df.get("ict_ob_top").iloc[i])
            cur_short_bottom = float(df.get("ict_ob_bottom").iloc[i])
        trend_state.append(state)
        long_ob_top.append(cur_long_top)
        long_ob_bottom.append(cur_long_bottom)
        short_ob_top.append(cur_short_top)
        short_ob_bottom.append(cur_short_bottom)
        last_state = state
    out = df[[c for c in ["datetime", "close"] if c in df.columns]].copy()
    if "datetime" not in out.columns:
        out["datetime"] = df.index
    out["trend_state"] = trend_state
    out["ob_long_top"] = long_ob_top
    out["ob_long_bottom"] = long_ob_bottom
    out["ob_short_top"] = short_ob_top
    out["ob_short_bottom"] = short_ob_bottom
    return out


def _price_in_range(price: float, bottom: float, top: float, tol_pct: float) -> bool:
    if np.isnan(bottom) or np.isnan(top):
        return False
    lo = bottom * (1.0 - tol_pct)
    hi = top * (1.0 + tol_pct)
    return lo <= price <= hi


def run_ict_mtf_backtest(
    df_daily: pd.DataFrame,
    df_exec: pd.DataFrame,
    config: Optional[IctMtfConfig] = None,
    initial_capital: float = 1_000_000.0,
) -> IctBacktestResult:
    if config is None:
        config = IctMtfConfig()
    for df in (df_daily, df_exec):
        if "datetime" in df.columns:
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            df.sort_index(inplace=True)
            df["datetime"] = df.index
    daily_cfg = ICTConfig(swing_length=config.swing_length_daily)
    daily_state = _compute_daily_trend(df_daily, daily_cfg)
    daily_state["trade_date"] = pd.to_datetime(daily_state["datetime"]).dt.date
    daily_state = daily_state.set_index("trade_date")
    exec_cfg = ICTConfig(swing_length=config.swing_length_exec)
    exec_df = compute_ict_structures(df_exec, exec_cfg).copy()
    exec_df["datetime"] = pd.to_datetime(exec_df["datetime"]) if "datetime" in exec_df.columns else pd.to_datetime(exec_df.index)
    exec_df["trade_date"] = exec_df["datetime"].dt.date
    merged = exec_df.merge(
        daily_state[["trend_state", "ob_long_top", "ob_long_bottom", "ob_short_top", "ob_short_bottom"]],
        left_on="trade_date",
        right_index=True,
        how="left",
        suffixes=("", "_D"),
    )
    def _normalize_trend_state(val) -> TrendState:
        if isinstance(val, TrendState):
            return val
        if isinstance(val, str):
            s = val.strip().lower()
            if "long" in s:
                return TrendState.LONG
            if "short" in s:
                return TrendState.SHORT
            return TrendState.FLAT
        return TrendState.FLAT
    merged["trend_state"] = merged["trend_state"].apply(_normalize_trend_state)
    cash = float(initial_capital)
    equity = float(initial_capital)
    position_side: Optional[Literal["long", "short"]] = None
    position_qty: int = 0
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    trades: List[IctTrade] = []
    equity_values: List[float] = []
    debug_rows: List[dict] = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        dt = row["datetime"]
        price = float(row["close"]) if "close" in merged.columns else float(row.get("close", 0.0))
        high = float(row["high"]) if "high" in merged.columns else price
        low = float(row["low"]) if "low" in merged.columns else price
        trend: TrendState = row["trend_state"]
        ob_flag = float(row.get("ict_ob_flag", 0.0))
        opened_long = False
        opened_short = False
        closed_pos = False
        o = float(row.get("open", price))
        c = float(row.get("close", price))
        if i > 0:
            prev_row = merged.iloc[i - 1]
            prev_o = float(prev_row.get("open", o))
            prev_c = float(prev_row.get("close", c))
            bull_rev = _is_bullish_reversal(prev_o, prev_c, o, c)
            bear_rev = _is_bearish_reversal(prev_o, prev_c, o, c)
        else:
            bull_rev = False
            bear_rev = False
        if position_side is not None and position_qty > 0:
            exit_reason = None
            exit_px = None
            if position_side == "long":
                if low <= stop_price:
                    exit_px = stop_price * (1.0 - config.slippage_pct)
                    exit_reason = "stop"
                elif high >= target_price:
                    exit_px = target_price * (1.0 - config.slippage_pct)
                    exit_reason = "target"
                elif trend == TrendState.SHORT:
                    exit_px = price * (1.0 - config.slippage_pct)
                    exit_reason = "daily_flip"
            elif position_side == "short":
                if high >= stop_price:
                    exit_px = stop_price * (1.0 + config.slippage_pct)
                    exit_reason = "stop"
                elif low <= target_price:
                    exit_px = target_price * (1.0 + config.slippage_pct)
                    exit_reason = "target"
                elif trend == TrendState.LONG:
                    exit_px = price * (1.0 + config.slippage_pct)
                    exit_reason = "daily_flip"
            if exit_px is not None:
                closed_pos = True
                if position_side == "long":
                    cash += exit_px * position_qty
                    gross_pnl = (exit_px - entry_price) * position_qty
                    R = entry_price - stop_price
                    rr = (exit_px - entry_price) / R if R > 0 else 0.0
                else:
                    cash -= exit_px * position_qty
                    gross_pnl = (entry_price - exit_px) * position_qty
                    R = stop_price - entry_price
                    rr = (entry_price - exit_px) / R if R > 0 else 0.0
                fee = config.fee_rate * (abs(entry_price) + abs(exit_px)) * position_qty
                pnl_after_fee = gross_pnl - fee
                cash -= fee
                equity = cash
                trades.append(
                    IctTrade(
                        side=position_side,
                        entry_time=merged.iloc[i - 1]["datetime"] if i > 0 else dt,
                        exit_time=dt,
                        entry_price=float(entry_price),
                        exit_price=float(exit_px),
                        stop_price=float(stop_price),
                        target_price=float(target_price),
                        qty=int(position_qty),
                        rr=float(rr),
                        pnl=float(gross_pnl),
                        pnl_after_fee=float(pnl_after_fee),
                    )
                )
                position_side = None
                position_qty = 0
                entry_price = 0.0
                stop_price = 0.0
                target_price = 0.0
        if position_side is None:
            can_open_long = False
            can_open_short = False
            if trend == TrendState.LONG:
                in_ob_long = False
                if ob_flag > 0 and "ict_ob_top" in row and "ict_ob_bottom" in row:
                    ob_bottom = float(row["ict_ob_bottom"])
                    ob_top = float(row["ict_ob_top"])
                    in_ob_long = _price_in_range(price, ob_bottom, ob_top, config.ob_tolerance_pct)
                else:
                    ob_bottom = float(row.get("ob_long_bottom", np.nan))
                    ob_top = float(row.get("ob_long_top", np.nan))
                    in_ob_long = _price_in_range(price, ob_bottom, ob_top, config.ob_tolerance_pct)
                if in_ob_long and bull_rev:
                    can_open_long = True
                    if np.isnan(ob_bottom) or ob_bottom <= 0 or ob_bottom >= price:
                        stop = price * (1.0 - 0.005)
                    else:
                        stop = ob_bottom
                    risk_per_share = price - stop
                    if risk_per_share > 0:
                        risk_budget = equity * config.risk_per_trade_pct
                        max_position_value = equity * config.max_position_pct
                        qty = int(risk_budget / risk_per_share)
                        qty = (qty // config.lot_size) * config.lot_size
                        if qty > 0:
                            position_value = qty * price
                            if position_value > max_position_value:
                                qty = int(max_position_value // price)
                                qty = (qty // config.lot_size) * config.lot_size
                        if qty > 0:
                            target = price + config.min_rr * risk_per_share
                            entry_px = price * (1.0 + config.slippage_pct)
                            if (target - entry_px) / (entry_px - stop) >= config.min_rr:
                                position_side = "long"
                                position_qty = int(qty)
                                entry_price = float(entry_px)
                                stop_price = float(stop)
                                target_price = float(target)
                                cash -= entry_price * position_qty
                                opened_long = True
            elif trend == TrendState.SHORT:
                in_ob_short = False
                if ob_flag < 0 and "ict_ob_top" in row and "ict_ob_bottom" in row:
                    ob_top = float(row["ict_ob_top"])
                    ob_bottom = float(row["ict_ob_bottom"])
                    in_ob_short = _price_in_range(price, ob_bottom, ob_top, config.ob_tolerance_pct)
                else:
                    ob_top = float(row.get("ob_short_top", np.nan))
                    ob_bottom = float(row.get("ob_short_bottom", np.nan))
                    in_ob_short = _price_in_range(price, ob_bottom, ob_top, config.ob_tolerance_pct)
                if in_ob_short and bear_rev:
                    can_open_short = True
                    if np.isnan(ob_top) or ob_top <= 0 or ob_top <= price:
                        stop = price * (1.0 + 0.005)
                    else:
                        stop = ob_top
                    risk_per_share = stop - price
                    if risk_per_share > 0:
                        risk_budget = equity * config.risk_per_trade_pct
                        max_position_value = equity * config.max_position_pct
                        qty = int(risk_budget / risk_per_share)
                        qty = (qty // config.lot_size) * config.lot_size
                        if qty > 0:
                            position_value = qty * price
                            if position_value > max_position_value:
                                qty = int(max_position_value // price)
                                qty = (qty // config.lot_size) * config.lot_size
                        if qty > 0:
                            target = price - config.min_rr * risk_per_share
                            entry_px = price * (1.0 - config.slippage_pct)
                            if (entry_px - target) / (stop - entry_px) >= config.min_rr:
                                position_side = "short"
                                position_qty = int(qty)
                                entry_price = float(entry_px)
                                stop_price = float(stop)
                                target_price = float(target)
                                cash += entry_price * position_qty
                                opened_short = True
        if position_side is None or position_qty == 0:
            equity = cash
        else:
            if position_side == "long":
                equity = cash + price * position_qty
            else:
                equity = cash - price * position_qty
        pos_state = position_side if position_side is not None else "flat"
        debug_rows.append({
            "datetime": dt,
            "close": float(price),
            "trend_state": str(trend),
            "in_exec_ob_flag": float(ob_flag),
            "bull_rev": bool(bull_rev),
            "bear_rev": bool(bear_rev),
            "can_open_long": bool('can_open_long' in locals() and can_open_long),
            "can_open_short": bool('can_open_short' in locals() and can_open_short),
            "opened_long": bool(opened_long),
            "opened_short": bool(opened_short),
            "closed_pos": bool(closed_pos),
            "position_side": pos_state,
            "position_qty": int(position_qty),
            "stop_price": float(stop_price) if stop_price else np.nan,
            "target_price": float(target_price) if target_price else np.nan,
            "equity": float(equity),
        })
        equity_values.append(equity)
    equity_index = merged["datetime"]
    equity_series = pd.Series(equity_values, index=equity_index, name="equity")
    debug_df = pd.DataFrame(debug_rows)
    if not debug_df.empty:
        debug_df.set_index("datetime", inplace=True)
    return IctBacktestResult(trades=trades, equity=equity_series, config=config, debug=debug_df)

if __name__ == "__main__":
    n_day = 60
    n_exec = 240
    idx_day = pd.date_range("2024-01-01", periods=n_day, freq="1D")
    base_day = np.linspace(10, 12, n_day)
    df_daily = pd.DataFrame({
        "datetime": idx_day,
        "open": base_day,
        "high": base_day + 0.5,
        "low": base_day - 0.5,
        "close": base_day + np.sin(np.linspace(0, 6.28, n_day)) * 0.2,
        "volume": np.ones(n_day) * 1_000_000,
    })
    idx_exec = pd.date_range("2024-03-01", periods=n_exec, freq="15min")
    base_exec = np.linspace(11, 12, n_exec)
    df_exec = pd.DataFrame({
        "datetime": idx_exec,
        "open": base_exec,
        "high": base_exec + 0.3,
        "low": base_exec - 0.3,
        "close": base_exec + np.sin(np.linspace(0, 20, n_exec)) * 0.1,
        "volume": np.ones(n_exec) * 10000,
    })
    cfg = IctMtfConfig(swing_length_daily=20, swing_length_exec=5, risk_per_trade_pct=0.01)
    res = run_ict_mtf_backtest(df_daily, df_exec, cfg)
    print({"trades": len(res.trades), "equity_last": float(res.equity.iloc[-1])})