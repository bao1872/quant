from __future__ import annotations

import numpy as np
import pandas as pd


def mark_bullish_reversal(df: pd.DataFrame) -> pd.Series:
    o = pd.to_numeric(df["open"], errors="coerce").values
    c = pd.to_numeric(df["close"], errors="coerce").values
    prev_o = np.roll(o, 1)
    prev_c = np.roll(c, 1)
    is_prev_bear = prev_c < prev_o
    is_cur_bull = c > o
    mid_prev = 0.5 * (prev_o + prev_c)
    recapture = c >= mid_prev
    flag = (is_prev_bear & is_cur_bull & recapture)
    if len(flag) > 0:
        flag[0] = False
    return pd.Series(flag.astype(int), index=df.index, name="bull_reversal_flag")


def mark_bearish_reversal(df: pd.DataFrame) -> pd.Series:
    o = pd.to_numeric(df["open"], errors="coerce").values
    c = pd.to_numeric(df["close"], errors="coerce").values
    prev_o = np.roll(o, 1)
    prev_c = np.roll(c, 1)
    is_prev_bull = prev_c > prev_o
    is_cur_bear = c < o
    mid_prev = 0.5 * (prev_o + prev_c)
    dump_back = c <= mid_prev
    flag = (is_prev_bull & is_cur_bear & dump_back)
    if len(flag) > 0:
        flag[0] = False
    return pd.Series(flag.astype(int), index=df.index, name="bear_reversal_flag")


def mark_bullish_pinbar(df: pd.DataFrame, min_tail_ratio: float = 0.6, max_body_ratio: float = 0.3) -> pd.Series:
    o = pd.to_numeric(df["open"], errors="coerce").values
    h = pd.to_numeric(df["high"], errors="coerce").values
    l = pd.to_numeric(df["low"], errors="coerce").values
    c = pd.to_numeric(df["close"], errors="coerce").values
    body = np.abs(c - o)
    rng = (h - l)
    rng[rng == 0] = np.nan
    body_ratio = body / rng
    lower = np.minimum(o, c)
    upper = np.maximum(o, c)
    lower_shadow = lower - l
    upper_shadow = h - upper
    tail_ratio = lower_shadow / rng
    cond_body = body_ratio <= max_body_ratio
    cond_tail = tail_ratio >= min_tail_ratio
    cond_upper_short = upper_shadow <= (0.3 * rng)
    flag = cond_body & cond_tail & cond_upper_short
    return pd.Series(flag.astype(int), index=df.index, name="bull_pinbar_flag")


def mark_bearish_pinbar(df: pd.DataFrame, min_tail_ratio: float = 0.6, max_body_ratio: float = 0.3) -> pd.Series:
    o = pd.to_numeric(df["open"], errors="coerce").values
    h = pd.to_numeric(df["high"], errors="coerce").values
    l = pd.to_numeric(df["low"], errors="coerce").values
    c = pd.to_numeric(df["close"], errors="coerce").values
    body = np.abs(c - o)
    rng = (h - l)
    rng[rng == 0] = np.nan
    body_ratio = body / rng
    lower = np.minimum(o, c)
    upper = np.maximum(o, c)
    lower_shadow = lower - l
    upper_shadow = h - upper
    tail_ratio = upper_shadow / rng
    cond_body = body_ratio <= max_body_ratio
    cond_tail = tail_ratio >= min_tail_ratio
    cond_lower_short = lower_shadow <= (0.3 * rng)
    flag = cond_body & cond_tail & cond_lower_short
    return pd.Series(flag.astype(int), index=df.index, name="bear_pinbar_flag")


def attach_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bull_reversal_flag"] = mark_bullish_reversal(df)
    df["bear_reversal_flag"] = mark_bearish_reversal(df)
    df["bull_pinbar_flag"] = mark_bullish_pinbar(df)
    df["bear_pinbar_flag"] = mark_bearish_pinbar(df)
    df["bull_entry_signal"] = ((df["bull_reversal_flag"] == 1) | (df["bull_pinbar_flag"] == 1)).astype(int)
    df["bear_entry_signal"] = ((df["bear_reversal_flag"] == 1) | (df["bear_pinbar_flag"] == 1)).astype(int)
    return df


if __name__ == "__main__":
    idx = pd.date_range("2024-03-01", periods=20, freq="15min")
    base = np.linspace(10, 10.5, 20)
    df = pd.DataFrame({
        "datetime": idx,
        "open": base,
        "high": base + 0.2,
        "low": base - 0.2,
        "close": base + np.sin(np.linspace(0, 10, 20)) * 0.1,
    })
    out = attach_entry_signals(df)
    print(int(out["bull_entry_signal"].sum()))
    print(int(out["bear_entry_signal"].sum()))