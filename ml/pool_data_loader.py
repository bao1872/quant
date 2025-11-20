from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
import time
import datetime as dt

from db.connection import get_engine

# ========= 关键时序特征（只对这些列做 lag1 + delta1） =========
# 请根据你实际的字段名调整：
# - price_position / v_price_position / band_width_zscore / v_band_width_zscore 是 stock_bollinger_data 里的
# - active_buy_vol_ratio 假设是你在 tick 聚合表里算好的“主买占比”
KEY_TS_FEATURES: List[str] = [
    "price_position",
    "v_price_position",
    "band_width_zscore",
    "v_band_width_zscore",
    # 如果你的 tick 日表里叫别的名字，比如 main_buy_vol_ratio，就把这行改成真实列名
    "active_buy_vol_ratio",
]
# ============================================================

LAG_BASE_FEATURES = [
    "price_position",
    "v_price_position",
    "band_width_zscore",
    "v_band_width_zscore",
    "vwap",
    "amount_sum",
    "vol_sum",
]


@dataclass
class PoolDataRangeConfig:
    start_date: dt.date
    end_date: dt.date
    label_col: str = "y_ret_10d"


def _date_chunks(start_date: dt.date, end_date: dt.date, chunk_days: int = 30) -> List[Tuple[dt.date, dt.date]]:
    chunks = []
    cur = start_date
    delta = dt.timedelta(days=chunk_days - 1)
    while cur <= end_date:
        end = min(cur + delta, end_date)
        chunks.append((cur, end))
        cur = end + dt.timedelta(days=1)
    return chunks


def load_pool_merged_dataset_iter(dr: PoolDataRangeConfig, chunk_days: int = 30):
    eng = get_engine()
    sql_s = (
        "SELECT ts_code, trade_date::date AS trade_date, " + dr.label_col + " AS " + dr.label_col +
        " FROM stock_pool_ml_samples WHERE trade_date>=:start_date AND trade_date<=:end_date "
        " AND " + dr.label_col + " IS NOT NULL ORDER BY ts_code, trade_date"
    )
    sql_b = (
        "SELECT b.* FROM stock_bollinger_data b INNER JOIN (SELECT ts_code, trade_date FROM stock_pool_ml_samples "
        " WHERE trade_date>=:start_date AND trade_date<=:end_date AND " + dr.label_col + " IS NOT NULL) s "
        " ON b.ts_code=s.ts_code AND b.trade_date=s.trade_date"
    )
    sql_t = (
        "SELECT t.* FROM stock_tick_daily_features t "
        "INNER JOIN (SELECT DISTINCT ts_code, trade_date FROM stock_pool_ml_samples "
        "            WHERE trade_date>=:start_date AND trade_date<=:end_date) s "
        "ON t.ts_code=s.ts_code AND t.trade_date=s.trade_date"
    )
    chunks = _date_chunks(dr.start_date, dr.end_date, chunk_days)
    print("[load_pool] query start", dr.start_date, dr.end_date, "chunks=", len(chunks), flush=True)
    with eng.connect() as conn:
        for i, (s, e) in enumerate(chunks, start=1):
            t0 = time.perf_counter()
            print("[load_pool] chunk", i, "/", len(chunks), "samples query", s, e, flush=True)
            df_s = pd.read_sql(text(sql_s), conn, params={"start_date": s, "end_date": e}, parse_dates=["trade_date"])
            df_s = df_s.loc[:, ~df_s.columns.duplicated()]
            if df_s.empty:
                print("[load_pool] chunk", i, "/", len(chunks), "samples empty", flush=True)
                yield df_s
                continue
            df_b = pd.read_sql(text(sql_b), conn, params={"start_date": s, "end_date": e}, parse_dates=["trade_date"])
            df_b = df_b.loc[:, ~df_b.columns.duplicated()]
            df_sb = pd.merge(df_s, df_b, on=["ts_code", "trade_date"], how="inner", suffixes=("", "_b"))
            t1 = time.perf_counter()
            print("[load_pool] chunk", i, "/", len(chunks), "s+b rows=", len(df_sb), "elapsed=", round(t1 - t0, 3), "s", flush=True)
            t2 = time.perf_counter()
            print("[load_pool] chunk", i, "/", len(chunks), "t query", s, e, flush=True)
            df_t = pd.read_sql(text(sql_t), conn, params={"start_date": s, "end_date": e}, parse_dates=["trade_date"])
            df_t = df_t.loc[:, ~df_t.columns.duplicated()]
            t3 = time.perf_counter()
            print("[load_pool] chunk", i, "/", len(chunks), "t rows=", len(df_t), "elapsed=", round(t3 - t2, 3), "s", flush=True)
            df = pd.merge(df_sb, df_t, on=["ts_code", "trade_date"], how="left")
            df = df.loc[:, ~df.columns.duplicated()]
            df["trade_date"] = df["trade_date"].dt.date
            if dr.label_col in df.columns:
                df[dr.label_col] = pd.to_numeric(df[dr.label_col], errors="coerce")
            print("[load_pool] chunk", i, "/", len(chunks), "merge rows=", len(df), flush=True)
            yield df


def load_pool_merged_dataset(dr: PoolDataRangeConfig, enable_3day_features: bool = False) -> Tuple[pd.DataFrame, List[str], str]:
    frames = []
    total_rows = 0
    for df in load_pool_merged_dataset_iter(dr, chunk_days=30):
        frames.append(df)
        total_rows += len(df)
    df = pd.concat(frames, ignore_index=True)
    if dr.label_col in df.columns:
        df[dr.label_col] = pd.to_numeric(df[dr.label_col], errors="coerce")
    print("[load_pool] concat rows=", total_rows, flush=True)
    if df.empty:
        raise RuntimeError("No samples found in stock_pool_ml_samples for given range.")
    if enable_3day_features:
        df = add_3day_features(df, LAG_BASE_FEATURES, n_lags=2, use_mean3=True)
    df = add_lag1_and_delta_features(df, KEY_TS_FEATURES)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if dr.label_col not in numeric_cols:
        raise RuntimeError("Label column not found in numeric columns.")
    feature_cols = [c for c in numeric_cols if c != dr.label_col and not c.startswith("y_")]
    return df, feature_cols, dr.label_col


def clean_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        s = df[col].astype(float)
        if s.notna().sum() == 0:
            df[col] = 0.0
            continue
        q1, q99 = s.quantile([0.01, 0.99])
        df[col] = s.clip(q1, q99).fillna(0.0)
    return df


def add_lag1_and_delta_features(df: pd.DataFrame, key_features: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = df.groupby("ts_code", sort=False)
    for col in key_features:
        if col not in df.columns:
            print(f"[lag] skip feature={col}, not in df.columns", flush=True)
            continue
        lag_col = f"{col}_lag1"
        delta_col = f"{col}_delta1"
        df[lag_col] = grouped[col].shift(1)
        df[delta_col] = df[col] - df[lag_col]
    return df


def add_3day_features(df: pd.DataFrame, lag_base_features: List[str], n_lags: int = 2, use_mean3: bool = True) -> pd.DataFrame:
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    g = df.groupby("ts_code")
    for col in lag_base_features:
        if col not in df.columns:
            continue
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        if use_mean3:
            df[f"{col}_mean3"] = g[col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    return df


if __name__ == "__main__":
    dr = PoolDataRangeConfig(start_date=dt.date(2024, 11, 15), end_date=dt.date.today(), label_col="y_ret_10d")
    df, feat_cols, label_col = load_pool_merged_dataset(dr)
    df = clean_features(df, feat_cols)
    print(df.head())
    print("Feature num:", len(feat_cols), "Label:", label_col)
