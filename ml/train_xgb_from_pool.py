from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor
import time
import os

from ml.pool_data_loader import PoolDataRangeConfig, load_pool_merged_dataset, clean_features


@dataclass
class TrainConfig:
    model_path: str = "models/xgb_from_pool_ret10d.pkl"
    train_end: dt.date = dt.date(2025, 8, 31)
    valid_end: dt.date = dt.date(2025, 9, 30)


def time_split(df: pd.DataFrame, label_col: str, feature_cols: List[str], cfg: TrainConfig):
    df = df.sort_values("trade_date").reset_index(drop=True)
    train_mask = df["trade_date"] <= cfg.train_end
    valid_mask = (df["trade_date"] > cfg.train_end) & (df["trade_date"] <= cfg.valid_end)
    test_mask = df["trade_date"] > cfg.valid_end
    X_train = df.loc[train_mask, feature_cols].to_numpy(dtype=float)
    y_train = df.loc[train_mask, label_col].to_numpy(dtype=float)
    X_valid = df.loc[valid_mask, feature_cols].to_numpy(dtype=float)
    y_valid = df.loc[valid_mask, label_col].to_numpy(dtype=float)
    X_test = df.loc[test_mask, feature_cols].to_numpy(dtype=float)
    y_test = df.loc[test_mask, label_col].to_numpy(dtype=float)
    print(f"Train size: {X_train.shape}, Valid size: {X_valid.shape}, Test size: {X_test.shape}")
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def calc_daily_ic(df: pd.DataFrame, pred_col: str, label_col: str) -> pd.Series:
    ics = []
    dates = []
    for d, sub in df.groupby("trade_date"):
        if sub[pred_col].nunique() <= 1:
            continue
        ic, _ = spearmanr(sub[pred_col], sub[label_col])
        ics.append(ic)
        dates.append(d)
    return pd.Series(ics, index=dates, name="ic")


def quantile_backtest(df: pd.DataFrame, pred_col: str, label_col: str, n_quantiles: int = 5) -> pd.DataFrame:
    records = []
    for d, sub in df.groupby("trade_date"):
        if len(sub) < n_quantiles:
            continue
        try:
            sub = sub.copy()
            sub["q"] = pd.qcut(sub[pred_col], q=n_quantiles, labels=False) + 1
        except ValueError:
            continue
        for q, sub_q in sub.groupby("q"):
            records.append({"trade_date": d, "quantile": int(q), "avg_ret": sub_q[label_col].mean()})
    df_q = pd.DataFrame(records)
    if df_q.empty:
        return df_q
    pivot = df_q.pivot(index="trade_date", columns="quantile", values="avg_ret").sort_index()
    pivot.columns = [f"Q{int(c)}" for c in pivot.columns]
    return pivot


def train_from_pool():
    dr = PoolDataRangeConfig(start_date=dt.date(2024, 11, 15), end_date=dt.date.today(), label_col="y_ret_10d")
    print("[train] load start", flush=True)
    t0 = time.perf_counter()
    df, feature_cols, label_col = load_pool_merged_dataset(dr, enable_3day_features=False)
    t1 = time.perf_counter()
    print("[train] load done shape=", df.shape, "features=", len(feature_cols), "elapsed=", round(t1 - t0, 3), "s", flush=True)
    t2 = time.perf_counter()
    df = clean_features(df, feature_cols)
    t3 = time.perf_counter()
    print("[train] clean done elapsed=", round(t3 - t2, 3), "s", flush=True)
    cfg = TrainConfig()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = time_split(df, label_col, feature_cols, cfg)
    model = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0, objective="reg:squarederror", n_jobs=8, tree_method="hist")
    print("[train] fit start", flush=True)
    t4 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    t5 = time.perf_counter()
    print("[train] fit done elapsed=", round(t5 - t4, 3), "s", flush=True)
    df_eval = df.copy()
    X_all = df_eval[feature_cols].to_numpy(dtype=float)
    t6 = time.perf_counter()
    df_eval["pred_ret_10d"] = model.predict(X_all)
    t7 = time.perf_counter()
    print("[train] predict done elapsed=", round(t7 - t6, 3), "s", flush=True)
    df_test = df_eval[df_eval["trade_date"] > cfg.valid_end].copy()
    if not df_test.empty:
        t8 = time.perf_counter()
        ic_series = calc_daily_ic(df_test, "pred_ret_10d", label_col)
        t9 = time.perf_counter()
        print("[eval] ic done mean=", float(ic_series.mean()), "std=", float(ic_series.std()), "elapsed=", round(t9 - t8, 3), "s", flush=True)
        t10 = time.perf_counter()
        q_ret = quantile_backtest(df_test, "pred_ret_10d", label_col, n_quantiles=5)
        t11 = time.perf_counter()
        print("[eval] qcut done elapsed=", round(t11 - t10, 3), "s", flush=True)
        print(q_ret.mean())
    bundle = {"model": model, "feature_cols": feature_cols, "label_col": label_col, "train_end": cfg.train_end, "valid_end": cfg.valid_end}
    t12 = time.perf_counter()
    dirp = os.path.dirname(cfg.model_path)
    if dirp:
        os.makedirs(dirp, exist_ok=True)
    joblib.dump(bundle, cfg.model_path)
    t13 = time.perf_counter()
    print("[train] saved", cfg.model_path, "elapsed=", round(t13 - t12, 3), "s", flush=True)


if __name__ == "__main__":
    train_from_pool()
