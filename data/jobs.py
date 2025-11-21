# data/jobs.py
"""
定时任务入口：

- job_update_ohlc: 盘前/盘后更新日线 & 分钟线
- job_collect_full_day_ticks: 收盘后全市场 full-day tick 抓取
- job_finalize_ticks_and_levels: 收盘后（或盘前）计算关键位
"""

from __future__ import annotations

from datetime import date as _date
from datetime import timedelta as _timedelta
from datetime import time as _time

from .updater import update_daily_bars, update_minute_bars, collect_full_day_ticks
from .concepts_cache import update_concepts_cache, update_hk_industry_cache, validate_concepts_cache_count
from factors.bollinger import compute_stock_bollinger_for_date, compute_concept_bollinger_for_date, _table_columns, compute_stock_bollinger_from_db_range, compute_concept_bollinger_from_db_range
from db.connection import get_engine
from sqlalchemy import text
from .repository import get_all_stock_basics
from factors import AbuPriceLevelProvider
import pandas as pd
from tqdm import tqdm
from sqlalchemy import inspect
from data.tick_store import TickStore
from datetime import date as date_cls
from dataclasses import dataclass
import numpy as np
import logging
from pathlib import Path
from config import TICK_BASE_DIR
import pyarrow.parquet as pq


def job_update_ohlc(trade_date: _date) -> None:
    print(f"[job_update_ohlc] trade_date={trade_date}")
    update_daily_bars(trade_date=trade_date, count=600)
    job_update_minute_60m(trade_date)
    job_update_minute_15m(trade_date)


def _validate_minute_for_date(d: _date) -> int:
    eng = get_engine()
    with eng.connect() as conn:
        dfc = pd.read_sql(text("select count(1) as c from stock_minute where trade_date=:d"), conn, params={"d": d})
    cnt = int(pd.to_numeric(dfc.get("c", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not dfc.empty else 0
    print(f"[validate_minute] rows@{d}={cnt}")
    return cnt


def job_update_minute_60m(trade_date: _date) -> None:
    print(f"[job_update_minute_60m] trade_date={trade_date}")
    update_minute_bars(trade_date=trade_date, freq="60m", count=240)
    n = _validate_minute_for_date(trade_date)
    if n == 0:
        print("[job_update_minute_60m] empty result")


def job_update_minute_15m(trade_date: _date) -> None:
    print(f"[job_update_minute_15m] trade_date={trade_date}")
    update_minute_bars(trade_date=trade_date, freq="15m", count=240)
    n = _validate_minute_for_date(trade_date)
    if n == 0:
        print("[job_update_minute_15m] empty result")


def job_collect_full_day_ticks(trade_date: _date) -> None:
    print(f"[job_collect_full_day_ticks] trade_date={trade_date}")
    collect_full_day_ticks(trade_date)


def job_finalize_ticks_and_levels(trade_date: _date) -> None:
    print(f"[job_finalize_ticks_and_levels] trade_date={trade_date}")
    provider = AbuPriceLevelProvider()
    provider.precompute(trade_date)


def _list_tick_parquet_for_date(d: _date) -> list[Path]:
    base = Path(TICK_BASE_DIR)
    if not base.exists():
        return []
    pat = f"{pd.Timestamp(d).strftime('%Y%m%d')}_交易数据.parquet"
    return list(base.rglob(pat))


def _scan_fs_tick_dates() -> list[_date]:
    base = Path(TICK_BASE_DIR)
    if not base.exists():
        return []
    dates: set[_date] = set()
    for p in base.rglob("*_交易数据.parquet"):
        name = p.name
        s = name.split("_")[0]
        if len(s) == 8 and s.isdigit():
            y = int(s[0:4]); m = int(s[4:6]); d = int(s[6:8])
            dates.add(_date(y, m, d))
    return sorted(list(dates))


def _tick_daily_features_group(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("datetime")
    price = pd.to_numeric(g["price"], errors="coerce").astype(float)
    vol = pd.to_numeric(g["volume"], errors="coerce").astype(float)
    amt = pd.to_numeric(g.get("amount", 0.0), errors="coerce").astype(float)
    n = len(g)
    if n == 0:
        return pd.Series(dtype=float)
    open_p = float(price.iloc[0])
    close_p = float(price.iloc[-1])
    high_p = float(price.max())
    low_p = float(price.min())
    intraday_ret = (close_p / open_p - 1.0) if open_p > 0 else np.nan
    hl_range = (high_p / low_p - 1.0) if low_p > 0 else np.nan
    if n > 1:
        log_p = np.log(price.replace(0, np.nan)).ffill().bfill()
        log_ret = log_p.diff().dropna()
        rv = float(np.sqrt((log_ret ** 2).sum()))
        up_moves_ratio = float((log_ret > 0).mean())
    else:
        rv = 0.0
        up_moves_ratio = 0.0
    vol_sum = float(vol.sum())
    amount_sum = float(amt.sum())
    num_trades = int(n)
    avg_trade_size = float(vol_sum / num_trades) if num_trades > 0 else 0.0
    median_trade_size = float(np.median(vol)) if num_trades > 0 else 0.0
    if vol_sum > 0:
        vwap = float(np.sum(price * vol) / vol_sum)
    else:
        vwap = np.nan
    close_vs_vwap = (close_p / vwap - 1.0) if (vwap is not None and vwap > 0) else np.nan
    close_vs_low = (close_p / low_p - 1.0) if low_p > 0 else np.nan
    close_vs_high = (close_p / high_p - 1.0) if high_p > 0 else np.nan
    t = pd.to_datetime(g["datetime"]).dt.time
    morning_mask = t < _time(11, 30)
    afternoon_mask = t >= _time(13, 0)
    vol_morning = float(vol[morning_mask].sum())
    vol_afternoon = float(vol[afternoon_mask].sum())
    open_vol_ratio = vol_morning / vol_sum if vol_sum > 0 else 0.0
    close_vol_ratio = vol_afternoon / vol_sum if vol_sum > 0 else 0.0
    if morning_mask.any():
        g_m = price[morning_mask]
        open_m = float(g_m.iloc[0])
        close_m = float(g_m.iloc[-1])
        return_morning = (close_m / open_m - 1.0) if open_m > 0 else 0.0
    else:
        return_morning = 0.0
    if afternoon_mask.any():
        g_a = price[afternoon_mask]
        open_a = float(g_a.iloc[0])
        close_a = float(g_a.iloc[-1])
        return_afternoon = (close_a / open_a - 1.0) if open_a > 0 else 0.0
    else:
        return_afternoon = 0.0
    if "side" in g.columns:
        s = g["side"].astype(str).str.upper().str.strip()
    else:
        s = pd.Series(["N"] * n, index=g.index)
    buy_mask = s.str.startswith("B")
    sell_mask = s.str.startswith("S")
    active_buy_vol = float(vol[buy_mask].sum())
    active_sell_vol = float(vol[sell_mask].sum())
    active_buy_count = int(buy_mask.sum())
    active_sell_count = int(sell_mask.sum())
    total_bs_vol = active_buy_vol + active_sell_vol
    total_bs_count = active_buy_count + active_sell_count
    active_buy_vol_ratio = (active_buy_vol / total_bs_vol) if total_bs_vol > 0 else np.nan
    active_buy_count_ratio = (active_buy_count / total_bs_count) if total_bs_count > 0 else np.nan
    order_imbalance = ((active_buy_vol - active_sell_vol) / total_bs_vol) if total_bs_vol > 0 else np.nan
    price_change_rate = (float((price.diff().fillna(0) != 0).sum()) / float(num_trades)) if num_trades > 0 else 0.0
    return pd.Series(
        {
            "open_price": open_p,
            "close_price": close_p,
            "high_price": high_p,
            "low_price": low_p,
            "intraday_ret": intraday_ret,
            "hl_range": hl_range,
            "rv": rv,
            "up_moves_ratio": up_moves_ratio,
            "vol_sum": vol_sum,
            "amount_sum": amount_sum,
            "num_trades": num_trades,
            "avg_trade_size": avg_trade_size,
            "median_trade_size": median_trade_size,
            "vol_morning": vol_morning,
            "vol_afternoon": vol_afternoon,
            "open_vol_ratio": open_vol_ratio,
            "close_vol_ratio": close_vol_ratio,
            "return_morning": return_morning,
            "return_afternoon": return_afternoon,
            "vwap": vwap,
            "close_vs_vwap": close_vs_vwap,
            "close_vs_low": close_vs_low,
            "close_vs_high": close_vs_high,
            "price_change_rate": price_change_rate,
            "active_buy_vol": active_buy_vol,
            "active_sell_vol": active_sell_vol,
            "active_buy_count": active_buy_count,
            "active_sell_count": active_sell_count,
            "active_buy_vol_ratio": active_buy_vol_ratio,
            "active_buy_count_ratio": active_buy_count_ratio,
            "order_imbalance": order_imbalance,
        }
    )


def job_build_tick_daily_features(trade_date: _date) -> int:
    print(f"[job_build_tick_daily_features] trade_date={trade_date}")
    files = _list_tick_parquet_for_date(trade_date)
    if not files:
        eng = get_engine()
        with eng.begin() as conn:
            inspector = inspect(conn)
            if inspector.has_table("stock_tick_daily_features", schema="public"):
                conn.execute(text("delete from stock_tick_daily_features where trade_date = :d"), {"d": trade_date})
        print("[job_build_tick_daily_features] no files")
        return 0
    rows = []
    bar = tqdm(total=len(files), desc="tick_feat_files", unit="file")
    for f in files:
        df = pd.read_parquet(f)
        if df is None or df.empty:
            bar.update(1)
            continue
        code = f.parent.name
        market = f.parent.parent.name
        ts = f"{code}.{market}"
        df["ts_code"] = ts
        df["trade_date"] = pd.Timestamp(trade_date)
        if "datetime" not in df.columns:
            if "trade_time" in df.columns:
                df["datetime"] = pd.to_datetime(df["trade_time"])
            else:
                df["datetime"] = pd.Timestamp(trade_date)
        df["price"] = pd.to_numeric(df.get("price", 0.0), errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0.0), errors="coerce")
        df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce")
        if "side" not in df.columns:
            df["side"] = "N"
        rows.append(df[["ts_code", "trade_date", "datetime", "price", "volume", "amount", "side"]])
        bar.update(1)
    bar.close()
    if not rows:
        print("[job_build_tick_daily_features] empty rows")
        return 0
    df_all = pd.concat(rows, ignore_index=True)
    feat_df = (
        df_all.groupby(["ts_code", "trade_date"], group_keys=False)
        .apply(lambda g: _tick_daily_features_group(g[["datetime", "price", "volume", "amount", "side"]]))
        .reset_index()
    )
    eng = get_engine()
    with eng.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("stock_tick_daily_features", schema="public"):
            conn.execute(text("delete from stock_tick_daily_features where trade_date = :d"), {"d": trade_date})
        feat_df.to_sql("stock_tick_daily_features", con=conn, if_exists="append", index=False)
    print(f"[job_build_tick_daily_features] inserted rows={len(feat_df)}")
    return len(feat_df)


def job_build_missing_tick_daily_features_for_date(trade_date: _date) -> int:
    eng = get_engine()
    files = _list_tick_parquet_for_date(trade_date)
    if not files:
        return 0
    pairs = []
    for f in files:
        code = f.parent.name
        market = f.parent.parent.name
        ts = f"{code}.{market}"
        pairs.append((ts, f))
    df_pairs = pd.DataFrame(pairs, columns=["ts_code", "path"]).drop_duplicates(subset=["ts_code"], keep="last")
    ts_available = df_pairs["ts_code"].astype(str).tolist()
    with eng.connect() as conn:
        inspector = inspect(conn)
        has_table = inspector.has_table("stock_tick_daily_features", schema="public")
        existing = pd.DataFrame()
        if has_table:
            existing = pd.read_sql(
                text("select ts_code, num_trades, vol_sum, vwap from stock_tick_daily_features where trade_date=:d"),
                conn,
                params={"d": trade_date},
            )
    if existing.empty:
        missing_ts = ts_available
    else:
        existing_ts = existing["ts_code"].astype(str).tolist()
        num_trades = pd.to_numeric(existing.get("num_trades", pd.Series(index=existing.index)), errors="coerce")
        vol_sum = pd.to_numeric(existing.get("vol_sum", pd.Series(index=existing.index)), errors="coerce")
        vwap = pd.to_numeric(existing.get("vwap", pd.Series(index=existing.index)), errors="coerce")
        null_mask = num_trades.isna() | vol_sum.isna() | vwap.isna()
        need_fix = existing.loc[null_mask, "ts_code"].astype(str).tolist()
        missing_ts = sorted(list(set(ts_available) - set(existing_ts)) | set(need_fix))
    if not missing_ts:
        return 0
    df_sel = df_pairs[df_pairs["ts_code"].isin(missing_ts)]
    bar = tqdm(total=len(df_sel), desc="tick_feat_missing", unit="stk")
    rows = []
    for _, r in df_sel.iterrows():
        f = Path(str(r["path"]))
        df = pd.read_parquet(f)
        if df is None or df.empty:
            bar.update(1)
            continue
        ts = str(r["ts_code"]) 
        df["ts_code"] = ts
        df["trade_date"] = pd.Timestamp(trade_date)
        if "datetime" not in df.columns:
            if "trade_time" in df.columns:
                df["datetime"] = pd.to_datetime(df["trade_time"])
            else:
                df["datetime"] = pd.Timestamp(trade_date)
        df["price"] = pd.to_numeric(df.get("price", 0.0), errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0.0), errors="coerce")
        df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce")
        if "side" not in df.columns:
            df["side"] = "N"
        rows.append(df[["ts_code", "trade_date", "datetime", "price", "volume", "amount", "side"]])
        bar.update(1)
    bar.close()
    if not rows:
        return 0
    df_all = pd.concat(rows, ignore_index=True)
    feat_df = (
        df_all.groupby(["ts_code", "trade_date"], group_keys=False)
        .apply(_tick_daily_features_group)
        .reset_index()
    )
    with eng.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("stock_tick_daily_features", schema="public"):
            conn.execute(
                text("delete from stock_tick_daily_features where trade_date=:d and ts_code = any(:codes)"),
                {"d": trade_date, "codes": missing_ts},
            )
        feat_df.to_sql("stock_tick_daily_features", con=conn, if_exists="append", index=False)
    return len(feat_df)


def job_init_tick_daily_features() -> int:
    eng = get_engine()
    with eng.connect() as conn:
        inspector = inspect(conn)
        has_table = inspector.has_table("stock_tick_daily_features", schema="public")
        cnt = 0
        if has_table:
            dfc = pd.read_sql(text("select count(1) as c from stock_tick_daily_features"), conn)
            if not dfc.empty:
                cnt = int(pd.to_numeric(dfc["c"], errors="coerce").fillna(0).iloc[0])
    if cnt > 0:
        return 0
    days = _scan_fs_tick_dates()
    if not days:
        return 0
    bar = tqdm(total=len(days), desc="tick_feat_init_days", unit="day")
    total = 0
    for d in days:
        total += job_build_missing_tick_daily_features_for_date(d)
        bar.update(1)
    bar.close()
    return total


def job_build_missing_tick_daily_features_by_stock() -> int:
    eng = get_engine()
    from data.updater import _get_all_stock_codes
    ts_codes = _get_all_stock_codes(None)
    if not ts_codes:
        return 0
    store = TickStore()
    bar_stk = tqdm(total=len(ts_codes), desc="tick_feat_stocks", unit="stk")
    total_rows = 0
    with eng.begin() as conn:
        inspector = inspect(conn)
        has_table = inspector.has_table("stock_tick_daily_features", schema="public")
        if not has_table:
            conn.execute(
                text(
                    """
                    create table if not exists public.stock_tick_daily_features (
                        ts_code text not null,
                        trade_date date not null,
                        open_price double precision,
                        close_price double precision,
                        high_price double precision,
                        low_price double precision,
                        intraday_ret double precision,
                        hl_range double precision,
                        rv double precision,
                        up_moves_ratio double precision,
                        vol_sum double precision,
                        amount_sum double precision,
                        num_trades integer,
                        avg_trade_size double precision,
                        median_trade_size double precision,
                        vol_morning double precision,
                        vol_afternoon double precision,
                        open_vol_ratio double precision,
                        close_vol_ratio double precision,
                        return_morning double precision,
                        return_afternoon double precision,
                        vwap double precision,
                        close_vs_vwap double precision,
                        close_vs_low double precision,
                        close_vs_high double precision,
                        price_change_rate double precision,
                        active_buy_vol double precision,
                        active_sell_vol double precision,
                        active_buy_count integer,
                        active_sell_count integer,
                        active_buy_vol_ratio double precision,
                        active_buy_count_ratio double precision,
                        order_imbalance double precision
                    )
                    """
                )
            )
            conn.execute(
                text(
                    "create index if not exists idx_stdf_ts_date on public.stock_tick_daily_features(ts_code, trade_date)"
                )
            )
    days = _scan_fs_tick_dates()
    if not days:
        bar_stk.close()
        return 0
    start_date = days[0]
    end_date = days[-1]
    for ts in ts_codes:
        bar_stk.set_description(f"tick_feat_stocks {ts}")
        idx_rows = store.list_tick_files(ts, start_date, end_date)
        if not idx_rows:
            bar_stk.update(1)
            continue
        fs_dates = sorted(list({r.trade_date for r in idx_rows}))
        with eng.connect() as conn:
            inspector = inspect(conn)
            has_table = inspector.has_table("stock_tick_daily_features", schema="public")
            existing = pd.DataFrame()
            if has_table:
                existing = pd.read_sql(
                    text("select trade_date, num_trades, vol_sum, vwap from stock_tick_daily_features where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2"),
                    conn,
                    params={"ts": ts, "d1": start_date, "d2": end_date},
                    parse_dates=["trade_date"],
                )
        if existing.empty:
            missing_dates = fs_dates
        else:
            existing["trade_date"] = pd.to_datetime(existing["trade_date"]).dt.date
            ok_mask = (~pd.isna(existing.get("num_trades"))) & (~pd.isna(existing.get("vol_sum"))) & (~pd.isna(existing.get("vwap")))
            ok_dates = set(existing.loc[ok_mask, "trade_date"].tolist())
            null_dates = set(existing.loc[~ok_mask, "trade_date"].tolist())
            fs_set = set(fs_dates)
            missing_dates = sorted(list((fs_set - ok_dates) | null_dates))
        if not missing_dates:
            bar_stk.update(1)
            continue
        rows = []
        for r in idx_rows:
            if r.trade_date in missing_dates:
                df = pd.read_parquet(r.file_path)
                if df is None or df.empty:
                    continue
                df["ts_code"] = ts
                df["trade_date"] = pd.Timestamp(r.trade_date)
                if "datetime" not in df.columns:
                    if "trade_time" in df.columns:
                        df["datetime"] = pd.to_datetime(df["trade_time"])
                    else:
                        df["datetime"] = pd.Timestamp(r.trade_date)
                df["price"] = pd.to_numeric(df.get("price", 0.0), errors="coerce")
                df["volume"] = pd.to_numeric(df.get("volume", 0.0), errors="coerce")
                df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce")
                if "side" not in df.columns:
                    df["side"] = "N"
                rows.append(df[["ts_code", "trade_date", "datetime", "price", "volume", "amount", "side"]])
        if not rows:
            bar_stk.update(1)
            continue
        df_all = pd.concat(rows, ignore_index=True)
        feat_df = (
            df_all.groupby(["ts_code", "trade_date"], group_keys=False)
            .apply(lambda g: _tick_daily_features_group(g[["datetime", "price", "volume", "amount", "side"]]))
            .reset_index()
        )
        with eng.begin() as conn:
            conn.execute(
                text("delete from stock_tick_daily_features where ts_code=:ts and trade_date = any(:dates)"),
                {"ts": ts, "dates": missing_dates},
            )
            feat_df.to_sql("stock_tick_daily_features", con=conn, if_exists="append", index=False)
        total_rows += len(feat_df)
        bar_stk.update(1)
    bar_stk.close()
    return total_rows


def job_update_concepts() -> None:
    print("[job_update_concepts] start")
    n1 = update_concepts_cache()
    n2 = update_hk_industry_cache()
    print(f"[job_update_concepts] a={n1} hk={n2}")
    from .concepts_cache import reconcile_concepts_cache_with_universe, purge_non_universe_from_concepts_cache
    fixed = reconcile_concepts_cache_with_universe()
    removed = purge_non_universe_from_concepts_cache()
    if fixed > 0 or removed > 0:
        print(f"[job_update_concepts] reconciled missing={fixed} removed_extra={removed}")
    validate_concepts_cache_count()


def job_update_bollinger(trade_date: _date) -> None:
    """
    仅计算并落库价格与成交量相关的布林带指标（不含任何人气相关逻辑）：
    - 价格布林线：upper/middle/lower/band_width/price_position/upper_return/lower_return
    - 成交量加权布林线：v_upper/v_middle/v_lower/v_band_width/v_price_position 及 zscore
    - 概念层聚合：按概念对当日个股布林指标取中位数与数量
    """
    print(f"[job_update_bollinger] trade_date={trade_date}")
    eng = get_engine()
    with eng.connect() as conn:
        df_daily = pd.read_sql("select distinct trade_date from stock_daily", conn)
        daily_dates = set(pd.to_datetime(df_daily["trade_date"]).dt.date.tolist()) if not df_daily.empty else set()
        cols_s = _table_columns(eng, "stock_bollinger_data")
        dc_s = "trade_date" if "trade_date" in cols_s else ("date" if "date" in cols_s else None)
        sbd_dates = set()
        if dc_s is not None and cols_s:
            df_s = pd.read_sql(f"select distinct {dc_s} as d from stock_bollinger_data", conn)
            sbd_dates = set(pd.to_datetime(df_s["d"]).dt.date.tolist()) if not df_s.empty else set()
        cols_c = _table_columns(eng, "concept_bollinger_data")
        dc_c = "trade_date" if "trade_date" in cols_c else ("date" if "date" in cols_c else None)
        cbd_dates = set()
        if dc_c is not None and cols_c:
            df_c = pd.read_sql(f"select distinct {dc_c} as d from concept_bollinger_data", conn)
            cbd_dates = set(pd.to_datetime(df_c["d"]).dt.date.tolist()) if not df_c.empty else set()
    earliest_sbd_default = _date(2025, 4, 16)
    earliest_sbd = (min(sbd_dates) if sbd_dates else earliest_sbd_default)
    missing_hist = sorted(d for d in (daily_dates - sbd_dates - {trade_date}) if d >= earliest_sbd_default)
    missing_concepts = sorted(d for d in (sbd_dates - cbd_dates) if d >= earliest_sbd)
    if missing_hist:
        print(f"[job_update_bollinger] backfill_missing_stock_days={len(missing_hist)}")
        with eng.begin() as conn:
            bar = tqdm(total=len(missing_hist), desc="backfill_stock", unit="day")
            ins_s = 0
            ins_c = 0
            for d0 in missing_hist:
                ins_s += compute_stock_bollinger_for_date(d0, conn=conn)
                ins_c += compute_concept_bollinger_for_date(d0, conn=conn)
                bar.update(1)
            bar.close()
        print(f"[job_update_bollinger] backfill_stock inserted stock_rows={ins_s} concept_rows={ins_c}")
    if missing_concepts:
        print(f"[job_update_bollinger] backfill_missing_concept_days={len(missing_concepts)}")
        bar2 = tqdm(total=len(missing_concepts), desc="backfill_concept", unit="day")
        ins_c2 = 0
        for d1 in missing_concepts:
            with eng.begin() as conn:
                ins_c2 += compute_concept_bollinger_for_date(d1, conn=conn)
            bar2.update(1)
        bar2.close()
        print(f"[job_update_bollinger] backfill_concept inserted concept_rows={ins_c2}")
    
    with eng.begin() as conn:
        cols_s = _table_columns(eng, "stock_bollinger_data")
        dc_s = "trade_date" if "trade_date" in cols_s else ("date" if "date" in cols_s else None)
        deleted_s = 0
        if dc_s is not None and cols_s:
            r = conn.execute(text(f"delete from stock_bollinger_data where {dc_s}=:d"), {"d": trade_date})
            deleted_s = r.rowcount or 0
        cols_c = _table_columns(eng, "concept_bollinger_data")
        dc_c = "trade_date" if "trade_date" in cols_c else ("date" if "date" in cols_c else None)
        deleted_c = 0
        if dc_c is not None and cols_c:
            r2 = conn.execute(text(f"delete from concept_bollinger_data where {dc_c}=:d"), {"d": trade_date})
            deleted_c = r2.rowcount or 0
        n1 = compute_stock_bollinger_for_date(trade_date, conn=conn)
        n2 = compute_concept_bollinger_for_date(trade_date, conn=conn)
        print(f"[job_update_bollinger] deleted stock_rows={deleted_s} concept_rows={deleted_c}")
        print(f"[job_update_bollinger] inserted stock_rows={n1} concept_rows={n2}")


def job_backfill_bollinger_from_db(z_window: int = 120) -> None:
    eng = get_engine()
    with eng.connect() as conn:
        rng = conn.execute(text("select min(trade_date) as d1, max(trade_date) as d2 from stock_daily")).mappings().first()
        df_dates = pd.read_sql("select distinct trade_date from stock_daily order by trade_date", conn)
    d1 = rng["d1"]
    d2 = rng["d2"]
    all_days = pd.to_datetime(df_dates["trade_date"]).dt.date.tolist() if not df_dates.empty else []
    print(f"[job_backfill_bollinger_from_db] range {d1}~{d2} days={len(all_days)}")
    if not all_days:
        return
    start_2y = d2 - _timedelta(days=730)
    days_2y = [d for d in all_days if d >= start_2y]
    print(f"[job_backfill_bollinger_from_db] limit to last_2y {start_2y}~{d2} days={len(days_2y)}")
    cols_s = _table_columns(eng, "stock_bollinger_data")
    dc_s = "trade_date" if "trade_date" in cols_s else ("date" if "date" in cols_s else None)
    cols_c = _table_columns(eng, "concept_bollinger_data")
    dc_c = "trade_date" if "trade_date" in cols_c else ("date" if "date" in cols_c else None)
    start_idx = (z_window - 1) if len(days_2y) >= z_window else len(days_2y)
    work_days = days_2y[start_idx:]
    print(f"[job_backfill_bollinger_from_db] start_index={start_idx} work_days={len(work_days)}")
    if not work_days:
        print("[job_backfill_bollinger_from_db] no work due to insufficient history")
        return
    bar = tqdm(total=len(work_days), desc="backfill", unit="day", mininterval=1)
    ins_s = 0
    ins_c = 0
    for d0 in work_days:
        with eng.begin() as conn:
            if dc_s is not None and cols_s:
                conn.execute(text(f"delete from stock_bollinger_data where {dc_s}=:d"), {"d": d0})
            if dc_c is not None and cols_c:
                conn.execute(text(f"delete from concept_bollinger_data where {dc_c}=:d"), {"d": d0})
            ins_s += compute_stock_bollinger_for_date(d0, conn=conn, z_window=z_window)
            ins_c += compute_concept_bollinger_for_date(d0, conn=conn)
        bar.update(1)
    bar.close()
    print(f"[job_backfill_bollinger_from_db] inserted stock_rows={ins_s} concept_rows={ins_c}")


def job_update_bollinger_range(start_date: _date, end_date: _date) -> None:
    print(f"[job_update_bollinger_range] {start_date}~{end_date}")
    from factors.bollinger import compute_stock_bollinger_between, compute_concept_bollinger_between
    n1 = compute_stock_bollinger_between(start_date, end_date)
    n2 = compute_concept_bollinger_between(start_date, end_date)
    print(f"[job_update_bollinger_range] stock_rows={n1} concept_rows={n2}")


def build_active_pool(trade_date: _date, window_days: int = 20, ppos_threshold: float = 50.0, min_avg_amount: float = 100_000_000.0) -> int:
    print(f"[build_active_pool] trade_date={trade_date} window_days={window_days}")
    eng = get_engine()
    d_start = pd.Timestamp(trade_date) - pd.Timedelta(days=window_days - 1)
    with eng.connect() as conn:
        cols_s = _table_columns(eng, "stock_bollinger_data")
        dc_s = "trade_date" if "trade_date" in cols_s else ("date" if "date" in cols_s else None)
        if not cols_s or dc_s is None:
            return 0
        # 仅当日记录，用于事件日同日量价齐升
        df_sbd_day = pd.read_sql(
            text(f"select ts_code, v_price_position, price_position from stock_bollinger_data where {dc_s}=:d"),
            conn,
            params={"d": trade_date},
        )
    if df_sbd_day.empty:
        with eng.begin() as conn:
            inspector = inspect(conn)
            if inspector.has_table("stock_active_pool", schema="public"):
                conn.execute(text("delete from stock_active_pool where trade_date=:d"), {"d": trade_date})
        print("[build_active_pool] no candidates")
        return 0
    df_sbd_day["has_both"] = (pd.to_numeric(df_sbd_day["v_price_position"], errors="coerce") > 99.0) & (
        pd.to_numeric(df_sbd_day["price_position"], errors="coerce") > float(ppos_threshold)
    )
    with eng.connect() as conn:
        df_daily = pd.read_sql(
            text("select ts_code, trade_date as d, amount from stock_daily where trade_date>=:d1 and trade_date<=:d2"),
            conn,
            params={"d1": d_start.date(), "d2": trade_date},
        )
    if df_daily.empty:
        return 0
    df_daily["d"] = pd.to_datetime(df_daily["d"])
    df_amt = df_daily.groupby("ts_code", as_index=False)["amount"].mean().rename(columns={"amount": "avg_amount"})
    df = df_sbd_day.merge(df_amt, on="ts_code", how="left")
    df["avg_amount"] = pd.to_numeric(df["avg_amount"], errors="coerce").fillna(0.0)
    # 前20交易日约束：不得出现同日 price_position>99 且 v_price_position>99
    with eng.connect() as conn:
        df_prev = pd.read_sql(
            text(f"select ts_code, price_position from stock_bollinger_data where {dc_s}>=:d1 and {dc_s}<:d2"),
            conn,
            params={"d1": d_start.date(), "d2": trade_date},
        )
    if not df_prev.empty:
        prev_ppos = (pd.to_numeric(df_prev["price_position"], errors="coerce") > 99.0)
        df_prev_flag = df_prev.assign(prev_ppos=prev_ppos).groupby("ts_code", as_index=False)["prev_ppos"].max()
        df = df.merge(df_prev_flag, on="ts_code", how="left")
        df["prev_ppos"] = df["prev_ppos"].fillna(False)
        df = df[(df["has_both"]) & (~df["prev_ppos"]) & (df["avg_amount"] > float(min_avg_amount))]
    else:
        df = df[(df["has_both"]) & (df["avg_amount"] > float(min_avg_amount))]
    if df.empty:
        with eng.begin() as conn:
            inspector = inspect(conn)
            if inspector.has_table("stock_active_pool", schema="public"):
                conn.execute(text("delete from stock_active_pool where trade_date=:d"), {"d": trade_date})
        print("[build_active_pool] no candidates")
        return 0
    df_out = df.copy()
    df_out["trade_date"] = pd.Timestamp(trade_date)
    df_out["window_days"] = int(window_days)
    df_out["remarks"] = ""
    df_out["has_high_vpos"] = True
    df_out["has_high_ppos"] = True
    df_out.rename(columns={"v_price_position": "max_vpos", "price_position": "max_ppos"}, inplace=True)
    cols = ["ts_code", "trade_date", "window_days", "has_high_vpos", "has_high_ppos", "max_vpos", "max_ppos", "avg_amount", "remarks"]
    df_out = df_out[cols].reset_index(drop=True)
    with eng.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("stock_active_pool", schema="public"):
            conn.execute(text("delete from stock_active_pool where trade_date=:d"), {"d": trade_date})
        df_out.to_sql("stock_active_pool", conn, if_exists="append", index=False)
    print(f"[build_active_pool] inserted rows={len(df_out)}")
    return len(df_out)


def summarize_microstructure_for_stock(ts_code: str, trade_date: _date, q: float, hist_window_days: int = 20, floor_min: float = 300_000.0) -> dict:
    store = TickStore()
    df_ticks = store.load_ticks(ts_code, trade_date)
    if df_ticks is None or df_ticks.empty:
        return {
            "ts_code": ts_code,
            "trade_date": pd.Timestamp(trade_date),
            "total_value": 0.0,
            "total_volume": 0,
            "vwap": 0.0,
            "big_buy_value": 0.0,
            "big_sell_value": 0.0,
            "big_net_value": 0.0,
            "big_net_ratio": 0.0,
            "big_buy_ratio": 0.0,
            "big_sell_ratio": 0.0,
            "big_net_open": 0.0,
            "big_net_mid": 0.0,
            "big_net_close": 0.0,
            "order_flow_imbalance": 0.0,
            "aggressive_trade_count_ratio": 0.0,
        }
    df = df_ticks.copy()
    dt_col = next((c for c in ["datetime", "time", "timestamp"] if c in df.columns), None)
    if dt_col is None:
        return {
            "ts_code": ts_code,
            "trade_date": pd.Timestamp(trade_date),
            "total_value": 0.0,
            "total_volume": 0,
            "vwap": 0.0,
            "big_buy_value": 0.0,
            "big_sell_value": 0.0,
            "big_net_value": 0.0,
            "big_net_ratio": 0.0,
            "big_buy_ratio": 0.0,
            "big_sell_ratio": 0.0,
            "big_net_open": 0.0,
            "big_net_mid": 0.0,
            "big_net_close": 0.0,
            "order_flow_imbalance": 0.0,
            "aggressive_trade_count_ratio": 0.0,
        }
    df["datetime"] = pd.to_datetime(df[dt_col])
    price_col = next((c for c in ["price", "last", "成交价", "new_price"] if c in df.columns), None)
    volume_col = next((c for c in ["volume", "vol", "qty", "成交量"] if c in df.columns), None)
    amount_col = next((c for c in ["amount", "turnover", "成交额"] if c in df.columns), None)
    price_series = pd.to_numeric(df[price_col], errors="coerce") if price_col is not None else pd.Series(0.0, index=df.index)
    if amount_col is not None:
        amount_series = pd.to_numeric(df[amount_col], errors="coerce")
    else:
        amount_series = pd.Series(0.0, index=df.index)
    if volume_col is not None:
        volume_series = pd.to_numeric(df[volume_col], errors="coerce")
    else:
        if (amount_col is not None) and (price_col is not None):
            volume_series = amount_series / price_series.replace(0, pd.NA)
            volume_series = pd.to_numeric(volume_series, errors="coerce").fillna(0.0)
        else:
            volume_series = pd.Series(0.0, index=df.index)
    df["trade_value"] = amount_series.where(amount_series > 0, price_series * volume_series)
    side_col = next((c for c in ["side", "buyorsell", "bsflag"] if c in df.columns), None)
    if side_col is None:
        df["side"] = "N"
    else:
        vals = df[side_col]
        if pd.api.types.is_numeric_dtype(vals):
            s = pd.to_numeric(vals, errors="coerce")
            df["side"] = s.map({0: "B", 1: "S", 2: "N"}).fillna("N")
        else:
            bs = vals.astype(str).str.upper().str.strip()
            df["side"] = bs.map({"B": "B", "S": "S", "N": "N"}).fillna(bs.apply(lambda x: "B" if x.startswith("B") else ("S" if x.startswith("S") else "N")))
    df["is_buy"] = df["side"].astype(str).str.upper().str.startswith("B")
    df["is_sell"] = df["side"].astype(str).str.upper().str.startswith("S")
    total_value = float(df["trade_value"].sum())
    total_volume = float(pd.to_numeric(volume_series, errors="coerce").sum())
    vwap = (float(df["trade_value"].sum()) / max(total_volume, 1e-9)) if total_volume > 0 else 0.0
    trade_cnt = int(len(df))
    buy_cnt = int(df["is_buy"].sum())
    sell_cnt = int(df["is_sell"].sum())
    aggressive_trade_count_ratio = ((buy_cnt - sell_cnt) / float(trade_cnt)) if trade_cnt > 0 else 0.0
    tv = pd.to_numeric(df["trade_value"], errors="coerce").fillna(0.0)
    tv_pos = tv[tv > 0]
    Qg = float(tv_pos.quantile(q)) if len(tv_pos) > 0 else 0.0
    # 历史 P90（最近 hist_window_days）
    try_hist_start = pd.Timestamp(trade_date) - pd.Timedelta(days=hist_window_days - 1)
    store = TickStore()
    idx_rows = store.list_tick_files(ts_code, try_hist_start.date(), trade_date)
    if idx_rows:
        hist_vals = []
        for r in idx_rows:
            dfh = store.load_ticks(ts_code, r.trade_date)
            if dfh is not None and not dfh.empty:
                price_h = pd.to_numeric(dfh.get("price", 0.0), errors="coerce")
                vol_h = pd.to_numeric(dfh.get("volume", 0.0), errors="coerce")
                amt_h = pd.to_numeric(dfh.get("amount", 0.0), errors="coerce")
                tv_h = amt_h.where(amt_h > 0, price_h * vol_h)
                hist_vals.append(tv_h[tv_h > 0])
        if hist_vals:
            tv_hist = pd.concat(hist_vals, ignore_index=True)
            Qhist = float(tv_hist.quantile(0.9)) if len(tv_hist) > 0 else 0.0
            Thist = Qhist
        else:
            Thist = Qg
    else:
        Thist = Qg
    is_big = tv >= Thist
    big_buy_value = float(df.loc[is_big & df["is_buy"], "trade_value"].sum())
    big_sell_value = float(df.loc[is_big & df["is_sell"], "trade_value"].sum())
    big_net_value = big_buy_value - big_sell_value
    big_net_ratio = (big_net_value / max(total_value, 1e-9)) if total_value > 0 else 0.0
    big_buy_ratio = (big_buy_value / max(total_value, 1e-9)) if total_value > 0 else 0.0
    big_sell_ratio = (big_sell_value / max(total_value, 1e-9)) if total_value > 0 else 0.0
    t = df["datetime"].dt.time
    open_win = (t >= pd.to_datetime("09:30").time()) & (t < pd.to_datetime("10:00").time())
    mid_win = (t >= pd.to_datetime("10:00").time()) & (t < pd.to_datetime("14:00").time())
    close_win = (t >= pd.to_datetime("14:00").time()) & (t <= pd.to_datetime("15:00").time())
    tv_open = tv[open_win]
    tv_mid = tv[mid_win]
    tv_close = tv[close_win]
    Q_open = float(tv_open[tv_open > 0].quantile(q)) if len(tv_open) > 0 else Qg
    Q_mid = float(tv_mid[tv_mid > 0].quantile(q)) if len(tv_mid) > 0 else Qg
    Q_close = float(tv_close[tv_close > 0].quantile(q)) if len(tv_close) > 0 else Qg
    is_big_open = (tv >= Q_open) & open_win
    is_big_mid = (tv >= Q_mid) & mid_win
    is_big_close = (tv >= Q_close) & close_win
    big_net_open = float(df.loc[is_big_open & df["is_buy"], "trade_value"].sum() - df.loc[is_big_open & df["is_sell"], "trade_value"].sum())
    big_net_mid = float(df.loc[is_big_mid & df["is_buy"], "trade_value"].sum() - df.loc[is_big_mid & df["is_sell"], "trade_value"].sum())
    big_net_close = float(df.loc[is_big_close & df["is_buy"], "trade_value"].sum() - df.loc[is_big_close & df["is_sell"], "trade_value"].sum())
    buy_value = float(df.loc[df["is_buy"], "trade_value"].sum())
    sell_value = float(df.loc[df["is_sell"], "trade_value"].sum())
    order_flow_imbalance = (buy_value - sell_value) / max(total_value, 1e-9) if total_value > 0 else 0.0
    return {
        "ts_code": ts_code,
        "trade_date": pd.Timestamp(trade_date),
        "total_value": total_value,
        "total_volume": int(total_volume),
        "vwap": vwap,
        "big_buy_value": big_buy_value,
        "big_sell_value": big_sell_value,
        "big_net_value": big_net_value,
        "big_net_ratio": big_net_ratio,
        "big_buy_ratio": big_buy_ratio,
        "big_sell_ratio": big_sell_ratio,
        "big_net_open": big_net_open,
        "big_net_mid": big_net_mid,
        "big_net_close": big_net_close,
        "order_flow_imbalance": order_flow_imbalance,
        "aggressive_trade_count_ratio": aggressive_trade_count_ratio,
    }


def build_microstructure_daily(trade_date: _date, only_active_pool: bool = True, q: float = 0.9) -> int:
    print(f"[build_microstructure_daily] trade_date={trade_date} only_active_pool={only_active_pool}")
    eng = get_engine()
    if only_active_pool:
        episodes = fetch_active_episodes(trade_date - _timedelta(days=9), trade_date)
        ts_list = [] if episodes.empty else episodes["ts_code"].astype(str).drop_duplicates().tolist()
    else:
        basics = get_all_stock_basics()
        ts_list = [b.ts_code for b in basics] if basics else []
    # 仅处理当日有 tick 的股票，避免大量 0 值
    if ts_list:
        with eng.connect() as conn:
            df_idx = pd.read_sql(
                text("select ts_code, record_cnt from tick_file_index where trade_date=:d"),
                conn,
                params={"d": trade_date},
            )
        if not df_idx.empty:
            idx_map = {str(r["ts_code"]): int(pd.to_numeric(r["record_cnt"], errors="coerce") or 0) for _, r in df_idx.iterrows()}
            ts_list = [ts for ts in ts_list if idx_map.get(ts, 0) > 0]
    if not ts_list:
        with eng.begin() as conn:
            inspector = inspect(conn)
            if inspector.has_table("stock_microstructure_daily", schema="public"):
                conn.execute(text("delete from stock_microstructure_daily where trade_date=:d"), {"d": trade_date})
        print("[build_microstructure_daily] empty universe")
        return 0
    # 按市值分组给不同 q（若有自由流通市值）
    q_map = {}
    with eng.connect() as conn:
        cols_basic = _table_columns(eng, "stock_basic")
        ff_col = None
        for c in ["free_float_value", "free_float_mv", "float_mv", "free_float_market_cap"]:
            if c in cols_basic:
                ff_col = c
                break
        if ff_col is not None and ts_list:
            df_ff = pd.read_sql(text(f"select ts_code, {ff_col} as ff from stock_basic where ts_code = any(:codes)"), conn, params={"codes": ts_list})
            if not df_ff.empty:
                for _, r in df_ff.iterrows():
                    ff = float(pd.to_numeric(r["ff"], errors="coerce") or 0.0)
                    if ff <= 5_000_000_000:
                        q_map[str(r["ts_code"])] = 0.9
                    elif ff <= 20_000_000_000:
                        q_map[str(r["ts_code"])] = 0.95
                    else:
                        q_map[str(r["ts_code"])] = 0.97
    rows = []
    bar = tqdm(total=len(ts_list), desc="micro_daily", unit="stk")
    for ts in ts_list:
        q_use = q_map.get(ts, q)
        rows.append(summarize_microstructure_for_stock(ts, trade_date, q_use))
        bar.update(1)
    bar.close()
    df_out = pd.DataFrame(rows)
    with eng.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("stock_microstructure_daily", schema="public"):
            conn.execute(text("alter table stock_microstructure_daily add column if not exists aggressive_trade_count_ratio double precision"))
            conn.execute(text("delete from stock_microstructure_daily where trade_date=:d"), {"d": trade_date})
        df_out.to_sql("stock_microstructure_daily", conn, if_exists="append", index=False)
    print(f"[build_microstructure_daily] inserted rows={len(df_out)}")
    return len(df_out)




def detect_rise_window(ts_code: str, trade_date: _date, max_lookback: int = 20, max_drawdown: float = 0.06) -> tuple[_date, _date, int]:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text("select trade_date as d, close from stock_daily where ts_code=:ts and trade_date<=:d order by trade_date desc limit :n"),
            conn,
            params={"ts": ts_code, "d": trade_date, "n": max_lookback},
        )
    if df.empty:
        return trade_date, trade_date, 1
    df = df.sort_values("d").reset_index(drop=True)
    closes = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    end_close = float(closes.iloc[-1])
    fwd_min = pd.Series(closes[::-1].cummin()[::-1].values, index=df.index)
    rr = end_close / closes
    dd = fwd_min / closes
    mask = (rr > 1.0) & (dd >= (1.0 - float(max_drawdown)))
    idxs = df.index[mask]
    start_idx = int(idxs.min()) if len(idxs) > 0 else int(max(len(df) - 5, 0))
    start_date = pd.to_datetime(df.loc[start_idx, "d"]).date()
    end_date = pd.to_datetime(df.loc[len(df) - 1, "d"]).date()
    return start_date, end_date, (len(df) - start_idx)


def detect_window_by_min_close(ts_code: str, trade_date: _date, lookback_days: int = 20) -> tuple[_date, _date, int]:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text("select trade_date as d, close from stock_daily where ts_code=:ts and trade_date<=:d order by trade_date desc limit :n"),
            conn,
            params={"ts": ts_code, "d": trade_date, "n": lookback_days},
        )
    if df.empty:
        return trade_date, trade_date, 1
    df = df.sort_values("d").reset_index(drop=True)
    closes = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    idx_min = int(closes.idxmin())
    start_date = pd.to_datetime(df.loc[idx_min, "d"]).date()
    end_date = pd.to_datetime(df.loc[len(df) - 1, "d"]).date()
    days = int(len(df) - idx_min)
    return start_date, end_date, days


def compute_fund_quality_for_window(ts_code: str, window_start: _date, window_end: _date) -> dict:
    eng = get_engine()
    with eng.connect() as conn:
        df_m = pd.read_sql(
            text("select trade_date as d, total_value, big_net_value from stock_microstructure_daily where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2 order by trade_date"),
            conn,
            params={"ts": ts_code, "d1": window_start, "d2": window_end},
        )
        df_d = pd.read_sql(
            text("select trade_date as d, close, amount from stock_daily where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2 order by trade_date"),
            conn,
            params={"ts": ts_code, "d1": window_start, "d2": window_end},
        )
        cols_basic = []
        df_basic = pd.DataFrame()
        if eng is not None:
            cols_basic = _table_columns(eng, "stock_basic")
        free_float_col = None
        for c in ["free_float_value", "free_float_mv", "float_mv", "free_float_market_cap"]:
            if c in cols_basic:
                free_float_col = c
                break
        free_float_value = 0.0
        if free_float_col is not None:
            df_basic = pd.read_sql(text(f"select {free_float_col} as ff from stock_basic where ts_code=:ts"), conn, params={"ts": ts_code})
            if not df_basic.empty:
                free_float_value = float(pd.to_numeric(df_basic["ff"], errors="coerce").fillna(0.0).iloc[0])
    if df_d.empty:
        return {
            "ts_code": ts_code,
            "window_start": pd.Timestamp(window_start),
            "window_end": pd.Timestamp(window_end),
            "window_days": 0,
            "price_return": 0.0,
            "big_net_sum": 0.0,
            "turnover_ratio": 0.0,
            "pos_days_ratio": 0.0,
            "trend_corr": 0.0,
            "big_cost_gain": float("nan"),
            "support_ratio": 0.0,
            "fund_quality_score": 0.0,
            "created_at": pd.Timestamp.utcnow(),
        }
    df_d["d"] = pd.to_datetime(df_d["d"])
    df_m["d"] = pd.to_datetime(df_m["d"]) if not df_m.empty else pd.Series([], dtype="datetime64[ns]")
    days = len(df_d)
    close = pd.to_numeric(df_d["close"], errors="coerce").ffill().bfill()
    price_return = float(close.iloc[-1] / max(close.iloc[0], 1e-9) - 1.0)
    total_value_sum = float(pd.to_numeric(df_m.get("total_value", 0.0), errors="coerce").sum()) if not df_m.empty else 0.0
    big_net_sum = float(pd.to_numeric(df_m.get("big_net_value", 0.0), errors="coerce").sum()) if not df_m.empty else 0.0
    turnover_ratio = (total_value_sum / max(free_float_value, 1e-9)) if free_float_value > 0 else 0.0
    big_net_to_float_ratio = (big_net_sum / max(free_float_value, 1e-9)) if free_float_value > 0 else 0.0
    pos_days_ratio = float((pd.to_numeric(df_m.get("big_net_value", 0.0), errors="coerce") > 0).mean()) if not df_m.empty else 0.0
    cum_big = pd.to_numeric(df_m.get("big_net_value", 0.0), errors="coerce").cumsum() if not df_m.empty else pd.Series([0.0] * days)
    # 对齐到 df_d 的日期索引
    df_join = df_d[["d"]].merge(pd.DataFrame({"d": df_m["d"], "cum_big": cum_big}), on="d", how="left").fillna(0.0)
    trend_corr = float(pd.to_numeric(df_join["cum_big"], errors="coerce").corr(pd.to_numeric(close, errors="coerce"))) if days > 1 else 0.0
    # 回调段支持率：日收益<0 的天中，big_net_value>=0 的比例
    ret = close.pct_change().fillna(0.0)
    m_map = df_m.set_index("d")["big_net_value"] if not df_m.empty else pd.Series([], dtype="float64")
    bn = df_d["d"].map(m_map).fillna(0.0)
    pull_mask = ret < 0
    support_ratio = float((bn[pull_mask] >= 0).mean()) if pull_mask.any() else 0.0
    # big_cost_gain 暂缺（需逐笔 VWAP）；置为空
    big_cost_gain = float("nan")
    # 综合分数（简版）：标准化后加权求和
    # 使用简单缩放以避免跨标的一致性问题
    corr_w = 0.2 if days >= 5 else 0.05
    score = (
        0.25 * max(min(price_return, 1.0), -1.0)
        + 0.25 * (big_net_sum / max(total_value_sum, 1e-9) if total_value_sum > 0 else 0.0)
        + 0.2 * pos_days_ratio
        + corr_w * (trend_corr if pd.notna(trend_corr) else 0.0)
        + 0.1 * support_ratio
    )
    return {
        "ts_code": ts_code,
        "window_start": pd.Timestamp(window_start),
        "window_end": pd.Timestamp(window_end),
        "window_days": days,
        "price_return": price_return,
        "big_net_sum": big_net_sum,
        "turnover_ratio": turnover_ratio,
        "big_net_to_float_ratio": big_net_to_float_ratio,
        "pos_days_ratio": pos_days_ratio,
        "trend_corr": trend_corr if pd.notna(trend_corr) else 0.0,
        "big_cost_gain": big_cost_gain,
        "support_ratio": support_ratio,
        "fund_quality_score": score,
        "created_at": pd.Timestamp.utcnow(),
    }


def build_fund_quality(trade_date: _date, max_lookback: int = 20) -> int:
    print(f"[build_fund_quality] trade_date={trade_date}")
    eng = get_engine()
    episodes = fetch_active_episodes(trade_date - _timedelta(days=9), trade_date)
    ts_list = [] if episodes.empty else episodes["ts_code"].astype(str).drop_duplicates().tolist()
    if not ts_list:
        print("[build_fund_quality] empty universe")
        return 0
    rows = []
    bar = tqdm(total=len(ts_list), desc="fund_quality", unit="stk")
    for ts in ts_list:
        ws, we, wd = detect_window_by_min_close(ts, trade_date, lookback_days=max_lookback)
        rows.append(compute_fund_quality_for_window(ts, ws, we))
        bar.update(1)
    bar.close()
    df_out = pd.DataFrame(rows)
    with eng.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("stock_fund_quality", schema="public"):
            conn.execute(text("delete from stock_fund_quality where window_end=:d"), {"d": trade_date})
            conn.execute(text("alter table stock_fund_quality add column if not exists big_net_to_float_ratio double precision"))
        df_out.to_sql("stock_fund_quality", conn, if_exists="append", index=False)
    print(f"[build_fund_quality] inserted rows={len(df_out)}")
    return len(df_out)


def fetch_active_episodes(start_date: _date, end_date: _date) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text("select ts_code, trade_date as pool_date from stock_active_pool where trade_date>=:d1 and trade_date<=:d2 order by ts_code, pool_date"),
            conn,
            params={"d1": start_date, "d2": end_date},
        )
    return df


def load_time_series_for_stock(ts_code: str, min_date: _date, max_date: _date) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as conn:
        df_daily = pd.read_sql(
            text("select ts_code, trade_date, open, high, low, close, amount from stock_daily where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2 order by trade_date"),
            conn,
            params={"ts": ts_code, "d1": min_date, "d2": max_date},
        )
    if df_daily.empty:
        return df_daily
    df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"])
    df_daily.set_index("trade_date", inplace=True)
    df_daily.sort_index(inplace=True)
    df_daily["ret_1d"] = df_daily["close"] / df_daily["close"].shift(1) - 1
    df_daily["ma_amount_20"] = df_daily["amount"].rolling(20, min_periods=1).mean()
    with eng.connect() as conn:
        df_boll = pd.read_sql(
            text("select ts_code, trade_date, price_position, band_width from stock_bollinger_data where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2"),
            conn,
            params={"ts": ts_code, "d1": min_date, "d2": max_date},
        )
    df_boll["trade_date"] = pd.to_datetime(df_boll["trade_date"])
    df_boll.set_index("trade_date", inplace=True)
    df_boll = df_boll[["price_position", "band_width"]]
    with eng.connect() as conn:
        df_micro = pd.read_sql(
            text("select ts_code, trade_date, big_net_value, total_value, big_net_open, big_net_close, order_flow_imbalance from stock_microstructure_daily where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2"),
            conn,
            params={"ts": ts_code, "d1": min_date, "d2": max_date},
        )
    if df_micro.empty:
        df_micro = pd.DataFrame(columns=["big_net_value", "total_value", "big_net_open", "big_net_close", "order_flow_imbalance"])
        df_micro.index = df_daily.index
    else:
        df_micro["trade_date"] = pd.to_datetime(df_micro["trade_date"])
        df_micro.set_index("trade_date", inplace=True)
        df_micro = df_micro[["big_net_value", "total_value", "big_net_open", "big_net_close", "order_flow_imbalance"]]
    with eng.connect() as conn:
        df_fq = pd.read_sql(
            text("select ts_code, window_end, fund_quality_score from stock_fund_quality where ts_code=:ts and window_end<=:d2 order by window_end"),
            conn,
            params={"ts": ts_code, "d2": max_date},
        )
    if not df_fq.empty:
        df_fq["window_end"] = pd.to_datetime(df_fq["window_end"])
        df_fq.set_index("window_end", inplace=True)
        df_fq = df_fq[["fund_quality_score"]]
        df_fq = df_fq.reindex(df_daily.index, method="ffill")
    else:
        df_fq = pd.DataFrame(index=df_daily.index, data={"fund_quality_score": np.nan})
    df = df_daily.join(df_boll, how="left")
    df = df.join(df_micro, how="left")
    df = df.join(df_fq, how="left")
    for col in ["big_net_value", "total_value", "big_net_open", "big_net_close", "order_flow_imbalance"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def _safe_div(a: float, b: float) -> float:
    if b is None or b == 0 or np.isnan(b):
        return 0.0
    return float(a) / float(b)


@dataclass
class PoolMLConfig:
    max_tracking_days: int = 10
    recent_window_days: int = 3


@dataclass
class BestEntryLabelConfig:
    max_entry_days: int = 5
    max_holding_days: int = 10
    take_profit: float = 0.10
    stop_loss: float = -0.05
    dd_penalty: float = 0.5
    score_threshold: float = 0.02

def simulate_trade_for_entry(price_series: pd.Series, entry_date: _date, cfg: BestEntryLabelConfig) -> dict | None:
    entry_ts = pd.Timestamp(entry_date)
    after_entry = price_series.loc[price_series.index >= entry_ts]
    if after_entry.empty or len(after_entry) < 2:
        return None
    window = after_entry.iloc[: cfg.max_holding_days + 1]
    entry_price = float(window.iloc[0])
    if entry_price <= 0:
        return None
    max_price = entry_price
    max_dd = 0.0
    exit_price = None
    for i in range(1, len(window)):
        price_t = float(window.iloc[i])
        if price_t <= 0:
            continue
        if price_t > max_price:
            max_price = price_t
        dd_t = (max_price - price_t) / max_price
        if dd_t > max_dd:
            max_dd = dd_t
        ret_t = price_t / entry_price - 1.0
        if ret_t >= cfg.take_profit or ret_t <= cfg.stop_loss:
            exit_price = price_t
            break
    if exit_price is None:
        exit_price = float(window.iloc[-1])
        if exit_price > max_price:
            max_price = exit_price
        dd_t = (max_price - exit_price) / max_price
        if dd_t > max_dd:
            max_dd = dd_t
    trade_ret = exit_price / entry_price - 1.0
    score = trade_ret - cfg.dd_penalty * max_dd
    return {"trade_ret": trade_ret, "max_dd": max_dd, "score": score}

def _load_episode_samples(engine, ts_code: str, pool_date: _date) -> pd.DataFrame:
    sql = text("""
    SELECT * FROM stock_pool_ml_samples
    WHERE ts_code = :ts_code AND pool_date = :pool_date
    ORDER BY current_date
    """)
    df = pd.read_sql(sql, con=engine, params={"ts_code": ts_code, "pool_date": pool_date}, parse_dates=["pool_date", "current_date"])
    if df.empty:
        return df
    df["pool_date"] = pd.to_datetime(df["pool_date"]).dt.date
    df["current_date"] = pd.to_datetime(df["current_date"]).dt.date
    return df

def _load_price_series_for_episode(engine, ts_code: str, pool_date: _date, cfg: BestEntryLabelConfig) -> pd.Series:
    end_date = pd.to_datetime(pool_date) + pd.Timedelta(days=cfg.max_entry_days + cfg.max_holding_days + 5)
    sql = text("""
    SELECT trade_date, close FROM stock_daily
    WHERE ts_code = :ts_code AND trade_date >= :start_date AND trade_date <= :end_date
    ORDER BY trade_date
    """)
    df = pd.read_sql(sql, con=engine, params={"ts_code": ts_code, "start_date": pool_date, "end_date": end_date.date()}, parse_dates=["trade_date"])
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    s = df.set_index("trade_date")["close"].astype(float)
    s.index = pd.DatetimeIndex(s.index)
    return s

def compute_labels_for_episode(engine, ts_code: str, pool_date: _date, cfg: BestEntryLabelConfig) -> pd.DataFrame:
    samples = _load_episode_samples(engine, ts_code, pool_date)
    if samples.empty:
        return samples
    prices = _load_price_series_for_episode(engine, ts_code, pool_date, cfg)
    samples["y_trade_ret"] = pd.NA
    samples["y_max_dd"] = pd.NA
    samples["y_score"] = pd.NA
    samples["y_is_best"] = 0
    if prices.empty:
        return samples
    mask = samples["days_since_pool"].between(0, cfg.max_entry_days)
    scores: list[tuple[int, float]] = []
    for idx, row in samples.loc[mask].iterrows():
        entry_date = pd.to_datetime(row["current_date"]).date()
        res = simulate_trade_for_entry(prices, entry_date, cfg)
        if res is None:
            continue
        samples.at[idx, "y_trade_ret"] = res["trade_ret"]
        samples.at[idx, "y_max_dd"] = res["max_dd"]
        samples.at[idx, "y_score"] = res["score"]
        scores.append((idx, float(res["score"])) )
    if scores:
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        best_idx, best_score = scores_sorted[0]
        if best_score is not None and best_score >= cfg.score_threshold:
            samples.at[best_idx, "y_is_best"] = 1
    return samples

def job_label_best_entry_for_date(pool_date: _date, cfg: BestEntryLabelConfig | None = None, batch_size: int = 200) -> None:
    eng = get_engine()
    if cfg is None:
        cfg = BestEntryLabelConfig()
    with eng.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("stock_pool_ml_samples", schema="public"):
            conn.execute(text("alter table stock_pool_ml_samples add column if not exists y_trade_ret double precision"))
            conn.execute(text("alter table stock_pool_ml_samples add column if not exists y_max_dd double precision"))
            conn.execute(text("alter table stock_pool_ml_samples add column if not exists y_score double precision"))
            conn.execute(text("alter table stock_pool_ml_samples add column if not exists y_is_best integer"))
    start_d = pool_date - _timedelta(days=9)
    sql_eps = text("""
    SELECT DISTINCT ts_code, pool_date FROM stock_pool_ml_samples
    WHERE pool_date >= :d1 AND pool_date <= :d2
    ORDER BY pool_date, ts_code
    """)
    with eng.connect() as conn:
        eps_df = pd.read_sql(sql_eps, con=conn, params={"d1": start_d, "d2": pool_date}, parse_dates=["pool_date"])
    if eps_df.empty:
        return
    eps_df["pool_date"] = pd.to_datetime(eps_df["pool_date"]).dt.date
    eps_df = eps_df.sort_values(["ts_code", "pool_date"]).drop_duplicates(subset=["ts_code"], keep="last")
    episodes = list(eps_df[["ts_code", "pool_date"]].itertuples(index=False, name=None))
    bar = tqdm(total=len(episodes), desc="label_best_entry", unit="ep")
    updates: list[dict] = []
    with eng.begin() as conn:
        upd_sql = text(
            """
            UPDATE stock_pool_ml_samples
            SET y_trade_ret = :y_trade_ret, y_max_dd = :y_max_dd, y_score = :y_score, y_is_best = :y_is_best
            WHERE ts_code = :ts_code AND pool_date = :pool_date AND "current_date" = :current_date
            """
        )
        for ts_code, pd0 in episodes:
            labeled = compute_labels_for_episode(eng, ts_code, pd0, cfg)
            if labeled.empty:
                bar.update(1)
                continue
            for _, r in labeled.iterrows():
                updates.append({
                    "y_trade_ret": (None if pd.isna(r.get("y_trade_ret")) else float(r.get("y_trade_ret"))),
                    "y_max_dd": (None if pd.isna(r.get("y_max_dd")) else float(r.get("y_max_dd"))),
                    "y_score": (None if pd.isna(r.get("y_score")) else float(r.get("y_score"))),
                    "y_is_best": int(r.get("y_is_best", 0) or 0),
                    "ts_code": str(r.get("ts_code")),
                    "pool_date": pd.to_datetime(r.get("pool_date")).date(),
                    "current_date": pd.to_datetime(r.get("current_date")).date(),
                })
                if len(updates) >= batch_size:
                    conn.execute(upd_sql, updates)
                    updates = []
            bar.update(1)
        if updates:
            conn.execute(upd_sql, updates)
    bar.close()

def compute_tier1_features_for_episode(ts_code: str, pool_date: _date, df: pd.DataFrame, cfg: PoolMLConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be DatetimeIndex")
    pool_ts = pd.to_datetime(pool_date)
    if pool_ts not in df.index:
        return pd.DataFrame()
    pool_idx = df.index.get_loc(pool_ts)
    pool_row = df.iloc[pool_idx]
    pool_price_pos = float(pool_row.get("price_position", np.nan))
    pool_rel_amount = _safe_div(float(pool_row.get("amount", 0.0)), float(pool_row.get("ma_amount_20", 0.0)))
    pool_total_value_t0 = float(pool_row.get("total_value", 0.0))
    pool_big_net_value_t0 = float(pool_row.get("big_net_value", 0.0))
    pool_big_net_ratio_t0 = _safe_div(pool_big_net_value_t0, pool_total_value_t0)
    pool_fund_quality_score = float(pool_row.get("fund_quality_score", np.nan))
    close_series = df["close"]
    ret_1d_series = df["ret_1d"]
    big_net_series = df["big_net_value"]
    total_value_series = df["total_value"]
    max_idx = min(pool_idx + cfg.max_tracking_days, len(df) - 1)
    rows = []
    for idx in range(pool_idx, max_idx + 1):
        current_ts = df.index[idx]
        current_row = df.iloc[idx]
        days_since_pool = idx - pool_idx
        close_t = float(current_row.get("close", 0.0))
        close_T0 = float(pool_row.get("close", 0.0))
        ret_since_pool = _safe_div(close_t, close_T0) - 1.0
        window_close = close_series.iloc[pool_idx : idx + 1]
        min_close_since_pool = float(window_close.min())
        max_down_since_pool = _safe_div(min_close_since_pool, close_T0) - 1.0
        window_big_net = big_net_series.iloc[pool_idx : idx + 1]
        window_total_value = total_value_series.iloc[pool_idx : idx + 1]
        big_net_sum_since_pool = float(window_big_net.sum())
        total_value_sum_since_pool = float(window_total_value.sum())
        big_net_ratio_since_pool = _safe_div(big_net_sum_since_pool, total_value_sum_since_pool)
        window_ret = ret_1d_series.iloc[pool_idx : idx + 1]
        down_mask = window_ret < 0
        down_dates = window_ret.index[down_mask]
        if len(down_dates) == 0:
            support_ratio_since_pool = 0.0
        else:
            bn_down = big_net_series.reindex(down_dates).fillna(0.0)
            support_days = (bn_down >= 0).sum()
            support_ratio_since_pool = float(support_days) / float(len(down_dates))
        ret_1d = float(current_row.get("ret_1d", np.nan))
        price_pos_t = float(current_row.get("price_position", np.nan))
        band_width_t = float(current_row.get("band_width", np.nan))
        amount_t = float(current_row.get("amount", 0.0))
        ma_amount_20_t = float(current_row.get("ma_amount_20", 0.0))
        rel_amount_t = _safe_div(amount_t, ma_amount_20_t)
        big_net_t = float(current_row.get("big_net_value", 0.0))
        total_value_t = float(current_row.get("total_value", 0.0))
        big_net_ratio_t = _safe_div(big_net_t, total_value_t)
        ofi_t = float(current_row.get("order_flow_imbalance", 0.0))
        big_net_close_t = float(current_row.get("big_net_close", 0.0))
        big_net_close_ratio_t = _safe_div(big_net_close_t, total_value_t)
        recent_window_len = cfg.recent_window_days
        recent_start_idx = max(pool_idx, idx - recent_window_len + 1)
        recent_big_net = big_net_series.iloc[recent_start_idx : idx + 1]
        recent_total_value = total_value_series.iloc[recent_start_idx : idx + 1]
        recent_big_net_sum = float(recent_big_net.sum())
        recent_total_value_sum = float(recent_total_value.sum())
        big_net_ratio_recent = _safe_div(recent_big_net_sum, recent_total_value_sum)
        recent_ret = ret_1d_series.iloc[recent_start_idx : idx + 1]
        recent_down_dates = recent_ret.index[recent_ret < 0]
        if len(recent_down_dates) == 0:
            support_ratio_recent = 0.0
        else:
            bn_recent_down = big_net_series.reindex(recent_down_dates).fillna(0.0)
            recent_support_days = (bn_recent_down >= 0).sum()
            support_ratio_recent = float(recent_support_days) / float(len(recent_down_dates))
        rows.append({
            "ts_code": ts_code,
            "pool_date": pool_ts.date(),
            "current_date": current_ts.date(),
            "days_since_pool": days_since_pool,
            "ret_since_pool": ret_since_pool,
            "max_down_since_pool": max_down_since_pool,
            "big_net_ratio_since_pool": big_net_ratio_since_pool,
            "support_ratio_since_pool": support_ratio_since_pool,
            "ret_1d": ret_1d,
            "price_pos_t": price_pos_t,
            "band_width_t": band_width_t,
            "rel_amount_t": rel_amount_t,
            "big_net_ratio_t": big_net_ratio_t,
            "ofi_t": ofi_t,
            "big_net_close_ratio_t": big_net_close_ratio_t,
            "pool_price_pos": pool_price_pos,
            "pool_rel_amount": pool_rel_amount,
            "pool_big_net_ratio_t0": pool_big_net_ratio_t0,
            "pool_fund_quality_score": pool_fund_quality_score,
            "big_net_ratio_recent": big_net_ratio_recent,
            "support_ratio_recent": support_ratio_recent,
        })
    return pd.DataFrame(rows)


def build_pool_ml_samples(start_date: _date, end_date: _date, max_tracking_days: int = 10) -> int:
    eng = get_engine()
    cfg = PoolMLConfig(max_tracking_days=max_tracking_days, recent_window_days=3)
    date_list = pd.date_range(start=start_date, end=end_date).date.tolist()
    bar_days = tqdm(total=len(date_list), desc="ml_samples_days", unit="day")
    total_n = 0
    for cur_d in date_list:
        episodes = fetch_active_episodes(cur_d - _timedelta(days=max_tracking_days), cur_d)
        if not episodes.empty:
            episodes = episodes.sort_values(["ts_code", "pool_date"]).drop_duplicates(subset=["ts_code"], keep="last")
        if episodes.empty:
            bar_days.update(1)
            continue
        out_rows: list[pd.DataFrame] = []
        bar_ep = tqdm(total=len(episodes), desc=f"episodes@{cur_d}", unit="ep", leave=False)
        for _, r in episodes.iterrows():
            ts_code = str(r["ts_code"]) 
            pool_date = pd.to_datetime(r["pool_date"]).date()
            df = load_time_series_for_stock(ts_code, pool_date, cur_d)
            if df.empty:
                bar_ep.update(1)
                continue
            feat_df = compute_tier1_features_for_episode(ts_code, pool_date, df, cfg)
            if feat_df is not None and not feat_df.empty:
                feat_df["current_date"] = pd.to_datetime(feat_df["current_date"]).dt.date
                sel = feat_df[feat_df["current_date"] == cur_d]
                if not sel.empty:
                    out_rows.append(sel)
            bar_ep.update(1)
        bar_ep.close()
        if out_rows:
            day_df = pd.concat(out_rows, ignore_index=True)
            day_df["current_date"] = pd.to_datetime(day_df["current_date"]).dt.date
            day_df = day_df.sort_values(["ts_code", "pool_date", "current_date"]).drop_duplicates(subset=["ts_code", "current_date"], keep="last")
            with eng.begin() as conn:
                inspector = inspect(conn)
                if not inspector.has_table("stock_pool_ml_features", schema="public"):
                    conn.execute(text(
                        """
                        create table if not exists public.stock_pool_ml_features (
                            ts_code text not null,
                            pool_date date not null,
                            current_date date not null,
                            days_since_pool integer,
                            ret_since_pool double precision,
                            max_down_since_pool double precision,
                            big_net_ratio_since_pool double precision,
                            support_ratio_since_pool double precision,
                            ret_1d double precision,
                            price_pos_t double precision,
                            band_width_t double precision,
                            rel_amount_t double precision,
                            big_net_ratio_t double precision,
                            ofi_t double precision,
                            big_net_close_ratio_t double precision,
                            pool_price_pos double precision,
                            pool_rel_amount double precision,
                            pool_big_net_ratio_t0 double precision,
                            pool_fund_quality_score double precision,
                            big_net_ratio_recent double precision,
                            support_ratio_recent double precision,
                            primary key (ts_code, current_date)
                        )
                        """
                    ))
                conn.execute(text("delete from stock_pool_ml_features where \"current_date\"=:d"), {"d": cur_d})
                day_df.to_sql("stock_pool_ml_features", conn, if_exists="append", index=False)
            total_n += len(day_df)
        bar_days.update(1)
    bar_days.close()
    return total_n
 
    
def job_label_forward_returns_for_range(start_date: _date, end_date: _date, forward_days: int = 10) -> int:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("alter table stock_bollinger_data add column if not exists y_ret_10d double precision"))
        conn.execute(text("alter table stock_bollinger_data add column if not exists y_ret_max_10d double precision"))
        conn.execute(text("alter table stock_bollinger_data add column if not exists y_ret_min_10d double precision"))
        conn.execute(text("create index if not exists idx_sbd_ts_date on public.stock_bollinger_data(ts_code, trade_date)"))
        conn.execute(text("create index if not exists idx_sd_ts_date on public.stock_daily(ts_code, trade_date)"))
    with eng.connect() as conn:
        df_days = pd.read_sql(
            text(
                "select distinct trade_date from stock_bollinger_data where trade_date>=:d1 and trade_date<=:d2 order by trade_date"
            ),
            conn,
            params={"d1": start_date, "d2": end_date},
            parse_dates=["trade_date"],
        )
    if df_days.empty:
        return 0
    df_days["trade_date"] = pd.to_datetime(df_days["trade_date"]).dt.date
    total_updates = 0
    bar_days = tqdm(total=len(df_days), desc="label_forward_days", unit="day")
    for _, rday in df_days.iterrows():
        d = rday["trade_date"]
        d2q = pd.to_datetime(d) + pd.Timedelta(days=forward_days * 6)
        d2q = d2q.date()
        sql = f"""
        select sd.ts_code,
               sd.trade_date,
               sd.close as c0,
               lead(sd.close, {forward_days}) over (partition by sd.ts_code order by sd.trade_date) as cN,
               max(sd.close) over (partition by sd.ts_code order by sd.trade_date rows between 1 following and {forward_days} following) as maxf,
               min(sd.close) over (partition by sd.ts_code order by sd.trade_date rows between 1 following and {forward_days} following) as minf
        from stock_daily sd
        where sd.trade_date>=:d and sd.trade_date<=:d2
          and sd.ts_code in (select ts_code from stock_bollinger_data where trade_date=:d)
        order by sd.ts_code, sd.trade_date
        """
        with eng.connect() as conn:
            df = pd.read_sql(text(sql), conn, params={"d": d, "d2": d2q}, parse_dates=["trade_date"])
        if df.empty:
            bar_days.update(1)
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        if "cN" in df.columns:
            df_day = df[df["trade_date"] == d].copy()
            df_day["y_ret_10d"] = pd.to_numeric(df_day["cN"], errors="coerce") / pd.to_numeric(df_day["c0"], errors="coerce") - 1.0
            df_day["y_ret_max_10d"] = pd.to_numeric(df_day["maxf"], errors="coerce") / pd.to_numeric(df_day["c0"], errors="coerce") - 1.0
            df_day["y_ret_min_10d"] = pd.to_numeric(df_day["minf"], errors="coerce") / pd.to_numeric(df_day["c0"], errors="coerce") - 1.0
            out = df_day[["ts_code", "trade_date", "y_ret_10d", "y_ret_max_10d", "y_ret_min_10d"]].copy()
        else:
            def _calc(g: pd.DataFrame) -> pd.DataFrame:
                c = pd.to_numeric(g["c0"], errors="coerce")
                y10 = c.shift(-forward_days) / c - 1.0
                cf = c.shift(-1)
                maxf = cf.iloc[::-1].rolling(forward_days).max().iloc[::-1]
                minf = cf.iloc[::-1].rolling(forward_days).min().iloc[::-1]
                return pd.DataFrame({
                    "ts_code": g["ts_code"],
                    "trade_date": g["trade_date"],
                    "y_ret_10d": y10,
                    "y_ret_max_10d": maxf / c - 1.0,
                    "y_ret_min_10d": minf / c - 1.0,
                })
            out_all = df.groupby("ts_code", sort=False).apply(_calc).reset_index(drop=True)
            out = out_all[out_all["trade_date"] == d].copy()
        rows = out.to_dict(orient="records")
        for rr in rows:
            rr["ts_code"] = str(rr["ts_code"]) 
            rr["trade_date"] = pd.to_datetime(rr["trade_date"]).date()
            rr["y_ret_10d"] = None if pd.isna(rr.get("y_ret_10d")) else float(rr.get("y_ret_10d"))
            rr["y_ret_max_10d"] = None if pd.isna(rr.get("y_ret_max_10d")) else float(rr.get("y_ret_max_10d"))
            rr["y_ret_min_10d"] = None if pd.isna(rr.get("y_ret_min_10d")) else float(rr.get("y_ret_min_10d"))
        if rows:
            with eng.begin() as conn:
                conn.execute(
                    text(
                        "create temporary table tmp_forward_labels (ts_code text, trade_date date, y_ret_10d double precision, y_ret_max_10d double precision, y_ret_min_10d double precision) on commit drop"
                    )
                )
                out.to_sql("tmp_forward_labels", conn, if_exists="append", index=False)
                conn.execute(
                    text(
                        """
                        update stock_bollinger_data s
                        set y_ret_10d=t.y_ret_10d,
                            y_ret_max_10d=t.y_ret_max_10d,
                            y_ret_min_10d=t.y_ret_min_10d
                        from tmp_forward_labels t
                        where s.ts_code=t.ts_code and s.trade_date=t.trade_date
                        """
                    )
                )
            total_updates += len(rows)
        bar_days.update(1)
    bar_days.close()
    return total_updates

def job_label_forward_returns_auto(forward_days: int = 10) -> int:
    eng = get_engine()
    with eng.connect() as conn:
        rng = conn.execute(text("select min(trade_date) as d1, max(trade_date) as d2 from stock_bollinger_data")).mappings().first()
    d1 = rng["d1"]
    d2 = rng["d2"]
    if d1 is None or d2 is None:
        return 0
    return job_label_forward_returns_for_range(d1, d2, forward_days=forward_days)

def build_bollinger_training_dataset(start_date: _date, end_date: _date, window_L: int = 20) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text(
                """
                select ts_code, trade_date, band_width_zscore, v_band_width_zscore, price_position, v_price_position, y_ret_10d
                from stock_bollinger_data
                where trade_date>=:d1 and trade_date<=:d2
                order by ts_code, trade_date
                """
            ),
            conn,
            params={"d1": start_date, "d2": end_date},
            parse_dates=["trade_date"],
        )
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    def _feat(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("trade_date").reset_index(drop=True)
        bw = pd.to_numeric(g["band_width_zscore"], errors="coerce")
        vbw = pd.to_numeric(g["v_band_width_zscore"], errors="coerce")
        pp = pd.to_numeric(g["price_position"], errors="coerce")
        vpp = pd.to_numeric(g["v_price_position"], errors="coerce")
        y = pd.to_numeric(g["y_ret_10d"], errors="coerce")
        def slope5(s: pd.Series) -> pd.Series:
            x = np.arange(len(s))
            m = s.rolling(5).apply(lambda z: float(((x[-5:] * z).sum() - x[-5:].mean() * z.sum()) / ((x[-5:] ** 2).sum() - x[-5:].mean() * x[-5:].sum())), raw=False)
            return m
        def slope3(s: pd.Series) -> pd.Series:
            x = np.arange(len(s))
            m = s.rolling(3).apply(lambda z: float(((x[-3:] * z).sum() - x[-3:].mean() * z.sum()) / ((x[-3:] ** 2).sum() - x[-3:].mean() * x[-3:].sum())), raw=False)
            return m
        df_out = pd.DataFrame({
            "ts_code": g["ts_code"],
            "trade_date": g["trade_date"],
            "y_ret_10d": y,
            "bw_min": bw.rolling(window_L).min(),
            "bw_max": bw.rolling(window_L).max(),
            "bw_mean": bw.rolling(window_L).mean(),
            "bw_std": bw.rolling(window_L).std(ddof=0),
            "bw_slope_5": slope5(bw),
            "vbw_min": vbw.rolling(window_L).min(),
            "vbw_max": vbw.rolling(window_L).max(),
            "vbw_mean": vbw.rolling(window_L).mean(),
            "vbw_std": vbw.rolling(window_L).std(ddof=0),
            "vbw_slope_5": slope5(vbw),
            "pp_min": pp.rolling(window_L).min(),
            "pp_max": pp.rolling(window_L).max(),
            "pp_mean": pp.rolling(window_L).mean(),
            "pp_std": pp.rolling(window_L).std(ddof=0),
            "pp_slope_3": slope3(pp),
            "pp_slope_5": slope5(pp),
            "vpp_min": vpp.rolling(window_L).min(),
            "vpp_max": vpp.rolling(window_L).max(),
            "vpp_mean": vpp.rolling(window_L).mean(),
            "vpp_std": vpp.rolling(window_L).std(ddof=0),
            "vpp_slope_3": slope3(vpp),
            "vpp_slope_5": slope5(vpp),
        })
        df_out = df_out.dropna(subset=["y_ret_10d"]) 
        return df_out
    feats = df.groupby("ts_code", sort=False).apply(_feat).reset_index(drop=True)
    return feats

def job_backfill_labels_upto_date(target_date: _date, forward_days: int = 10) -> int:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("create index if not exists idx_sbd_trade_date on public.stock_bollinger_data(trade_date)"))
    with eng.connect() as conn:
        cols = pd.read_sql(
            text("select column_name from information_schema.columns where table_schema='public' and table_name='stock_bollinger_data'"),
            conn,
        )
    need_cols = {"y_ret_10d", "y_ret_max_10d", "y_ret_min_10d"}
    have_cols = set(cols["column_name"].tolist()) if not cols.empty else set()
    to_add = list(need_cols - have_cols)
    if to_add:
        with eng.begin() as conn:
            if "y_ret_10d" in to_add:
                conn.execute(text("alter table stock_bollinger_data add column if not exists y_ret_10d double precision"))
            if "y_ret_max_10d" in to_add:
                conn.execute(text("alter table stock_bollinger_data add column if not exists y_ret_max_10d double precision"))
            if "y_ret_min_10d" in to_add:
                conn.execute(text("alter table stock_bollinger_data add column if not exists y_ret_min_10d double precision"))
    with eng.connect() as conn:
        days_all = pd.read_sql(
            text("select distinct trade_date from stock_bollinger_data where trade_date<=:d order by trade_date"),
            conn,
            params={"d": target_date},
            parse_dates=["trade_date"],
        )
    if days_all.empty:
        return 0
    days_all["trade_date"] = pd.to_datetime(days_all["trade_date"]).dt.date
    missing_days = []
    with eng.connect() as conn:
        bar_scan = tqdm(total=len(days_all), desc="scan_missing_days", unit="day")
        for _, r in days_all.iterrows():
            d = r["trade_date"]
            df1 = pd.read_sql(
                text("select 1 as x from stock_bollinger_data where trade_date=:d and (y_ret_10d is null or y_ret_max_10d is null or y_ret_min_10d is null) limit 1"),
                conn,
                params={"d": d},
            )
            if not df1.empty:
                missing_days.append(d)
            bar_scan.update(1)
        bar_scan.close()
    df_days = pd.DataFrame({"trade_date": missing_days})
    if df_days.empty:
        return 0
    total = 0
    bar = tqdm(total=len(df_days), desc="backfill_labels_days", unit="day")
    for _, r in df_days.iterrows():
        d = r["trade_date"]
        total += job_label_forward_returns_for_range(d, d, forward_days=forward_days)
        bar.update(1)
    bar.close()
    return total
def job_clear_bollinger_labels_before_date(target_date: _date) -> int:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("create index if not exists idx_sbd_trade_date on public.stock_bollinger_data(trade_date)"))
    with eng.connect() as conn:
        days = pd.read_sql(
            text("select distinct trade_date from stock_bollinger_data where trade_date<:d order by trade_date"),
            conn,
            params={"d": target_date},
            parse_dates=["trade_date"],
        )
    if days.empty:
        return 0
    days["trade_date"] = pd.to_datetime(days["trade_date"]).dt.date
    total = 0
    bar = tqdm(total=len(days), desc="clear_labels_days", unit="day")
    for _, r in days.iterrows():
        d = r["trade_date"]
        with eng.begin() as conn:
            res = conn.execute(
                text("update stock_bollinger_data set y_ret_10d=null, y_ret_max_10d=null, y_ret_min_10d=null where trade_date=:d"),
                {"d": d},
            )
            total += res.rowcount or 0
        bar.update(1)
    bar.close()
    return total

if __name__ == "__main__":
    from datetime import date as date_cls
    import pandas as pd
    print("请选择功能：")
    print("1 增量更新日线")
    print("2 运行当日布林更新")
    print("3 回补布林两年窗口")
    print("4 回补标签至目标日")
    print("5 清空Bollinger标签")
    print("6 构建/补齐Tick特征")
    choice = input("输入选项编号：").strip()
    if choice == "1":
        ds = input("输入更新日期(YYYY-MM-DD，留空为今天)：").strip()
        d = date_cls.today() if ds == "" else pd.to_datetime(ds).date()
        job_update_ohlc(d)
    elif choice == "2":
        ds = input("输入测试日期(YYYY-MM-DD，留空为今天)：").strip()
        d = date_cls.today() if ds == "" else pd.to_datetime(ds).date()
        job_update_bollinger(d)
    elif choice == "3":
        ws = input("zscore窗口(默认120)：").strip()
        z = 120 if ws == "" else int(ws)
        job_backfill_bollinger_from_db(z_window=z)
    elif choice == "4":
        ds = input("目标日期(YYYY-MM-DD)：").strip()
        fd = input("前向天数(默认10)：").strip()
        d = pd.to_datetime(ds).date()
        f = 10 if fd == "" else int(fd)
        job_backfill_labels_upto_date(d, forward_days=f)
    elif choice == "5":
        ds = input("目标日期(YYYY-MM-DD)：").strip()
        dts = pd.to_datetime(ds, errors="coerce")
        if pd.isna(dts):
            print("无效日期格式：YYYY-MM-DD")
        else:
            d = dts.date()
            n = job_clear_bollinger_labels_before_date(d)
            print(f"已清空至目标日前标签行数={n}")
    elif choice == "6":
        n = job_build_missing_tick_daily_features_by_stock()
        if n == 0:
            print("未发现缺失或未找到tick文件")
        else:
            print(f"已补齐缺失Tick特征行数={n}")
    else:
        print("无效选项")
