# data/jobs.py
"""
定时任务入口：

- job_update_ohlc: 盘前/盘后更新日线 & 分钟线
- job_collect_ticks_once: 测试用 tick 抓取
- job_finalize_ticks_and_levels: 收盘后（或盘前）计算关键位
"""

from __future__ import annotations

from datetime import date as _date
from datetime import timedelta as _timedelta

from .updater import update_daily_bars, update_minute_bars, collect_full_day_ticks
from .concepts_cache import update_concepts_cache, update_hk_industry_cache, validate_concepts_cache_count
from factors.bollinger import compute_stock_bollinger_for_date, compute_concept_bollinger_for_date, _table_columns, compute_stock_bollinger_from_db_range, compute_concept_bollinger_from_db_range
from db.connection import get_engine
from sqlalchemy import text
from .repository import get_all_stock_basics
from factors import AbuPriceLevelProvider
import pandas as pd
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm


def job_update_ohlc(trade_date: _date) -> None:
    print(f"[job_update_ohlc] trade_date={trade_date}")
    update_daily_bars(trade_date=trade_date, count=600)


def job_collect_full_day_ticks(trade_date: _date) -> None:
    print(f"[job_collect_full_day_ticks] trade_date={trade_date}")
    collect_full_day_ticks(trade_date)


def job_finalize_ticks_and_levels(trade_date: _date) -> None:
    print(f"[job_finalize_ticks_and_levels] trade_date={trade_date}")
    provider = AbuPriceLevelProvider()
    provider.precompute(trade_date)


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
    if eng is not None:
        with eng.connect() as conn:
            df_daily = pd.read_sql("select distinct trade_date from stock_daily", conn)
            daily_dates = set(pd.to_datetime(df_daily["trade_date"]).dt.date.tolist()) if not df_daily.empty else set()
            cols_s = _table_columns(eng, "stock_bollinger_data")
            dc_s = "trade_date" if "trade_date" in cols_s else ("date" if "date" in cols_s else None)
            sbd_dates = set()
            if dc_s is not None and cols_s:
                df_s = pd.read_sql(f"select distinct {dc_s} as d from stock_bollinger_data", conn)
                sbd_dates = set(pd.to_datetime(df_s["d"]).dt.date.tolist()) if not df_s.empty else set()
        missing_hist = sorted(daily_dates - sbd_dates - {trade_date})
        if missing_hist:
            print(f"[job_update_bollinger] backfill_missing_days={len(missing_hist)}")
            with eng.begin() as conn:
                bar = tqdm(total=len(missing_hist), desc="backfill", unit="day")
                ins_s = 0
                ins_c = 0
                for d0 in missing_hist:
                    ins_s += compute_stock_bollinger_for_date(d0, conn=conn)
                    ins_c += compute_concept_bollinger_for_date(d0, conn=conn)
                    bar.update(1)
                bar.close()
            print(f"[job_update_bollinger] backfill inserted stock_rows={ins_s} concept_rows={ins_c}")
    if eng is None:
        n1 = compute_stock_bollinger_for_date(trade_date)
        n2 = compute_concept_bollinger_for_date(trade_date)
        print(f"[job_update_bollinger] price_volume_ok stock_rows={n1} concept_rows={n2}")
        return
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
    if eng is None:
        return
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


if __name__ == "__main__":
    from datetime import date as date_cls
    import os
    os.environ["USE_REAL_DB"] = "1"
    today = date_cls.today()
    # job_update_ohlc(today)
    # job_collect_full_day_ticks(today)
    # job_finalize_ticks_and_levels(today)
    # job_update_concepts()
    # job_update_bollinger(today)
    job_backfill_bollinger_from_db()
    
