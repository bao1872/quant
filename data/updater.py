# data/updater.py
"""
数据增量更新逻辑。

职责：
- 使用 DataSource（当前为 PytdxDataSource）拉取最新行情
- 利用 repository 写入 Postgres
- 提供按天更新的函数，供 jobs 调度或手动调用

策略：
- 每次更新日线：
  - 对每只股票：从数据源取最近 N 条日线
  - 与 DB 中最后 trade_date 比较，只插入更晚的部分（通过“删后插”方式实现）；
- 分钟线同理。
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from db.models import StockBasic
from db.connection import get_session
from .pytdx_source import PytdxDataSource
from . import repository
from config import STOCK_POOL_LIMIT, TICK_COUNT_LIMIT, Settings, KLINE_FREQS, KLINE_HISTORY_DAYS

_BASICS_CACHE: List[str] | None = None


def _get_all_stock_codes(settings: Optional[Settings] = None) -> List[str]:
    global _BASICS_CACHE
    if _BASICS_CACHE is None:
        basics = repository.get_all_stock_basics()
        _BASICS_CACHE = [s.ts_code for s in basics]
    codes = _BASICS_CACHE
    limit = settings.stock_pool_limit if settings is not None else STOCK_POOL_LIMIT
    if limit is not None and codes is not None:
        return codes[:limit]
    return codes or []


def update_daily_bars(
    trade_date: date,
    count: int = 500,
    settings: Optional[Settings] = None,
) -> None:
    """
    更新所有股票的最近 count=500 个交易日的日线数据，统一写入 stock_kline。

    - 对每个 ts_code：从 pytdx 获取最近 500 根日线（对应最近 500 个有成交的交易日）。
    - 采用“删后插”批量 upsert 到 stock_kline（通过 repository.upsert_stock_kline）。
    - 不再向旧表 stock_daily 写入。
    """
    ts_codes = _get_all_stock_codes(settings)
    if not ts_codes:
        print("[update_daily_bars] No stock_basic records found.")
        return

    print(f"[update_daily_bars] Start for {len(ts_codes)} stocks, trade_date={trade_date}")

    RAW_COUNT = 2000
    with PytdxDataSource() as ds:
        for ts_code in tqdm(ts_codes, desc="daily", unit="stk"):
            df_raw = ds.get_daily_bars(ts_code, count=RAW_COUNT)
            if df_raw.empty:
                print(f"WARNING: {ts_code} daily bars empty")
                continue
            df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
            last_dates = sorted(df_raw["datetime"].dt.date.unique())[-count:]
            df_500 = df_raw[df_raw["datetime"].dt.date.isin(last_dates)].copy()
            if len(last_dates) < 400:
                print(f"WARNING: {ts_code} only {len(last_dates)} daily trading days (<400)")
            repository.upsert_stock_kline(ts_code, "1d", df_500)

    print("[update_daily_bars] Done.")


def update_minute_bars(
    trade_date: date,
    freq: str = "1m",
    count: int = 240,
    settings: Optional[Settings] = None,
) -> None:
    """
    更新所有股票的分钟线（如 1 分钟）。

    简化逻辑与日线类似，注意分钟线量大，可按需限制股票池。
    """
    ts_codes = _get_all_stock_codes(settings)
    if not ts_codes:
        print("[update_minute_bars] No stock_basic records found.")
        return

    print(
        f"[update_minute_bars] Start for {len(ts_codes)} stocks, "
        f"trade_date={trade_date}, freq={freq}"
    )

    with PytdxDataSource() as ds:
        for ts_code in tqdm(ts_codes, desc="minute", unit="stk"):
            df = ds.get_minute_bars(ts_code, freq=freq, count=count)
            if df.empty:
                continue
            # repository.upsert_stock_minute(ts_code, df, freq=freq)
            repository.upsert_stock_kline(ts_code, freq, df)

    print("[update_minute_bars] Done.")


def collect_intraday_ticks(
    trade_date: date,
    ts_codes: List[str],
    count: int = 2000,
    settings: Optional[Settings] = None,
) -> None:
    """
    盘中 tick 收集框架（当前只是简化版一次拉取）：
    - 对传入股票列表：从 pytdx 取最近 count 条 tick，写入 TickStore。
    - 后续实盘可以改成循环调用该函数或增加“追加写临时文件”的模式。
    """
    from .tick_store import TickStore

    store = TickStore()
    print(
        f"[collect_intraday_ticks] Start for {len(ts_codes)} stocks, "
        f"trade_date={trade_date}"
    )

    with PytdxDataSource() as ds:
        for ts_code in tqdm(ts_codes, desc="ticks", unit="stk"):
            tick_limit = (settings.tick_count_limit if settings is not None else TICK_COUNT_LIMIT) or count
            df_tick = ds.get_ticks(ts_code, trade_date=trade_date, count=tick_limit)
            if df_tick.empty:
                continue
            store.save_ticks(ts_code, trade_date, df_tick, already_sorted=True)

    print("[collect_intraday_ticks] Done.")


if __name__ == "__main__":
    # 自测：假设 stock_basic 已经有一些股票（比如 000001.SZ），尝试更新其中日线数据。
    from datetime import date as date_cls

    today = date_cls.today()
    print("[updater] self test, only run daily update for safety...")

    # 为了安全起见，这里只更新日线，不去更新分钟和 tick（避免短时间大量请求）
    update_daily_bars(trade_date=today, count=50)

def update_stock_basic(settings: Optional[Settings] = None) -> int:
    with PytdxDataSource() as ds:
        df_all = ds.fetch_all_stock_list()
    if df_all.empty:
        return 0
    df_all["code"] = df_all["code"].astype(str)
    df_all["exchange"] = df_all["exchange"].astype(str)
    df_all["name"] = df_all["name"].astype(str)
    name_upper = df_all["name"].str.upper()
    is_st = name_upper.str.contains("ST")
    is_sz = df_all["exchange"].str.upper().eq("SZ")
    is_sh = df_all["exchange"].str.upper().eq("SH")
    sz_code = df_all["code"].str.slice(0, 3)
    sh_code = df_all["code"].str.slice(0, 3)
    sz_ok = is_sz & (sz_code.isin(["000", "001", "002", "003", "004", "300", "301"]))
    sh_ok = is_sh & (sh_code.isin(["600", "601", "603", "605", "688"]))
    df_filt = df_all[(sz_ok | sh_ok) & (~is_st)].copy()
    n = repository.upsert_stock_basic(df_filt)
    return n

def collect_full_day_ticks(trade_date: date, settings: Optional[Settings] = None, ts_codes: Optional[List[str]] = None) -> None:
    ts_codes = ts_codes or _get_all_stock_codes(settings)
    from .tick_store import TickStore
    store = TickStore(settings=settings)
    store.defer_index = True
    print(f"[collect_full_day_ticks] Start for {len(ts_codes)} stocks, trade_date={trade_date}")
    with PytdxDataSource(enable_fallback=False) as ds:
        for ts_code in tqdm(ts_codes, desc="ticks_full", unit="stk"):
            df_tick = ds.get_ticks_full_day(ts_code, trade_date)
            if df_tick.empty:
                continue
            store.save_ticks(ts_code, trade_date, df_tick, already_sorted=True)
    print("[collect_full_day_ticks] Done.")
    # 本地存储模式：不刷新数据库索引


def update_kline_for_universe(
    trade_date: date,
    freqs: List[str],
    history_days: int = KLINE_HISTORY_DAYS,
    settings: Optional[Settings] = None,
) -> None:
    """
    对股票池中的所有股票，在给定 freqs 上更新 K 线数据，写入 stock_kline。
    第一次运行时，从 trade_date 往前 history_days 天起补齐所有 freq；之后每天运行时，只补齐缺失的部分。
    """
    ts_codes = _get_all_stock_codes(settings)
    if not ts_codes:
        print("[update_kline_for_universe] No stock_basic records found.")
        return
    start_date_target = pd.to_datetime(trade_date) - pd.Timedelta(days=history_days)
    start_date_target = start_date_target.date()
    with PytdxDataSource() as ds:
        for ts_code in tqdm(ts_codes, desc="kline", unit="stk"):
            base_start, base_end = repository.get_kline_date_range(ts_code, "1d")
            if base_start is None or base_end is None:
                df_day = ds.get_kline(ts_code, "1d", count=history_days + 20)
                repository.upsert_stock_kline(ts_code, "1d", df_day)
                base_start, base_end = repository.get_kline_date_range(ts_code, "1d")
            for fq in freqs:
                if fq == "1d":
                    continue
                cur_start, cur_end = repository.get_kline_date_range(ts_code, fq)
                target_start = base_start or start_date_target
                target_end = trade_date
                missing = repository.get_kline_missing_dates(ts_code, fq, target_start, target_end, trade_calendar=None)
                if missing:
                    s = min(missing)
                    e = max(missing)
                    df = ds.get_kline(ts_code, fq, start=s, end=e, count=None)
                    if not df.empty:
                        repository.upsert_stock_kline(ts_code, fq, df)

def update_minute_kline_for_universe(
    trade_date: date,
    freqs: List[str],
    settings: Optional[Settings] = None,
) -> None:
    """
    对股票池在指定分钟 freqs 上做“先查缺口再回补”，使分钟线时间范围与最近 500 个有成交的日线交易日保持一致。

    逻辑：
    - 取 freq='1d' 最近 500 个且 volume>0 的交易日作为 trading_dates（升序）。
    - 对每个分钟 freq：从 stock_kline 查询已有 distinct trade_date，计算缺失集合。
    - 若存在缺失，按缺失集合的最小/最大日期范围从 pytdx 拉取分钟线并写入 stock_kline。
    """
    ts_codes = _get_all_stock_codes(settings)
    if not ts_codes:
        print("[update_minute_kline_for_universe] No stock_basic records found.")
        return
    BARS_PER_DAY = {"60m": 4, "30m": 8, "15m": 16}
    with PytdxDataSource() as ds:
        for ts_code in tqdm(ts_codes, desc="minute_univ", unit="stk"):
            trading_dates = repository.get_recent_trading_dates(ts_code, limit=500)
            if not trading_dates:
                print(f"WARNING: {ts_code} no recent trading dates for 1d")
                continue
            start_date = min(trading_dates)
            end_date = max(trading_dates)
            target_days = len(trading_dates)
            for fq in freqs:
                if fq == "1d":
                    continue
                if fq not in BARS_PER_DAY:
                    raise ValueError(f"不支持的分钟频率: {fq}")
                bars_per_day = BARS_PER_DAY[fq]
                count_needed = target_days * bars_per_day + bars_per_day * 5
                df_db = repository.get_recent_kline(ts_code, fq, count_needed)
                missing_dates: List[date] = []
                if not df_db.empty:
                    df_db = df_db.sort_values("datetime").reset_index(drop=True)
                    df_db["trade_date"] = pd.to_datetime(df_db["datetime"]).dt.date
                    cnt = df_db.groupby("trade_date")["datetime"].nunique()
                    missing_dates = [d for d in trading_dates if int(cnt.get(d, 0) or 0) < bars_per_day]
                else:
                    missing_dates = trading_dates
                if not missing_dates:
                    continue
                s = min(missing_dates)
                e = max(missing_dates)
                df_raw = ds.get_minute_bars(ts_code, freq=fq, count=count_needed)
                if df_raw.empty:
                    print(f"WARNING: {ts_code} minute {fq} bars empty")
                    continue
                df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
                df_raw["trade_date"] = pd.to_datetime(df_raw["datetime"]).dt.date
                mask = (df_raw["trade_date"] >= s) & (df_raw["trade_date"] <= e)
                df_window = df_raw.loc[mask].copy()
                df_window = df_window[df_window["trade_date"].isin(missing_dates)].copy()
                repository.upsert_stock_kline(ts_code, fq, df_window)
