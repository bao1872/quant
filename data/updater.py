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
from config import STOCK_POOL_LIMIT, TICK_COUNT_LIMIT, Settings


def _get_all_stock_codes(settings: Optional[Settings] = None) -> List[str]:
    basics = repository.get_all_stock_basics()
    codes = [s.ts_code for s in basics]
    limit = settings.stock_pool_limit if settings is not None else STOCK_POOL_LIMIT
    if limit is not None:
        return codes[:limit]
    return codes


def update_daily_bars(
    trade_date: date,
    count: int = 240,
    settings: Optional[Settings] = None,
) -> None:
    """
    更新所有股票的日线数据。

    简化逻辑：
    - 对每个 ts_code：
      - 从 pytdx 获取最近 count 条日线；
      - 判断其中哪些日期晚于 DB 中已有的最后日期；
      - 将这部分数据 upsert 到 StockDaily。
    """
    ts_codes = _get_all_stock_codes(settings)
    if not ts_codes:
        print("[update_daily_bars] No stock_basic records found.")
        return

    print(f"[update_daily_bars] Start for {len(ts_codes)} stocks, trade_date={trade_date}")

    with PytdxDataSource() as ds:
        for i, ts_code in enumerate(tqdm(ts_codes, desc="daily", unit="stk"), start=1):
            last_date = repository.get_last_trade_date_for_stock(ts_code)
            df = ds.get_daily_bars(ts_code, count=count)
            if df.empty:
                continue
            df["trade_date"] = df["datetime"].dt.date
            df_new = df[df["trade_date"] > last_date] if last_date is not None else df
            if df_new.empty:
                continue
            repository.upsert_stock_daily(ts_code, df_new)

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
            repository.upsert_stock_minute(ts_code, df, freq=freq)

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

def collect_full_day_ticks(trade_date: date, settings: Optional[Settings] = None) -> None:
    basics = repository.get_all_stock_basics()
    ts_codes = [s.ts_code for s in basics]
    from .tick_store import TickStore
    store = TickStore(settings=settings)
    print(f"[collect_full_day_ticks] Start for {len(ts_codes)} stocks, trade_date={trade_date}")
    with PytdxDataSource() as ds:
        for ts_code in tqdm(ts_codes, desc="ticks_full", unit="stk"):
            df_tick = ds.get_ticks_full_day(ts_code, trade_date)
            if df_tick.empty:
                continue
            store.save_ticks(ts_code, trade_date, df_tick, already_sorted=True)
    print("[collect_full_day_ticks] Done.")
