# data/jobs.py
"""
定时任务入口：

- job_update_ohlc: 盘前/盘后更新日线 & 分钟线
- job_collect_ticks_once: 测试用 tick 抓取
- job_finalize_ticks_and_levels: 收盘后（或盘前）计算关键位
"""

from __future__ import annotations

from datetime import date as _date

from .updater import (
    update_daily_bars,
    update_minute_bars,
    collect_intraday_ticks,
)
from .repository import get_all_stock_basics
from factors import AbuPriceLevelProvider


def job_update_ohlc(trade_date: _date) -> None:
    print(f"[job_update_ohlc] trade_date={trade_date}")
    update_daily_bars(trade_date=trade_date, count=250)
    # 分钟线可按需开启
    # update_minute_bars(trade_date=trade_date, freq="1m", count=240)


def job_collect_ticks_once(trade_date: _date, limit: int = 20) -> None:
    basics = get_all_stock_basics()
    ts_codes = [s.ts_code for s in basics]
    if limit is not None and limit > 0:
        ts_codes = ts_codes[:limit]
    collect_intraday_ticks(trade_date=trade_date, ts_codes=ts_codes, count=1000)


def job_finalize_ticks_and_levels(trade_date: _date) -> None:
    print(f"[job_finalize_ticks_and_levels] trade_date={trade_date}")
    provider = AbuPriceLevelProvider()
    provider.precompute(trade_date)


if __name__ == "__main__":
    from datetime import date as date_cls

    today = date_cls.today()
    print("[jobs] self test -> job_update_ohlc")
    job_update_ohlc(today)
    print("[jobs] self test -> job_finalize_ticks_and_levels")
    job_finalize_ticks_and_levels(today)
