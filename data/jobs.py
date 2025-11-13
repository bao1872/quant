# data/jobs.py
"""
定时任务入口：

- job_update_ohlc: 盘前/盘后更新日线 & 分钟线
- job_collect_ticks_once: 测试用 tick 抓取
- job_finalize_ticks_and_levels: 收盘后（或盘前）计算关键位
"""

from __future__ import annotations

from datetime import date as _date

from .updater import update_daily_bars, update_minute_bars, collect_full_day_ticks
from .repository import get_all_stock_basics
from factors import AbuPriceLevelProvider


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


if __name__ == "__main__":
    from datetime import date as date_cls
    import os
    os.environ["USE_REAL_DB"] = "1"
    today = date_cls.today()
    job_update_ohlc(today)
    job_collect_full_day_ticks(today)
    job_finalize_ticks_and_levels(today)
