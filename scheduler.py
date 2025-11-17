from __future__ import annotations

from datetime import date
import pandas as pd

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from data.basic_universe import ensure_stock_basic_a_share
from db.connection import get_engine
from sqlalchemy import inspect
from data.jobs import job_update_ohlc, job_collect_ticks_once, job_finalize_ticks_and_levels


def is_trading_day(d: date) -> bool:
    eng = get_engine()
    inspector = inspect(eng)
    has_tbl = inspector.has_table("trading_calendar", schema="public")
    if not has_tbl:
        return False
    df = pd.read_sql_table("trading_calendar", eng)
    if df.empty:
        return False
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    sub = df[df["trade_date"] == d]
    if sub.empty:
        return False
    if "is_open" in sub.columns:
        return bool(sub["is_open"].iloc[0])
    return True


def run_post_close_jobs(trade_date: date | None = None) -> None:
    td = trade_date or date.today()
    if not is_trading_day(td):
        return
    ensure_stock_basic_a_share()
    job_update_ohlc(td)
    job_collect_ticks_once(td, limit=None)
    job_finalize_ticks_and_levels(td)


def main() -> None:
    sched = BlockingScheduler()
    sched.add_job(lambda: run_post_close_jobs(), CronTrigger(hour=15, minute=10, timezone="Asia/Shanghai"))
    sched.start()


if __name__ == "__main__":
    main()
