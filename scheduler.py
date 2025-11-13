from __future__ import annotations

from datetime import date

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from data.basic_universe import ensure_stock_basic_a_share
from data.jobs import job_update_ohlc, job_collect_ticks_once, job_finalize_ticks_and_levels


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5


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
    sched.add_job(lambda: run_post_close_jobs(), CronTrigger(hour=15, minute=10))
    sched.start()


if __name__ == "__main__":
    main()

