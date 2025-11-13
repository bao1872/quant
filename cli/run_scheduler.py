from __future__ import annotations

from datetime import date, datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from data.updater import update_stock_basic, update_daily_bars, collect_full_day_ticks
from data.jobs import job_finalize_ticks_and_levels
from config import Settings


def main() -> None:
    settings = Settings()
    sched = BlockingScheduler()

    def _today() -> date:
        return datetime.now().date()

    sched.add_job(lambda: update_stock_basic(settings=settings), CronTrigger(hour=17, minute=0))
    sched.add_job(lambda: update_daily_bars(trade_date=_today(), count=600, settings=settings), CronTrigger(hour=17, minute=10))
    sched.add_job(lambda: collect_full_day_ticks(trade_date=_today(), settings=settings), CronTrigger(hour=17, minute=20))
    sched.add_job(lambda: job_finalize_ticks_and_levels(_today()), CronTrigger(hour=18, minute=0))

    print("[scheduler] started")
    sched.start()


if __name__ == "__main__":
    main()

