from __future__ import annotations

from datetime import date, timedelta
import os

from data.basic_universe import ensure_stock_basic_a_share
from data.jobs import job_update_ohlc, job_finalize_ticks_and_levels


def main() -> None:
    today = date.today()
    os.environ["USE_REAL_DB"] = "1"
    ensure_stock_basic_a_share()
    job_update_ohlc(today)
    start = today - timedelta(days=730)
    cur = start
    while cur <= today:
        from data.updater import collect_full_day_ticks
        collect_full_day_ticks(cur)
        cur = cur + timedelta(days=1)
    job_finalize_ticks_and_levels(today)


if __name__ == "__main__":
    main()
