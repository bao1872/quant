from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

from config import Settings
from data.updater import update_stock_basic, update_daily_bars, collect_full_day_ticks


def _parse_date(s: str) -> date:
    d = datetime.strptime(s, "%Y-%m-%d").date()
    return d


def _date_range(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill A-share history: basics, 2y K-line and ticks")
    parser.add_argument("--start", type=str, default=(date.today() - timedelta(days=730)).isoformat())
    parser.add_argument("--end", type=str, default=date.today().isoformat())
    args = parser.parse_args()

    settings = Settings()
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    print("[fill] update stock_basic")
    update_stock_basic(settings=settings)
    print("[fill] update daily bars")
    update_daily_bars(trade_date=end, count=600, settings=settings)
    print("[fill] collect full-day ticks")
    for d in _date_range(start, end):
        collect_full_day_ticks(trade_date=d, settings=settings)


if __name__ == "__main__":
    main()

