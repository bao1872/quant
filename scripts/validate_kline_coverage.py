from __future__ import annotations

from datetime import date
from typing import List

import pandas as pd

from data.repository import get_all_stock_basics, get_kline_date_range, get_kline_missing_dates
from config import KLINE_FREQS


def main() -> None:
    basics = get_all_stock_basics()
    codes: List[str] = [b.ts_code for b in basics][:50]
    for ts in codes:
        base_start, base_end = get_kline_date_range(ts, "1d")
        if base_start is None or base_end is None:
            print(f"=== {ts} ===")
            print("1d: range=None .. None (missing_days=ALL)")
            continue
        print(f"=== {ts} ===")
        print(f"1d: range={base_start} .. {base_end} (missing_days=0)")
        for fq in [f for f in KLINE_FREQS if f != "1d"]:
            m = get_kline_missing_dates(ts, fq, base_start, base_end, trade_calendar=None)
            print(f"{fq}: range={base_start} .. {base_end} (missing_days={len(m)})" + (f" -> {m[:3]}..." if len(m) > 3 else (f" -> {m}" if m else "")))


if __name__ == "__main__":
    main()