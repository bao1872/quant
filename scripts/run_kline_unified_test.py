from __future__ import annotations

from datetime import date, timedelta

from data.pytdx_source import PytdxDataSource
from data.repository import upsert_stock_kline, get_kline_date_range, get_kline_missing_dates


def main() -> None:
    code = "SH.688122"
    start = (date.today() - timedelta(days=180))
    end = date.today()
    with PytdxDataSource() as ds:
        d1 = ds.get_kline(code, "1d", start=start, end=end)
        upsert_stock_kline(code, "1d", d1)
        for fq in ["60m", "30m", "15m"]:
            dm = ds.get_kline(code, fq, start=start, end=end)
            upsert_stock_kline(code, fq, dm)
    b1, b2 = get_kline_date_range(code, "1d")
    lines = []
    lines.append(f"range_1d {b1} {b2}")
    for fq in ["60m", "30m", "15m"]:
        miss = get_kline_missing_dates(code, fq, b1, b2, trade_calendar=None)
        lines.append(f"missing {fq} {len(miss)}")
    with open(".run_kline_unified_test.out", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()