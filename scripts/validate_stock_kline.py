"""
验证 stock_kline 的日线与分钟线时间范围一致性，以及最近交易日的缺口情况。

检查要点：
- 对若干 ts_code，打印 freq=1d/60m/15m 的日期范围与行数/天数；
- 用 1d 的 min/max 为基准，对分钟 freq 的起点差距做 WARNING 检查；
- 最近 5 个有成交的交易日（volume>0）检查分钟线是否缺失，缺失打印 ERROR；

运行：
  python scripts/validate_stock_kline.py
"""

from __future__ import annotations

from typing import List
from datetime import date

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine


FREQS: List[str] = ["1d", "60m", "15m"]
MAX_RECENT_DAYS: int = 5
MAX_START_LAG_DAYS: int = 5
MIN_DAILY_DAYS: int = 400
MIN_MINUTE_RATIO: float = 0.9


def _select_sample_codes(n: int = 3) -> List[str]:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text("select distinct ts_code from stock_kline where freq='1d' order by ts_code limit :lim"),
            conn,
            params={"lim": int(n)},
        )
    return df["ts_code"].astype(str).tolist() if not df.empty else []


def _range_stats(ts_code: str, freq: str) -> dict:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text(
                "select min(trade_date) as min_date, max(trade_date) as max_date, count(*) as rows, count(distinct trade_date) as days from stock_kline where ts_code=:ts and freq=:fq"
            ),
            conn,
            params={"ts": ts_code, "fq": freq},
            parse_dates=["min_date", "max_date"],
        )
    if df.empty:
        return {"min_date": None, "max_date": None, "rows": 0, "days": 0}
    rec = df.iloc[0]
    return {
        "min_date": rec.get("min_date"),
        "max_date": rec.get("max_date"),
        "rows": int(pd.to_numeric(rec.get("rows"), errors="coerce") or 0),
        "days": int(pd.to_numeric(rec.get("days"), errors="coerce") or 0),
    }


def _recent_trading_days(ts_code: str, limit: int = MAX_RECENT_DAYS) -> List[date]:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text(
                "select trade_date from stock_kline where ts_code=:ts and freq='1d' and volume>0 order by trade_date desc limit :lim"
            ),
            conn,
            params={"ts": ts_code, "lim": int(limit)},
            parse_dates=["trade_date"],
        )
    if df.empty:
        return []
    dates = df["trade_date"].dt.date.tolist()
    return dates


def _exists_minute_on_date(ts_code: str, freq: str, d: date) -> bool:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text("select count(1) as c from stock_kline where ts_code=:ts and freq=:fq and trade_date=:d"),
            conn,
            params={"ts": ts_code, "fq": freq, "d": d},
        )
    if df.empty:
        return False
    c = int(pd.to_numeric(df.get("c", pd.Series([0])), errors="coerce").fillna(0).iloc[0])
    return c > 0


def main() -> None:
    codes = _select_sample_codes(n=3)
    if not codes:
        print("No sample ts_code found in stock_kline.")
        return
    total_warnings = 0
    total_errors = 0
    print("=== Validate stock_kline ===")
    for ts in codes:
        print(f"=== {ts} ===")
        stats = {fq: _range_stats(ts, fq) for fq in FREQS}
        for fq in FREQS:
            s = stats[fq]
            md = s["min_date"]
            xd = s["max_date"]
            print(f"  freq={fq}: {md} .. {xd}, rows={s['rows']}, days={s['days']}")
        base = stats.get("1d", {})
        base_min = base.get("min_date")
        days_1d = int(base.get("days") or 0)
        if days_1d < MIN_DAILY_DAYS:
            print(f"ERROR: {ts} freq=1d only {days_1d} trading days (<{MIN_DAILY_DAYS})")
            total_errors += 1
        if base_min is not None:
            for fq in [f for f in FREQS if f != "1d"]:
                start_min = stats[fq].get("min_date")
                if start_min is None:
                    continue
                lag = (pd.to_datetime(start_min).date() - pd.to_datetime(base_min).date()).days
                if lag > MAX_START_LAG_DAYS:
                    print(f"WARNING: {ts} freq={fq} start_date {pd.to_datetime(start_min).date()} is much later than 1d {pd.to_datetime(base_min).date()}")
                    total_warnings += 1
        for fq in [f for f in FREQS if f != "1d"]:
            days_min = int(stats[fq].get("days") or 0)
            if days_1d > 0 and days_min < int(days_1d * MIN_MINUTE_RATIO):
                print(f"ERROR: {ts} freq={fq} has only {days_min} days, but 1d has {days_1d} days")
                total_errors += 1
        recent = _recent_trading_days(ts, limit=MAX_RECENT_DAYS)
        for d in recent:
            for fq in [f for f in FREQS if f != "1d"]:
                ok = _exists_minute_on_date(ts, fq, d)
                if not ok:
                    print(f"ERROR: {ts} freq={fq} missing trade_date={d}")
                    total_errors += 1
    print("Summary:")
    print(f"  codes_checked = {len(codes)}")
    print(f"  total_warnings = {total_warnings}")
    print(f"  total_errors = {total_errors}")
    if total_errors > 0:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()