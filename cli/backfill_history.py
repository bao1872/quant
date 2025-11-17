from __future__ import annotations

from datetime import date, timedelta
import argparse
import os
from time import perf_counter
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from db.connection import get_engine
from sqlalchemy import text
from data.jobs import job_update_ohlc, job_finalize_ticks_and_levels
from config import TICK_BASE_DIR


def _table_exists(eng, table_name: str) -> bool:
    q = (
        "select 1 from information_schema.tables where table_schema='public' and table_name='"
        + table_name
        + "'"
    )
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)
        return len(df) > 0


def _get_max_date(eng, table_name: str, col: str = "trade_date") -> date | None:
    if not _table_exists(eng, table_name):
        return None
    with eng.connect() as conn:
        df = pd.read_sql(f"select max({col}) as d from {table_name}", conn)
        if df.empty:
            return None
        v = df["d"].iloc[0]
        if pd.isna(v):
            return None
        return v.date() if hasattr(v, "date") else v


def _is_db_empty(eng) -> bool:
    d1 = _get_max_date(eng, "stock_daily")
    d2 = _get_max_date(eng, "tick_file_index")
    return d1 is None and d2 is None


def _last_processed_date(eng) -> date | None:
    d2 = _get_max_date(eng, "tick_file_index")
    if d2 is not None:
        return d2
    return _get_max_date(eng, "stock_daily")


def _count_tick_stats_for_date_fs(d: date) -> tuple[int, int]:
    base = Path(TICK_BASE_DIR)
    if not base.exists():
        return 0, 0
    pat = f"{d.strftime('%Y%m%d')}_交易数据.parquet"
    files = list(base.rglob(pat))
    n_files = len(files)
    total = 0
    if n_files == 0:
        return 0, 0
    import pyarrow.parquet as pq
    for f in files:
        pf = pq.ParquetFile(str(f))
        total += int(pf.metadata.num_rows or 0)
    return n_files, total


def _unique_dates(eng, table: str, start: date, end: date, col: str = "trade_date") -> pd.Series:
    if not _table_exists(eng, table):
        return pd.Series(dtype="datetime64[ns]")
    with eng.connect() as conn:
        df = pd.read_sql(
            text(f"select {col} as d from {table} where {col}>=:d1 and {col}<=:d2 group by {col} order by {col}"),
            conn,
            params={"d1": start, "d2": end},
        )
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df["d"]).dt.date

def _missing_tick_dates(eng, start: date, end: date) -> pd.Series:
    daily = _unique_dates(eng, "stock_daily", start, end)
    base = Path(TICK_BASE_DIR)
    ticks = set()
    if base.exists():
        for p in base.rglob("*_交易数据.parquet"):
            name = p.name
            s = name.split("_")[0]
            if len(s) == 8 and s.isdigit():
                dt = date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
                if start <= dt <= end:
                    ticks.add(dt)
    if len(daily) == 0:
        return pd.Series(dtype="datetime64[ns]")
    if len(ticks) == 0:
        return pd.Series(sorted(list(set(daily))))
    missing = sorted(list(set(daily) - ticks))
    return pd.Series(missing)

def _batch_dates(dates: pd.Series, batch_size: int = 100) -> list[list[date]]:
    if dates is None or len(dates) == 0:
        return []
    arr = pd.Series(sorted(list(dates))).tolist()
    n = len(arr)
    batches: list[list[date]] = []
    idx = pd.Series(range(n))
    cuts = (idx // batch_size).tolist()
    current = -1
    for i, c in enumerate(cuts):
        if c != current:
            batches.append([])
            current = c
        batches[-1].append(arr[i])
    return batches

def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--ts", type=str, default="", help="comma separated ts_codes, e.g., 000001.SZ,600000.SH")
    parser.add_argument("--date", type=str, default="", help="YYYY-MM-DD for tick collection")
    parser.add_argument("--mode", type=str, default="intraday", choices=["intraday", "full"], help="tick collection mode for --ts")
    parser.add_argument("--count", type=int, default=2000, help="tick count for intraday mode")
    args = parser.parse_args()

    today = date.today() if not args.date else date.fromisoformat(args.date)
    os.environ["USE_REAL_DB"] = "1"
    eng = get_engine()
    start_time = perf_counter()
    empty = _is_db_empty(eng)
    mode = "full"

    # If specific ts_codes provided, run targeted action and exit
    if args.ts:
        ts_list = [s.strip() for s in args.ts.split(",") if s.strip()]
        if args.mode == "full":
            from data.pytdx_source import PytdxDataSource
            from data.tick_store import TickStore
            store = TickStore()
            print(f"[collect_full_day_ticks:selected] Start for {len(ts_list)} stocks, trade_date={today}")
            with PytdxDataSource(enable_fallback=False) as ds:
                for ts_code in tqdm(ts_list, desc="ticks_sel_full", unit="stk"):
                    df_tick = ds.get_ticks_full_day(ts_code, today)
                    if df_tick.empty:
                        continue
                    store.save_ticks(ts_code, today, df_tick, already_sorted=True)
            print("[collect_full_day_ticks:selected] Done.")
        else:
            from data.updater import collect_intraday_ticks
            print(f"[collect_intraday_ticks:selected] Start for {len(ts_list)} stocks, trade_date={today}")
            collect_intraday_ticks(today, ts_codes=ts_list, count=args.count)
            print("[collect_intraday_ticks:selected] Done.")
        n, total = _count_tick_stats_for_date_fs(today)
        tick_files = n
        tick_records = total
        print(
            f"[backfill_history:selected] mode={args.mode} date={today} selected_count={len(ts_list)} today_tick_files={tick_files} today_records={tick_records} status=ok"
        )
        return

    # 分离两种数据的获取范围：
    ohlc_start = today - timedelta(days=730)
    tick_start = today - timedelta(days=365)
    end = today

    # K线获取暂不执行（仅调试 tick），如下保留但注释
    # from data.updater import update_daily_bars
    # print(f"[update_ohlc] range={ohlc_start}~{end} count=730")
    # update_daily_bars(trade_date=today, count=730)

    # Tick 缺失按近一年日期集合处理（保留但注释的调用）
    missing = _missing_tick_dates(eng, tick_start, end)
    if len(missing) == 0:
        print(f"[backfill_history] no missing tick dates between {tick_start}~{end}")
    else:
        batches = _batch_dates(missing, batch_size=100)
        print(f"[backfill_history] tick_missing_dates={len(missing)} batches={len(batches)} range={missing.min()}~{missing.max()}")
        from data.updater import collect_full_day_ticks
        for b in batches:
            b_start = b[0]
            b_end = b[-1]
            print(f"[backfill_tick_batch] start={b_start} end={b_end} size={len(b)}")
            for d in tqdm(b, desc="ticks_days", unit="day"):
                collect_full_day_ticks(d)
                n, total = _count_tick_stats_for_date_fs(d)
                print(f"[verify] date={d} files={n} records={total}")

    # job_finalize_ticks_and_levels(today)
    elapsed = perf_counter() - start_time
    n, total = _count_tick_stats_for_date_fs(today)
    tick_files = n
    tick_records = total
    print(f"[backfill_history] mode={mode} ohlc_range={ohlc_start}~{end} tick_range={tick_start}~{end} today_tick_files={tick_files} today_records={tick_records} elapsed={elapsed:.2f}s status=ok")


if __name__ == "__main__":
    main()
