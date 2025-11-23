# data/repository.py
"""
Repository 层：封装对数据库中行情相关表的读写操作。

目标：
- 上层 updater 只调用这里的方法，不直接操作 Session/query。
- 便于未来替换 ORM 或增加缓存。

当前支持：
- StockBasic 列表读取
- StockDaily / StockMinute 的增量写入（简单“删后插”方式）
- 查询某只股票的最后交易日
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import func, select, text

from db.connection import get_session, get_engine
from db.models import StockBasic, StockDaily, StockMinute


def _table_exists(eng, table_name: str) -> bool:
    q = (
        "select 1 from information_schema.tables where table_schema='public' and table_name='"
        + table_name
        + "'"
    )
    with eng.connect() as conn:
        conn.rollback()
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        df = pd.read_sql(q, conn)
    return len(df) > 0

# -------- StockBasic --------

def get_all_stock_basics() -> List[StockBasic]:
    eng = get_engine()
    if _table_exists(eng, "stock_basic"):
        with eng.connect() as conn:
            conn.rollback()
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            df = pd.read_sql("select * from stock_basic", conn)
        if "ts_code" in df.columns:
            codes = df["ts_code"].astype(str).tolist()
        else:
            market_map = {0: "SZ", 1: "SH", "SZ": "SZ", "SH": "SH"}
            if "market" in df.columns:
                exch = df["market"].map(market_map).fillna("SZ").astype(str)
                codes = (df["code"].astype(str) + "." + exch).tolist()
            elif "exchange" in df.columns:
                exch = df["exchange"].map(market_map).fillna("SZ").astype(str)
                codes = (df["code"].astype(str) + "." + exch).tolist()
            else:
                codes = (df["code"].astype(str) + ".SZ").tolist()
        return [StockBasic(ts) for ts in codes]
    return []


def upsert_stock_basic(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    eng = get_engine()
    existing = pd.DataFrame()
    if _table_exists(eng, "stock_basic"):
        with eng.connect() as conn:
            conn.rollback()
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            existing = pd.read_sql("select ts_code, code, exchange, name from stock_basic", conn)
    df = df.copy()
    df["ts_code"] = df["code"].astype(str) + "." + df["exchange"].astype(str)
    df = df[["ts_code", "code", "exchange", "name"]]
    to_insert = df
    to_delete = pd.DataFrame(columns=["ts_code"])
    if not existing.empty:
        merged = existing.merge(df, on=["ts_code"], how="outer", indicator=True, suffixes=("_old", ""))
        changed = merged[(merged["_merge"] == "both") & (merged["name_old"] != merged["name"])]["ts_code"].dropna()
        new_codes = merged[merged["_merge"] == "right_only"]["ts_code"].dropna()
        to_delete = pd.concat([changed]).to_frame(name="ts_code")
        to_insert = df[df["ts_code"].isin(pd.concat([changed, new_codes]).astype(str))]
    with eng.begin() as conn:
        if not to_delete.empty and _table_exists(eng, "stock_basic"):
            payload = [{"code": c} for c in to_delete["ts_code"].astype(str).tolist()]
            if payload:
                conn.execute(text("delete from stock_basic where ts_code = :code"), payload)
        if not to_insert.empty:
            to_insert.to_sql("stock_basic", conn, if_exists="append", index=False)
    return len(to_insert)


# -------- StockDaily --------

def get_last_trade_date_for_stock(ts_code: str) -> Optional[date]:
    """
    返回某只股票在 StockDaily 中最新的 trade_date。
    """
    eng = get_engine()
    if _table_exists(eng, "stock_daily"):
        with eng.connect() as conn:
            conn.rollback()
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            df = pd.read_sql(
                f"select max(trade_date) as last_date from stock_daily where ts_code = '{ts_code}'",
                conn,
            )
        if df.empty:
            return None
        last = df["last_date"].iloc[0]
        if pd.isna(last):
            return None
        return last.date() if hasattr(last, "date") else last
    return None


def upsert_stock_daily(ts_code: str, df: pd.DataFrame) -> int:
    """
    将 DataFrame 中的日线数据写入 StockDaily。
    简化做法：先删除 df 范围内该 ts_code 的记录，再插入。

    要求 df 至少有：
    - datetime
    - open/high/low/close/volume
    """
    if df.empty:
        return 0

    df = df.copy()
    df["trade_date"] = df["datetime"].dt.date

    min_date = df["trade_date"].min()
    max_date = df["trade_date"].max()

    eng = get_engine()
    with eng.begin() as conn:
        if _table_exists(eng, "stock_daily"):
            conn.execute(
                text(
                    "delete from stock_daily where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2"
                ),
                {"ts": ts_code, "d1": min_date, "d2": max_date},
            )
        out = df[[
            "trade_date","open","high","low","close","volume","amount"
        ]].copy()
        out.insert(0, "ts_code", ts_code)
        out.to_sql("stock_daily", conn, if_exists="append", index=False)
    return len(df)


# -------- StockMinute --------

def upsert_stock_minute(
    ts_code: str,
    df: pd.DataFrame,
    freq: str = "1m",
) -> int:
    """
    将 DataFrame 中的分钟线数据写入 StockMinute。

    要求 df 至少有：
    - datetime
    - open/high/low/close/volume
    """
    if df.empty:
        return 0

    df = df.copy()
    df["trade_date"] = df["datetime"].dt.date
    df["minute"] = df["datetime"].dt.strftime("%H:%M")

    min_date = df["trade_date"].min()
    max_date = df["trade_date"].max()

    eng = get_engine()
    with eng.begin() as conn:
        if _table_exists(eng, "stock_minute"):
            conn.execute(
                text(
                    "delete from stock_minute where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2"
                ),
                {"ts": ts_code, "d1": min_date, "d2": max_date},
            )
        out = df[[
            "trade_date","minute","open","high","low","close","volume","amount"
        ]].copy()
        out.insert(0, "ts_code", ts_code)
        out.to_sql("stock_minute", conn, if_exists="append", index=False)
    return len(df)


# -------- StockKline --------

def _ensure_stock_kline(eng) -> None:
    with eng.begin() as conn:
        conn.execute(text(
            """
            create table if not exists public.stock_kline (
                ts_code text not null,
                freq text not null,
                datetime timestamp not null,
                trade_date date not null,
                open double precision not null,
                high double precision not null,
                low double precision not null,
                close double precision not null,
                volume bigint not null,
                amount double precision,
                primary key (ts_code, freq, datetime)
            )
            """
        ))
        conn.execute(text(
            "create index if not exists idx_stock_kline_ts_freq_date on public.stock_kline(ts_code, freq, trade_date)"
        ))
        conn.execute(text(
            "create index if not exists idx_stock_kline_code_freq_date on public.stock_kline(ts_code, freq, trade_date)"
        ))
        conn.execute(text(
            "create index if not exists idx_stock_kline_ts_freq_dt_desc on public.stock_kline(ts_code, freq, datetime desc)"
        ))


def upsert_stock_kline(ts_code: str, freq: str, df: pd.DataFrame) -> int:
    """
    将 df 中的 K 线写入 stock_kline。
    要求 df 至少包含 datetime/open/high/low/close/volume，可选 amount。
    删除 [min_dt, max_dt] 区间旧记录后批量插入。
    """
    if df is None or df.empty:
        return 0
    eng = get_engine()
    _ensure_stock_kline(eng)
    df = df.copy()
    df["trade_date"] = df["datetime"].dt.date
    min_dt = df["datetime"].min()
    max_dt = df["datetime"].max()
    with eng.begin() as conn:
        conn.execute(text(
            "delete from stock_kline where ts_code=:ts and freq=:fq and datetime>=:t1 and datetime<=:t2"
        ), {"ts": ts_code, "fq": freq, "t1": min_dt, "t2": max_dt})
        out = df[["datetime","trade_date","open","high","low","close","volume","amount"]].copy()
        out.insert(0, "freq", freq)
        out.insert(0, "ts_code", ts_code)
        out.to_sql("stock_kline", conn, if_exists="append", index=False)
    return len(df)


def get_recent_trading_dates(ts_code: str, limit: int) -> List[date]:
    """
    返回该股票在 stock_kline 中 freq='1d' 的最近 limit 个有成交的交易日（volume>0），按时间升序。
    """
    eng = get_engine()
    _ensure_stock_kline(eng)
    with eng.connect() as conn:
        conn.rollback()
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        df = pd.read_sql(
            text(
                "select trade_date, volume from stock_kline where ts_code=:ts and freq='1d' order by trade_date desc limit :lim"
            ),
            conn,
            params={"ts": ts_code, "lim": int(limit)},
            parse_dates=["trade_date"],
        )
    if df.empty:
        return []
    df = df[df["volume"].astype(float) > 0]
    dates = df["trade_date"].dt.date.tolist()
    dates.reverse()
    return dates


def get_kline_date_range(ts_code: str, freq: str) -> tuple[date | None, date | None]:
    """
    返回给定 ts_code+freq 在 stock_kline 中的 (min_trade_date, max_trade_date)。
    如果没有记录，返回 (None, None)。
    """
    eng = get_engine()
    _ensure_stock_kline(eng)
    with eng.connect() as conn:
        conn.rollback()
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        df = pd.read_sql(
            text("select min(trade_date) as d1, max(trade_date) as d2 from stock_kline where ts_code=:ts and freq=:fq"),
            conn,
            params={"ts": ts_code, "fq": freq},
            parse_dates=["d1","d2"],
        )
    if df.empty:
        return None, None
    d1 = df["d1"].iloc[0]
    d2 = df["d2"].iloc[0]
    if pd.isna(d1) or pd.isna(d2):
        return None, None
    return (d1.date() if hasattr(d1, "date") else d1, d2.date() if hasattr(d2, "date") else d2)


def get_kline_missing_dates(
    ts_code: str,
    freq: str,
    start_date: date,
    end_date: date,
    trade_calendar: List[date] | None = None,
) -> List[date]:
    """
    返回在 [start_date, end_date] 之间，stock_kline 中缺失的交易日列表。
    如果提供 trade_calendar，以其为基准；否则使用 stock_kline 出现过的 trade_date 去推断连续性。
    """
    eng = get_engine()
    _ensure_stock_kline(eng)
    with eng.connect() as conn:
        conn.rollback()
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        df = pd.read_sql(
            text("select distinct trade_date from stock_kline where ts_code=:ts and freq=:fq and trade_date>=:d1 and trade_date<=:d2 order by trade_date"),
            conn,
            params={"ts": ts_code, "fq": freq, "d1": start_date, "d2": end_date},
            parse_dates=["trade_date"],
        )
    have = set(df["trade_date"].dt.date.tolist()) if not df.empty else set()
    if trade_calendar is None:
        base = have
    else:
        base = set([d for d in trade_calendar if start_date <= d <= end_date])
    missing = sorted(list(base - have))
    return missing


if __name__ == "__main__":
    # 自测：构造虚拟 df 写入，再读取最后交易日
    from datetime import datetime

    print("[repository] self test...")

    # 假设 stock_basic 已有 000001.SZ 记录，否则这一步不会报错，但只是插入日线。
    test_ts = "000001.SZ"
    now = datetime.now()

    df_daily = pd.DataFrame(
        {
            "datetime": [
                now.replace(hour=0, minute=0, second=0, microsecond=0),
            ],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [123456],
            "amount": [1234567.0],
        }
    )

    n = upsert_stock_daily(test_ts, df_daily)
    print(f"Inserted/updated {n} daily rows")

    last_date = get_last_trade_date_for_stock(test_ts)
    print(f"Last trade date for {test_ts}: {last_date}")
