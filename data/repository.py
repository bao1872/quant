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

from db.connection import get_session, DummySession, get_engine
from db.models import StockBasic, StockDaily, StockMinute


def _table_exists(eng, table_name: str) -> bool:
    q = (
        "select 1 from information_schema.tables where table_schema='public' and table_name='"
        + table_name
        + "'"
    )
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)
    return len(df) > 0

# -------- StockBasic --------

def get_all_stock_basics() -> List[StockBasic]:
    eng = get_engine()
    if eng is not None:
        if _table_exists(eng, "stock_basic"):
            with eng.connect() as conn:
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
        return [StockBasic("000001.SZ")]
    with get_session() as session:
        return [StockBasic("000001.SZ")]


def upsert_stock_basic(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    eng = get_engine()
    if eng is None:
        with get_session() as session:
            return len(df)
    existing = pd.DataFrame()
    if _table_exists(eng, "stock_basic"):
        with eng.connect() as conn:
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
    if eng is not None:
        if _table_exists(eng, "stock_daily"):
            with eng.connect() as conn:
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
    if eng is None:
        with get_session() as session:
            objs = []
            for _, row in df.iterrows():
                objs.append(
                    StockDaily(
                        ts_code=ts_code,
                        trade_date=row["trade_date"],
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                        amount=row.get("amount", 0.0),
                    )
                )
            session.bulk_save_objects(objs)
            return len(objs)
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
    if eng is None:
        with get_session() as session:
            objs = []
            for _, row in df.iterrows():
                objs.append(
                    StockMinute(
                        ts_code=ts_code,
                        trade_date=row["trade_date"],
                        minute=row["minute"],
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                        amount=row.get("amount", 0.0),
                    )
                )
            session.bulk_save_objects(objs)
            return len(objs)
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
