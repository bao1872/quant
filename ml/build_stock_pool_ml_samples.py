from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine


def create_stock_pool_ml_samples_table() -> None:
    eng = get_engine()
    with eng.begin() as conn:
        from sqlalchemy import inspect
        inspector = inspect(conn)
        has_table = inspector.has_table("stock_pool_ml_samples", schema="public")
        if not has_table:
            ddl = (
                "CREATE TABLE IF NOT EXISTS stock_pool_ml_samples ("
                "ts_code text NOT NULL,"
                "trade_date date NOT NULL,"
                "y_ret_10d double precision,"
                "created_at timestamp without time zone DEFAULT now(),"
                "PRIMARY KEY (ts_code, trade_date)"
                ");"
            )
            conn.execute(text(ddl))
        else:
            conn.execute(text("ALTER TABLE stock_pool_ml_samples ADD COLUMN IF NOT EXISTS trade_date date"))
            conn.execute(text("ALTER TABLE stock_pool_ml_samples ADD COLUMN IF NOT EXISTS y_ret_10d double precision"))
            conn.execute(text("ALTER TABLE stock_pool_ml_samples ADD COLUMN IF NOT EXISTS created_at timestamp without time zone DEFAULT now()"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_spms_ts_trade ON stock_pool_ml_samples(ts_code, trade_date)"))


def get_trade_dates(start_date: dt.date, end_date: dt.date) -> List[dt.date]:
    eng = get_engine()
    sql = (
        "SELECT DISTINCT b.trade_date FROM stock_bollinger_data b "
        "WHERE b.trade_date >= :start_date AND b.trade_date <= :end_date "
        "AND EXISTS (SELECT 1 FROM stock_tick_daily_features t WHERE t.trade_date = b.trade_date) "
        "ORDER BY b.trade_date"
    )
    with eng.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"start_date": start_date, "end_date": end_date}, parse_dates=["trade_date"])
    if df.empty:
        return []
    return [d.date() for d in df["trade_date"]]


def insert_samples_for_window(window_start: dt.date, window_end: dt.date) -> None:
    eng = get_engine()
    delete_sql = (
        "DELETE FROM stock_pool_ml_samples WHERE trade_date >= :start_date AND trade_date <= :end_date"
    )
    insert_sql = (
        "INSERT INTO stock_pool_ml_samples (ts_code, trade_date, y_ret_10d) "
        "SELECT ts_code, trade_date, y_ret_10d FROM ("
        "SELECT ts_code, trade_date::date AS trade_date, y_ret_10d, band_width_zscore, v_band_width_zscore, "
        "price_position, v_price_position, LAG(price_position, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) AS price_position_lag1 "
        "FROM stock_bollinger_data WHERE trade_date >= :start_date AND trade_date <= :end_date AND y_ret_10d IS NOT NULL"
        ") s WHERE band_width_zscore < 0.5 AND v_band_width_zscore < 0.5 "
        "AND price_position_lag1 IS NOT NULL AND price_position > price_position_lag1 "
        "AND price_position <= 60 AND v_price_position <= 40 "
        "ON CONFLICT (ts_code, trade_date) DO NOTHING"
    )
    with eng.begin() as conn:
        conn.execute(text(delete_sql), {"start_date": window_start, "end_date": window_end})
        conn.execute(text(insert_sql), {"start_date": window_start, "end_date": window_end})


def build_stock_pool_ml_samples(start_date: dt.date, end_date: dt.date, window_days: int = 30) -> None:
    create_stock_pool_ml_samples_table()
    trade_dates = get_trade_dates(start_date, end_date)
    if not trade_dates:
        return
    n = len(trade_dates)
    i = 0
    while i < n:
        ws = trade_dates[i]
        we = trade_dates[min(i + window_days - 1, n - 1)]
        i += window_days
        insert_samples_for_window(ws, we)


if __name__ == "__main__":
    today = dt.date.today()
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("TRUNCATE stock_pool_ml_samples"))
    with eng.connect() as conn:
        df_min = pd.read_sql(
            text("SELECT min(b.trade_date) AS d FROM stock_bollinger_data b WHERE y_ret_10d IS NOT NULL AND EXISTS (SELECT 1 FROM stock_tick_daily_features t WHERE t.trade_date=b.trade_date)"),
            conn,
        )
    if df_min.empty or pd.isna(df_min["d"].iloc[0]):
        start = dt.date(2024, 11, 15)
    else:
        start_val = df_min["d"].iloc[0]
        start = start_val.date() if hasattr(start_val, "date") else start_val
    end = today
    build_stock_pool_ml_samples(start_date=start, end_date=end, window_days=30)
    with eng.connect() as conn:
        dfc = pd.read_sql(
            text("SELECT count(1) as c FROM stock_pool_ml_samples WHERE trade_date>=:d1 AND trade_date<=:d2"),
            conn,
            params={"d1": start, "d2": end},
        )
    if not dfc.empty:
        print(int(pd.to_numeric(dfc["c"], errors="coerce").fillna(0).iloc[0]))
