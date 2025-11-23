from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine


def check_bi_direction(code: str, freq: str, begin_time: str, end_time: str) -> int:
    eng = get_engine()
    with eng.connect() as conn:
        q_up = text(
            "select count(1) as c from stock_chan_bi where code=:c and freq=:f and start_ts>=:d1 and end_ts<=:d2 and direction='UP' and end_price < start_price"
        )
        q_dn = text(
            "select count(1) as c from stock_chan_bi where code=:c and freq=:f and start_ts>=:d1 and end_ts<=:d2 and direction='DOWN' and end_price > start_price"
        )
        df1 = pd.read_sql(q_up, conn, params={"c": code, "f": freq, "d1": begin_time, "d2": end_time})
        df2 = pd.read_sql(q_dn, conn, params={"c": code, "f": freq, "d1": begin_time, "d2": end_time})
    n1 = int(df1["c"].iloc[0]) if not df1.empty else 0
    n2 = int(df2["c"].iloc[0]) if not df2.empty else 0
    print("bi_direction_errors:", n1 + n2)
    return n1 + n2


def check_bi_time_order(code: str, freq: str) -> int:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text("select start_ts, end_ts from stock_chan_bi where code=:c and freq=:f order by start_ts"),
            conn,
            params={"c": code, "f": freq},
            parse_dates=["start_ts", "end_ts"],
        )
    if df.empty:
        print("bi_time_errors:", 0)
        return 0
    e_lt_s = (df["end_ts"] < df["start_ts"]).sum()
    next_start_lt_prev_end = (df["start_ts"].shift(-1) < df["end_ts"]).fillna(False).sum()
    total = int(e_lt_s) + int(next_start_lt_prev_end)
    print("bi_time_errors:", total)
    return total


def check_center_range(code: str, freq: str) -> int:
    eng = get_engine()
    with eng.connect() as conn:
        cen = pd.read_sql(
            text("select center_id, high as center_high, low as center_low, related_seg_index from stock_chan_center where code=:c and freq=:f"),
            conn,
            params={"c": code, "f": freq},
        )
        seg = pd.read_sql(
            text("select seg_id, high as seg_high, low as seg_low from stock_chan_segment where code=:c and freq=:f"),
            conn,
            params={"c": code, "f": freq},
        )
    if cen.empty or seg.empty:
        print("center_range_errors:", 0)
        return 0
    merged = cen.merge(seg, left_on="related_seg_index", right_on="seg_id", how="inner")
    bad = ((merged["center_low"] < merged["seg_low"]) | (merged["center_high"] > merged["seg_high"]))
    cnt = int(bad.sum())
    print("center_range_errors:", cnt)
    return cnt


if __name__ == "__main__":
    code = "SH.688122"
    e1 = check_bi_direction(code, "day", "2024-01-01", "2024-12-31")
    e2 = check_bi_time_order(code, "day")
    e3 = check_center_range(code, "day")
    print("summary:", e1, e2, e3)