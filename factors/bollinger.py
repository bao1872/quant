from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List, Optional

from sqlalchemy import text
from db.connection import get_engine
from data import repository


def _table_exists(eng, table_name: str) -> bool:
    q = (
        "select 1 from information_schema.tables where table_schema='public' and table_name='"
        + table_name
        + "'"
    )
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)
        return len(df) > 0


def _table_columns(eng, table_name: str) -> list[str]:
    q = (
        "select column_name from information_schema.columns where table_schema='public' and table_name='"
        + table_name
        + "' order by ordinal_position"
    )
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)
        return df["column_name"].astype(str).tolist() if not df.empty else []


def _stock_name_map(eng) -> dict:
    cols = _table_columns(eng, "stock_basic")
    if not cols:
        return {}
    with eng.connect() as conn:
        df = pd.read_sql("select * from stock_basic", conn)
    if df.empty:
        return {}
    market_map = {0: "SZ", 1: "SH", "SZ": "SZ", "SH": "SH"}
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str)
    else:
        exch_col = "market" if "market" in df.columns else ("exchange" if "exchange" in df.columns else None)
        if exch_col is None:
            df["ts_code"] = df["code"].astype(str) + ".SZ"
        else:
            df["ts_code"] = df["code"].astype(str) + "." + df[exch_col].map(market_map).fillna("SZ").astype(str)
    name_col = "name" if "name" in df.columns else None
    if name_col is None:
        return {r["ts_code"]: "" for _, r in df.iterrows()}
    return {str(r["ts_code"]): str(r["name"]) for _, r in df.iterrows()}


def _fetch_stock_daily(eng, ts_codes: List[str], end_date: date, lookback: int, conn=None) -> pd.DataFrame:
    start_date = end_date - timedelta(days=lookback * 2)
    if conn is not None:
        df = pd.read_sql(
            text(
                "select ts_code, trade_date, open, high, low, close, volume, amount "
                "from stock_daily where trade_date>=:d1 and trade_date<=:d2"
            ),
            conn,
            params={"d1": start_date, "d2": end_date},
        )
    else:
        with eng.connect() as _c:
            df = pd.read_sql(
                text(
                    "select ts_code, trade_date, open, high, low, close, volume, amount "
                    "from stock_daily where trade_date>=:d1 and trade_date<=:d2"
                ),
                _c,
                params={"d1": start_date, "d2": end_date},
            )
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df[df["ts_code"].astype(str).isin(ts_codes)]
    return df.sort_values(["ts_code", "trade_date"])  # 向量化排序


def _rolling_bbands(close: pd.Series, window: int, k: float) -> pd.DataFrame:
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    up = mid + k * std
    lo = mid - k * std
    bw = (up - lo)
    pos = ((close - lo) / (up - lo)) * 100.0
    ur = (up / mid - 1.0) * 100.0
    lr = (lo / mid - 1.0) * 100.0
    return pd.DataFrame(
        {
            "upper_band": up,
            "middle_band": mid,
            "lower_band": lo,
            "band_width": bw,
            "price_position": pos,
            "upper_return": ur,
            "lower_return": lr,
        }
    )


def _rolling_bbands_volume(close: pd.Series, vol: pd.Series, window: int, k: float) -> pd.DataFrame:
    w = vol.fillna(0.0)
    pv = close * w
    p2v = (close ** 2) * w
    sum_w = w.rolling(window=window, min_periods=window).sum()
    sum_pv = pv.rolling(window=window, min_periods=window).sum()
    sum_p2v = p2v.rolling(window=window, min_periods=window).sum()
    mid = sum_pv / sum_w
    var = (sum_p2v / sum_w) - (mid ** 2)
    std = np.sqrt(var)
    up = mid + k * std
    lo = mid - k * std
    bw = (up - lo)
    pos = ((close - lo) / (up - lo)) * 100.0
    return pd.DataFrame(
        {
            "v_upper_band": up,
            "v_middle_band": mid,
            "v_lower_band": lo,
            "v_band_width": bw,
            "v_price_position": pos,
        }
    )


def _zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window=window, min_periods=window).mean()
    sd = x.rolling(window=window, min_periods=window).std(ddof=0)
    return (x - mu) / sd.replace(0.0, np.nan)


def compute_stock_bollinger_for_date(
    trade_date: date,
    window: int = 20,
    k: float = 2.0,
    z_window: int = 120,
    ts_codes: Optional[List[str]] = None,
    conn=None,
) -> int:
    eng = get_engine()
    if eng is None:
        return 0
    if ts_codes is None:
        basics = repository.get_all_stock_basics()
        ts_codes = [s.ts_code for s in basics]
    lookback = max(window, z_window) + 5
    df = _fetch_stock_daily(eng, ts_codes, trade_date, lookback, conn=conn)
    if df.empty:
        return 0
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    grp = df.groupby("ts_code", sort=False)
    mid = grp["close"].transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    std = grp["close"].transform(lambda s: s.rolling(window=window, min_periods=window).std(ddof=0))
    up = mid + k * std
    lo = mid - k * std
    bw = (up - lo)
    pos = ((df["close"] - lo) / (up - lo)) * 100.0
    ur = (up / mid - 1.0) * 100.0
    lr = (lo / mid - 1.0) * 100.0

    v_mid = grp["volume"].transform(lambda s: s.fillna(0.0).rolling(window=window, min_periods=window).mean())
    v_std = grp["volume"].transform(lambda s: s.fillna(0.0).rolling(window=window, min_periods=window).std(ddof=0))
    v_up = v_mid + k * v_std
    v_lo = v_mid - k * v_std
    v_bw = (v_up - v_lo)
    v_pos = ((df["volume"] - v_lo) / (v_up - v_lo)) * 100.0

    bw_z = bw.groupby(df["ts_code"]).transform(
        lambda s: (s - s.rolling(window=z_window, min_periods=z_window).mean()) / s.rolling(window=z_window, min_periods=z_window).std(ddof=0)
    )
    v_bw_z = v_bw.groupby(df["ts_code"]).transform(
        lambda s: (s - s.rolling(window=z_window, min_periods=z_window).mean()) / s.rolling(window=z_window, min_periods=z_window).std(ddof=0)
    )

    out = pd.DataFrame(
        {
            "ts_code": df["ts_code"],
            "trade_date": df["trade_date"],
            "upper_band": up,
            "middle_band": mid,
            "lower_band": lo,
            "band_width": bw,
            "band_width_zscore": bw_z,
            "price_position": pos,
            "upper_return": ur,
            "lower_return": lr,
            "v_upper_band": v_up,
            "v_middle_band": v_mid,
            "v_lower_band": v_lo,
            "v_band_width": v_bw,
            "v_band_width_zscore": v_bw_z,
            "v_price_position": v_pos,
        }
    )
    out = out[out["trade_date"] == trade_date]
    if out.empty:
        return 0
    names = _stock_name_map(eng)
    out["name"] = out["ts_code"].map(lambda x: names.get(str(x), ""))
    cols_order = [
        "ts_code",
        "name",
        "trade_date",
        "upper_band",
        "middle_band",
        "lower_band",
        "band_width",
        "band_width_zscore",
        "price_position",
        "upper_return",
        "lower_return",
        "v_upper_band",
        "v_middle_band",
        "v_lower_band",
        "v_band_width",
        "v_band_width_zscore",
        "v_price_position",
    ]
    out = out[cols_order]
    cols = _table_columns(eng, "stock_bollinger_data")
    needs_replace = (len(cols) > 0) and ("ts_code" not in cols)
    if conn is None:
        with eng.begin() as _conn:
            if _table_exists(eng, "stock_bollinger_data") and not needs_replace:
                date_col = "trade_date" if "trade_date" in cols else ("date" if "date" in cols else None)
                if date_col is not None:
                    _conn.execute(text(f"delete from stock_bollinger_data where {date_col}=:d"), {"d": trade_date})
    else:
        if _table_exists(eng, "stock_bollinger_data") and not needs_replace:
            date_col = "trade_date" if "trade_date" in cols else ("date" if "date" in cols else None)
            if date_col is not None:
                conn.execute(text(f"delete from stock_bollinger_data where {date_col}=:d"), {"d": trade_date})
    out_to_write = out.rename(columns={"trade_date": "date"}) if ("trade_date" not in cols and "date" in cols) else out
    target = conn if conn is not None else eng
    out_to_write.to_sql("stock_bollinger_data", target, if_exists=("replace" if needs_replace else "append"), index=False)
    return len(out)


def compute_concept_bollinger_for_date(trade_date: date, conn=None) -> int:
    eng = get_engine()
    if eng is None:
        return 0
    if not _table_exists(eng, "stock_bollinger_data"):
        return 0
    cols_sbd = _table_columns(eng, "stock_bollinger_data")
    date_col = "trade_date" if "trade_date" in cols_sbd else ("date" if "date" in cols_sbd else None)
    if date_col is None:
        return 0
    if conn is not None:
        sbd = pd.read_sql(
            text(
                f"select ts_code, band_width_zscore, price_position, v_band_width_zscore, v_price_position "
                f"from stock_bollinger_data where {date_col}=:d"
            ),
            conn,
            params={"d": trade_date},
        )
    else:
        with eng.connect() as _c:
            sbd = pd.read_sql(
                text(
                    f"select ts_code, band_width_zscore, price_position, v_band_width_zscore, v_price_position "
                    f"from stock_bollinger_data where {date_col}=:d"
                ),
                _c,
                params={"d": trade_date},
            )
    if sbd.empty:
        return 0
    if conn is not None:
        if not _table_exists(eng, "concepts_cache"):
            return 0
        cc = pd.read_sql(
            "select ts_code, concepts from concepts_cache",
            conn,
        )
    else:
        with eng.connect() as _c:
            if not _table_exists(eng, "concepts_cache"):
                return 0
            cc = pd.read_sql(
                "select ts_code, concepts from concepts_cache",
                _c,
            )
    cc["concepts"] = cc["concepts"].fillna("")
    rows = cc["concepts"].str.split(";")
    cc_exp = cc.loc[rows.index.repeat(rows.str.len())].copy()
    cc_exp["concept_name"] = np.concatenate(rows.values)
    cc_exp = cc_exp[cc_exp["concept_name"].str.len() > 0]
    joined = sbd.merge(cc_exp[["ts_code", "concept_name"]], on="ts_code", how="inner")
    if joined.empty:
        return 0
    grp = joined.groupby(["concept_name"], sort=False)
    out = grp.agg(
        median_bandwidth_zscore=("band_width_zscore", "median"),
        median_price_position=("price_position", "median"),
        median_v_bandwidth_zscore=("v_band_width_zscore", "median"),
        median_v_price_position=("v_price_position", "median"),
        stock_count=("ts_code", "nunique"),
    ).reset_index()
    out["trade_date"] = trade_date
    cols = [
        "concept_name",
        "trade_date",
        "median_bandwidth_zscore",
        "median_price_position",
        "median_v_bandwidth_zscore",
        "median_v_price_position",
        "stock_count",
    ]
    out = out[cols]
    cols_exist = _table_columns(eng, "concept_bollinger_data")
    needs_replace_c = (len(cols_exist) > 0) and ("concept_name" not in cols_exist)
    if conn is None:
        with eng.begin() as _conn:
            if _table_exists(eng, "concept_bollinger_data") and not needs_replace_c:
                dc = "trade_date" if "trade_date" in cols_exist else ("date" if "date" in cols_exist else None)
                if dc is not None:
                    _conn.execute(text(f"delete from concept_bollinger_data where {dc}=:d"), {"d": trade_date})
    else:
        if _table_exists(eng, "concept_bollinger_data") and not needs_replace_c:
            dc = "trade_date" if "trade_date" in cols_exist else ("date" if "date" in cols_exist else None)
            if dc is not None:
                conn.execute(text(f"delete from concept_bollinger_data where {dc}=:d"), {"d": trade_date})
    out_to_write = out.rename(columns={"trade_date": "date"}) if ("trade_date" not in cols_exist and "date" in cols_exist) else out
    target = conn if conn is not None else eng
    out_to_write.to_sql("concept_bollinger_data", target, if_exists=("replace" if needs_replace_c else "append"), index=False)
    return len(out)


def _fetch_stock_daily_range(eng, start_date: date, end_date: date, conn=None) -> pd.DataFrame:
    if conn is not None:
        df = pd.read_sql(
            text(
                "select ts_code, trade_date, close, volume, amount "
                "from stock_daily where trade_date>=:d1 and trade_date<=:d2"
            ),
            conn,
            params={"d1": start_date, "d2": end_date},
        )
    else:
        with eng.connect() as _c:
            df = pd.read_sql(
                text(
                    "select ts_code, trade_date, close, volume, amount "
                    "from stock_daily where trade_date>=:d1 and trade_date<=:d2"
                ),
                _c,
                params={"d1": start_date, "d2": end_date},
            )
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.sort_values(["ts_code", "trade_date"]) 


def compute_stock_bollinger_from_db_range(start_date: date, end_date: date, window: int = 20, k: float = 2.0, z_window: int = 120, conn=None) -> int:
    eng = get_engine()
    if eng is None:
        return 0
    df = _fetch_stock_daily_range(eng, start_date, end_date, conn=conn)
    if df.empty:
        return 0
    grp = df.groupby("ts_code", sort=False)
    mid = grp["close"].transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    std = grp["close"].transform(lambda s: s.rolling(window=window, min_periods=window).std(ddof=0))
    up = mid + k * std
    lo = mid - k * std
    bw = (up - lo)
    pos = ((df["close"] - lo) / (up - lo)) * 100.0
    ur = (up / mid - 1.0) * 100.0
    lr = (lo / mid - 1.0) * 100.0

    v_mid = grp["volume"].transform(lambda s: s.fillna(0.0).rolling(window=window, min_periods=window).mean())
    v_std = grp["volume"].transform(lambda s: s.fillna(0.0).rolling(window=window, min_periods=window).std(ddof=0))
    v_up = v_mid + k * v_std
    v_lo = v_mid - k * v_std
    v_bw = (v_up - v_lo)
    v_pos = ((df["volume"] - v_lo) / (v_up - v_lo)) * 100.0

    bw_z = bw.groupby(df["ts_code"]).transform(lambda s: (s - s.rolling(window=z_window, min_periods=z_window).mean()) / s.rolling(window=z_window, min_periods=z_window).std(ddof=0))
    v_bw_z = v_bw.groupby(df["ts_code"]).transform(lambda s: (s - s.rolling(window=z_window, min_periods=z_window).mean()) / s.rolling(window=z_window, min_periods=z_window).std(ddof=0))

    out = pd.DataFrame({
        "ts_code": df["ts_code"],
        "trade_date": df["trade_date"],
        "upper_band": up,
        "middle_band": mid,
        "lower_band": lo,
        "band_width": bw,
        "band_width_zscore": bw_z,
        "price_position": pos,
        "upper_return": ur,
        "lower_return": lr,
        "v_upper_band": v_up,
        "v_middle_band": v_mid,
        "v_lower_band": v_lo,
        "v_band_width": v_bw,
        "v_band_width_zscore": v_bw_z,
        "v_price_position": v_pos,
    })
    names = _stock_name_map(eng)
    out["name"] = out["ts_code"].map(lambda x: names.get(str(x), ""))
    cols = _table_columns(eng, "stock_bollinger_data")
    needs_replace = (len(cols) > 0) and ("ts_code" not in cols)
    date_col = "trade_date" if "trade_date" in cols else ("date" if "date" in cols else None)
    if conn is None:
        with eng.begin() as _conn:
            if _table_exists(eng, "stock_bollinger_data") and not needs_replace and date_col is not None:
                _conn.execute(text(f"delete from stock_bollinger_data where {date_col}>=:d1 and {date_col}<=:d2"), {"d1": start_date, "d2": end_date})
            out_to_write = out.rename(columns={"trade_date": "date"}) if ("trade_date" not in cols and "date" in cols) else out
            out_to_write.to_sql("stock_bollinger_data", _conn, if_exists=("replace" if needs_replace else "append"), index=False)
            return len(out)
    else:
        if _table_exists(eng, "stock_bollinger_data") and not needs_replace and date_col is not None:
            conn.execute(text(f"delete from stock_bollinger_data where {date_col}>=:d1 and {date_col}<=:d2"), {"d1": start_date, "d2": end_date})
        out_to_write = out.rename(columns={"trade_date": "date"}) if ("trade_date" not in cols and "date" in cols) else out
        out_to_write.to_sql("stock_bollinger_data", conn, if_exists=("replace" if needs_replace else "append"), index=False)
        return len(out)


def compute_concept_bollinger_from_db_range(start_date: date, end_date: date, conn=None) -> int:
    eng = get_engine()
    if eng is None:
        return 0
    cols_sbd = _table_columns(eng, "stock_bollinger_data")
    date_col = "trade_date" if "trade_date" in cols_sbd else ("date" if "date" in cols_sbd else None)
    if date_col is None:
        return 0
    if conn is not None:
        sbd = pd.read_sql(text(f"select ts_code, {date_col} as trade_date, band_width_zscore, price_position, v_band_width_zscore, v_price_position from stock_bollinger_data where {date_col}>=:d1 and {date_col}<=:d2"), conn, params={"d1": start_date, "d2": end_date})
    else:
        with eng.connect() as _c:
            sbd = pd.read_sql(text(f"select ts_code, {date_col} as trade_date, band_width_zscore, price_position, v_band_width_zscore, v_price_position from stock_bollinger_data where {date_col}>=:d1 and {date_col}<=:d2"), _c, params={"d1": start_date, "d2": end_date})
    if sbd.empty:
        return 0
    if conn is not None:
        if not _table_exists(eng, "concepts_cache"):
            return 0
        cc = pd.read_sql("select ts_code, concepts from concepts_cache", conn)
    else:
        with eng.connect() as _c:
            if not _table_exists(eng, "concepts_cache"):
                return 0
            cc = pd.read_sql("select ts_code, concepts from concepts_cache", _c)
    cc["concepts"] = cc["concepts"].fillna("")
    rows = cc["concepts"].str.split(";")
    cc_exp = cc.loc[rows.index.repeat(rows.str.len())].copy()
    cc_exp["concept_name"] = np.concatenate(rows.values)
    cc_exp = cc_exp[cc_exp["concept_name"].str.len() > 0]
    joined = sbd.merge(cc_exp[["ts_code", "concept_name"]], on="ts_code", how="inner")
    if joined.empty:
        return 0
    grp = joined.groupby(["concept_name", "trade_date"], sort=False)
    out = grp.agg(
        median_bandwidth_zscore=("band_width_zscore", "median"),
        median_price_position=("price_position", "median"),
        median_v_bandwidth_zscore=("v_band_width_zscore", "median"),
        median_v_price_position=("v_price_position", "median"),
        stock_count=("ts_code", "nunique"),
    ).reset_index()
    cols = _table_columns(eng, "concept_bollinger_data")
    needs_replace_c = (len(cols) > 0) and ("concept_name" not in cols)
    date_col_out = "trade_date" if "trade_date" in cols else ("date" if "date" in cols else None)
    if conn is None:
        with eng.begin() as _conn:
            if _table_exists(eng, "concept_bollinger_data") and not needs_replace_c and date_col_out is not None:
                _conn.execute(text(f"delete from concept_bollinger_data where {date_col_out}>=:d1 and {date_col_out}<=:d2"), {"d1": start_date, "d2": end_date})
            out_to_write = out.rename(columns={"trade_date": "date"}) if ("trade_date" not in cols and "date" in cols) else out
            out_to_write.to_sql("concept_bollinger_data", _conn, if_exists=("replace" if needs_replace_c else "append"), index=False)
            return len(out)
    else:
        if _table_exists(eng, "concept_bollinger_data") and not needs_replace_c and date_col_out is not None:
            conn.execute(text(f"delete from concept_bollinger_data where {date_col_out}>=:d1 and {date_col_out}<=:d2"), {"d1": start_date, "d2": end_date})
        out_to_write = out.rename(columns={"trade_date": "date"}) if ("trade_date" not in cols and "date" in cols) else out
        out_to_write.to_sql("concept_bollinger_data", conn, if_exists=("replace" if needs_replace_c else "append"), index=False)
        return len(out)


def compute_stock_bollinger_between(start_date: date, end_date: date, window: int = 20, k: float = 2.0, z_window: int = 120, ts_codes: Optional[List[str]] = None) -> int:
    eng = get_engine()
    if eng is None:
        return 0
    if ts_codes is None:
        basics = repository.get_all_stock_basics()
        ts_codes = [s.ts_code for s in basics]
    lookback = max(window, z_window) + 5
    df = _fetch_stock_daily(eng, ts_codes, end_date, lookback)
    if df.empty:
        return 0
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)].copy()
    if df.empty:
        return 0
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    grp = df.groupby("ts_code", sort=False)
    mid = grp["close"].transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    std = grp["close"].transform(lambda s: s.rolling(window=window, min_periods=window).std(ddof=0))
    up = mid + k * std
    lo = mid - k * std
    bw = (up - lo)
    pos = ((df["close"] - lo) / (up - lo)) * 100.0
    ur = (up / mid - 1.0) * 100.0
    lr = (lo / mid - 1.0) * 100.0
    v_mid = grp["volume"].transform(lambda s: s.fillna(0.0).rolling(window=window, min_periods=window).mean())
    v_std = grp["volume"].transform(lambda s: s.fillna(0.0).rolling(window=window, min_periods=window).std(ddof=0))
    v_up = v_mid + k * v_std
    v_lo = v_mid - k * v_std
    v_bw = (v_up - v_lo)
    v_pos = ((df["volume"] - v_lo) / (v_up - v_lo)) * 100.0
    bw_z = bw.groupby(df["ts_code"]).transform(lambda s: (s - s.rolling(window=z_window, min_periods=z_window).mean()) / s.rolling(window=z_window, min_periods=z_window).std(ddof=0))
    v_bw_z = v_bw.groupby(df["ts_code"]).transform(lambda s: (s - s.rolling(window=z_window, min_periods=z_window).mean()) / s.rolling(window=z_window, min_periods=z_window).std(ddof=0))
    out = pd.DataFrame({
        "ts_code": df["ts_code"],
        "trade_date": df["trade_date"],
        "upper_band": up,
        "middle_band": mid,
        "lower_band": lo,
        "band_width": bw,
        "band_width_zscore": bw_z,
        "price_position": pos,
        "upper_return": ur,
        "lower_return": lr,
        "v_upper_band": v_up,
        "v_middle_band": v_mid,
        "v_lower_band": v_lo,
        "v_band_width": v_bw,
        "v_band_width_zscore": v_bw_z,
        "v_price_position": v_pos,
    })
    names = _stock_name_map(eng)
    out["name"] = out["ts_code"].map(lambda x: names.get(str(x), ""))
    cols = _table_columns(eng, "stock_bollinger_data")
    needs_replace = (len(cols) > 0) and ("ts_code" not in cols)
    with eng.begin() as conn:
        if _table_exists(eng, "stock_bollinger_data") and not needs_replace:
            date_col = "trade_date" if "trade_date" in cols else ("date" if "date" in cols else None)
            if date_col is not None:
                conn.execute(text(f"delete from stock_bollinger_data where {date_col}>=:d1 and {date_col}<=:d2"), {"d1": start_date, "d2": end_date})
    out_to_write = out.rename(columns={"trade_date": "date"}) if ("trade_date" not in cols and "date" in cols) else out
    out_to_write.to_sql("stock_bollinger_data", eng, if_exists=("replace" if needs_replace else "append"), index=False)
    return len(out)


def compute_concept_bollinger_between(start_date: date, end_date: date) -> int:
    eng = get_engine()
    if eng is None:
        return 0
    if not _table_exists(eng, "stock_bollinger_data"):
        return 0
    cols_sbd = _table_columns(eng, "stock_bollinger_data")
    date_col = "trade_date" if "trade_date" in cols_sbd else ("date" if "date" in cols_sbd else None)
    if date_col is None:
        return 0
    with eng.connect() as conn:
        sbd = pd.read_sql(text(f"select ts_code, {date_col} as trade_date, band_width_zscore, price_position, v_band_width_zscore, v_price_position from stock_bollinger_data where {date_col}>=:d1 and {date_col}<=:d2"), conn, params={"d1": start_date, "d2": end_date})
    if sbd.empty:
        return 0
    with eng.connect() as conn:
        if not _table_exists(eng, "concepts_cache"):
            return 0
        cc = pd.read_sql("select ts_code, concepts from concepts_cache", conn)
    cc["concepts"] = cc["concepts"].fillna("")
    rows = cc["concepts"].str.split(";")
    cc_exp = cc.loc[rows.index.repeat(rows.str.len())].copy()
    cc_exp["concept_name"] = np.concatenate(rows.values)
    cc_exp = cc_exp[cc_exp["concept_name"].str.len() > 0]
    joined = sbd.merge(cc_exp[["ts_code", "concept_name"]], on="ts_code", how="inner")
    if joined.empty:
        return 0
    grp = joined.groupby(["concept_name", "trade_date"], sort=False)
    out = grp.agg(
        median_bandwidth_zscore=("band_width_zscore", "median"),
        median_price_position=("price_position", "median"),
        median_v_bandwidth_zscore=("v_band_width_zscore", "median"),
        median_v_price_position=("v_price_position", "median"),
        stock_count=("ts_code", "nunique"),
    ).reset_index()
    cols = _table_columns(eng, "concept_bollinger_data")
    needs_replace_c = (len(cols) > 0) and ("concept_name" not in cols)
    with eng.begin() as conn:
        if _table_exists(eng, "concept_bollinger_data") and not needs_replace_c:
            date_col_out = "trade_date" if "trade_date" in cols else ("date" if "date" in cols else None)
            if date_col_out is not None:
                conn.execute(text(f"delete from concept_bollinger_data where {date_col_out}>=:d1 and {date_col_out}<=:d2"), {"d1": start_date, "d2": end_date})
    out_to_write = out.rename(columns={"trade_date": "date"}) if ("trade_date" not in cols and "date" in cols) else out
    out_to_write.to_sql("concept_bollinger_data", eng, if_exists=("replace" if needs_replace_c else "append"), index=False)
    return len(out)


if __name__ == "__main__":
    td = date.today()
    n1 = compute_stock_bollinger_for_date(td)
    n2 = compute_concept_bollinger_for_date(td)
    print(n1, n2)
