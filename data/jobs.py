# data/jobs.py
"""
定时任务入口：

- job_update_ohlc: 盘前/盘后更新日线 & 分钟线
- job_collect_ticks_once: 测试用 tick 抓取
- job_finalize_ticks_and_levels: 收盘后（或盘前）计算关键位
"""

from __future__ import annotations

from datetime import date as _date
from datetime import timedelta as _timedelta

from .updater import update_daily_bars, update_minute_bars, collect_full_day_ticks
from .concepts_cache import update_concepts_cache, update_hk_industry_cache, validate_concepts_cache_count
from factors.bollinger import compute_stock_bollinger_for_date, compute_concept_bollinger_for_date, _table_columns, compute_stock_bollinger_from_db_range, compute_concept_bollinger_from_db_range
from db.connection import get_engine
from sqlalchemy import text
from .repository import get_all_stock_basics
from factors import AbuPriceLevelProvider
import pandas as pd
from tqdm import tqdm
from sqlalchemy import inspect
from data.tick_store import TickStore
from datetime import date as date_cls


def job_update_ohlc(trade_date: _date) -> None:
    print(f"[job_update_ohlc] trade_date={trade_date}")
    update_daily_bars(trade_date=trade_date, count=600)


def job_collect_full_day_ticks(trade_date: _date) -> None:
    print(f"[job_collect_full_day_ticks] trade_date={trade_date}")
    collect_full_day_ticks(trade_date)


def job_finalize_ticks_and_levels(trade_date: _date) -> None:
    print(f"[job_finalize_ticks_and_levels] trade_date={trade_date}")
    provider = AbuPriceLevelProvider()
    provider.precompute(trade_date)


def job_update_concepts() -> None:
    print("[job_update_concepts] start")
    n1 = update_concepts_cache()
    n2 = update_hk_industry_cache()
    print(f"[job_update_concepts] a={n1} hk={n2}")
    from .concepts_cache import reconcile_concepts_cache_with_universe, purge_non_universe_from_concepts_cache
    fixed = reconcile_concepts_cache_with_universe()
    removed = purge_non_universe_from_concepts_cache()
    if fixed > 0 or removed > 0:
        print(f"[job_update_concepts] reconciled missing={fixed} removed_extra={removed}")
    validate_concepts_cache_count()


def job_update_bollinger(trade_date: _date) -> None:
    """
    仅计算并落库价格与成交量相关的布林带指标（不含任何人气相关逻辑）：
    - 价格布林线：upper/middle/lower/band_width/price_position/upper_return/lower_return
    - 成交量加权布林线：v_upper/v_middle/v_lower/v_band_width/v_price_position 及 zscore
    - 概念层聚合：按概念对当日个股布林指标取中位数与数量
    """
    print(f"[job_update_bollinger] trade_date={trade_date}")
    eng = get_engine()
    with eng.connect() as conn:
        df_daily = pd.read_sql("select distinct trade_date from stock_daily", conn)
        daily_dates = set(pd.to_datetime(df_daily["trade_date"]).dt.date.tolist()) if not df_daily.empty else set()
        cols_s = _table_columns(eng, "stock_bollinger_data")
        dc_s = "trade_date" if "trade_date" in cols_s else ("date" if "date" in cols_s else None)
        sbd_dates = set()
        if dc_s is not None and cols_s:
            df_s = pd.read_sql(f"select distinct {dc_s} as d from stock_bollinger_data", conn)
            sbd_dates = set(pd.to_datetime(df_s["d"]).dt.date.tolist()) if not df_s.empty else set()
    missing_hist = sorted(daily_dates - sbd_dates - {trade_date})
    if missing_hist:
        print(f"[job_update_bollinger] backfill_missing_days={len(missing_hist)}")
        with eng.begin() as conn:
            bar = tqdm(total=len(missing_hist), desc="backfill", unit="day")
            ins_s = 0
            ins_c = 0
            for d0 in missing_hist:
                ins_s += compute_stock_bollinger_for_date(d0, conn=conn)
                ins_c += compute_concept_bollinger_for_date(d0, conn=conn)
                bar.update(1)
            bar.close()
        print(f"[job_update_bollinger] backfill inserted stock_rows={ins_s} concept_rows={ins_c}")
    
    with eng.begin() as conn:
        cols_s = _table_columns(eng, "stock_bollinger_data")
        dc_s = "trade_date" if "trade_date" in cols_s else ("date" if "date" in cols_s else None)
        deleted_s = 0
        if dc_s is not None and cols_s:
            r = conn.execute(text(f"delete from stock_bollinger_data where {dc_s}=:d"), {"d": trade_date})
            deleted_s = r.rowcount or 0
        cols_c = _table_columns(eng, "concept_bollinger_data")
        dc_c = "trade_date" if "trade_date" in cols_c else ("date" if "date" in cols_c else None)
        deleted_c = 0
        if dc_c is not None and cols_c:
            r2 = conn.execute(text(f"delete from concept_bollinger_data where {dc_c}=:d"), {"d": trade_date})
            deleted_c = r2.rowcount or 0
        n1 = compute_stock_bollinger_for_date(trade_date, conn=conn)
        n2 = compute_concept_bollinger_for_date(trade_date, conn=conn)
        print(f"[job_update_bollinger] deleted stock_rows={deleted_s} concept_rows={deleted_c}")
        print(f"[job_update_bollinger] inserted stock_rows={n1} concept_rows={n2}")


def job_backfill_bollinger_from_db(z_window: int = 120) -> None:
    eng = get_engine()
    with eng.connect() as conn:
        rng = conn.execute(text("select min(trade_date) as d1, max(trade_date) as d2 from stock_daily")).mappings().first()
        df_dates = pd.read_sql("select distinct trade_date from stock_daily order by trade_date", conn)
    d1 = rng["d1"]
    d2 = rng["d2"]
    all_days = pd.to_datetime(df_dates["trade_date"]).dt.date.tolist() if not df_dates.empty else []
    print(f"[job_backfill_bollinger_from_db] range {d1}~{d2} days={len(all_days)}")
    if not all_days:
        return
    start_2y = d2 - _timedelta(days=730)
    days_2y = [d for d in all_days if d >= start_2y]
    print(f"[job_backfill_bollinger_from_db] limit to last_2y {start_2y}~{d2} days={len(days_2y)}")
    cols_s = _table_columns(eng, "stock_bollinger_data")
    dc_s = "trade_date" if "trade_date" in cols_s else ("date" if "date" in cols_s else None)
    cols_c = _table_columns(eng, "concept_bollinger_data")
    dc_c = "trade_date" if "trade_date" in cols_c else ("date" if "date" in cols_c else None)
    start_idx = (z_window - 1) if len(days_2y) >= z_window else len(days_2y)
    work_days = days_2y[start_idx:]
    print(f"[job_backfill_bollinger_from_db] start_index={start_idx} work_days={len(work_days)}")
    if not work_days:
        print("[job_backfill_bollinger_from_db] no work due to insufficient history")
        return
    bar = tqdm(total=len(work_days), desc="backfill", unit="day", mininterval=1)
    ins_s = 0
    ins_c = 0
    for d0 in work_days:
        with eng.begin() as conn:
            if dc_s is not None and cols_s:
                conn.execute(text(f"delete from stock_bollinger_data where {dc_s}=:d"), {"d": d0})
            if dc_c is not None and cols_c:
                conn.execute(text(f"delete from concept_bollinger_data where {dc_c}=:d"), {"d": d0})
            ins_s += compute_stock_bollinger_for_date(d0, conn=conn, z_window=z_window)
            ins_c += compute_concept_bollinger_for_date(d0, conn=conn)
        bar.update(1)
    bar.close()
    print(f"[job_backfill_bollinger_from_db] inserted stock_rows={ins_s} concept_rows={ins_c}")


def job_update_bollinger_range(start_date: _date, end_date: _date) -> None:
    print(f"[job_update_bollinger_range] {start_date}~{end_date}")
    from factors.bollinger import compute_stock_bollinger_between, compute_concept_bollinger_between
    n1 = compute_stock_bollinger_between(start_date, end_date)
    n2 = compute_concept_bollinger_between(start_date, end_date)
    print(f"[job_update_bollinger_range] stock_rows={n1} concept_rows={n2}")


def build_active_pool(trade_date: _date, window_days: int = 20, ppos_threshold: float = 50.0, min_avg_amount: float = 100_000_000.0) -> int:
    print(f"[build_active_pool] trade_date={trade_date} window_days={window_days}")
    eng = get_engine()
    d_start = pd.Timestamp(trade_date) - pd.Timedelta(days=window_days - 1)
    with eng.connect() as conn:
        cols_s = _table_columns(eng, "stock_bollinger_data")
        dc_s = "trade_date" if "trade_date" in cols_s else ("date" if "date" in cols_s else None)
        if not cols_s or dc_s is None:
            return 0
        # 仅当日记录，用于事件日同日量价齐升
        df_sbd_day = pd.read_sql(
            text(f"select ts_code, v_price_position, price_position from stock_bollinger_data where {dc_s}=:d"),
            conn,
            params={"d": trade_date},
        )
    if df_sbd_day.empty:
        with eng.begin() as conn:
            inspector = inspect(conn)
            if inspector.has_table("stock_active_pool", schema="public"):
                conn.execute(text("delete from stock_active_pool where trade_date=:d"), {"d": trade_date})
        print("[build_active_pool] no candidates")
        return 0
    df_sbd_day["has_both"] = (pd.to_numeric(df_sbd_day["v_price_position"], errors="coerce") > 99.0) & (
        pd.to_numeric(df_sbd_day["price_position"], errors="coerce") > float(ppos_threshold)
    )
    with eng.connect() as conn:
        df_daily = pd.read_sql(
            text("select ts_code, trade_date as d, amount from stock_daily where trade_date>=:d1 and trade_date<=:d2"),
            conn,
            params={"d1": d_start.date(), "d2": trade_date},
        )
    if df_daily.empty:
        return 0
    df_daily["d"] = pd.to_datetime(df_daily["d"])
    df_amt = df_daily.groupby("ts_code", as_index=False)["amount"].mean().rename(columns={"amount": "avg_amount"})
    df = df_sbd_day.merge(df_amt, on="ts_code", how="left")
    df["avg_amount"] = pd.to_numeric(df["avg_amount"], errors="coerce").fillna(0.0)
    df = df[(df["has_both"]) & (df["avg_amount"] > float(min_avg_amount))]
    if df.empty:
        with eng.begin() as conn:
            inspector = inspect(conn)
            if inspector.has_table("stock_active_pool", schema="public"):
                conn.execute(text("delete from stock_active_pool where trade_date=:d"), {"d": trade_date})
        print("[build_active_pool] no candidates")
        return 0
    df_out = df.copy()
    df_out["trade_date"] = pd.Timestamp(trade_date)
    df_out["window_days"] = int(window_days)
    df_out["remarks"] = ""
    df_out["has_high_vpos"] = True
    df_out["has_high_ppos"] = True
    df_out.rename(columns={"v_price_position": "max_vpos", "price_position": "max_ppos"}, inplace=True)
    cols = ["ts_code", "trade_date", "window_days", "has_high_vpos", "has_high_ppos", "max_vpos", "max_ppos", "avg_amount", "remarks"]
    df_out = df_out[cols].reset_index(drop=True)
    with eng.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("stock_active_pool", schema="public"):
            conn.execute(text("delete from stock_active_pool where trade_date=:d"), {"d": trade_date})
        df_out.to_sql("stock_active_pool", conn, if_exists="append", index=False)
    print(f"[build_active_pool] inserted rows={len(df_out)}")
    return len(df_out)


def summarize_microstructure_for_stock(ts_code: str, trade_date: _date, q: float) -> dict:
    store = TickStore()
    df_ticks = store.load_ticks(ts_code, trade_date)
    if df_ticks is None or df_ticks.empty:
        return {
            "ts_code": ts_code,
            "trade_date": pd.Timestamp(trade_date),
            "total_value": 0.0,
            "total_volume": 0,
            "vwap": 0.0,
            "big_buy_value": 0.0,
            "big_sell_value": 0.0,
            "big_net_value": 0.0,
            "big_net_ratio": 0.0,
            "big_buy_ratio": 0.0,
            "big_sell_ratio": 0.0,
            "big_net_open": 0.0,
            "big_net_mid": 0.0,
            "big_net_close": 0.0,
            "order_flow_imbalance": 0.0,
            "aggressive_trade_count_ratio": 0.0,
        }
    df = df_ticks.copy()
    dt_col = next((c for c in ["datetime", "time", "timestamp"] if c in df.columns), None)
    if dt_col is None:
        return {
            "ts_code": ts_code,
            "trade_date": pd.Timestamp(trade_date),
            "total_value": 0.0,
            "total_volume": 0,
            "vwap": 0.0,
            "big_buy_value": 0.0,
            "big_sell_value": 0.0,
            "big_net_value": 0.0,
            "big_net_ratio": 0.0,
            "big_buy_ratio": 0.0,
            "big_sell_ratio": 0.0,
            "big_net_open": 0.0,
            "big_net_mid": 0.0,
            "big_net_close": 0.0,
            "order_flow_imbalance": 0.0,
            "aggressive_trade_count_ratio": 0.0,
        }
    df["datetime"] = pd.to_datetime(df[dt_col])
    price_col = next((c for c in ["price", "last", "成交价", "new_price"] if c in df.columns), None)
    volume_col = next((c for c in ["volume", "vol", "qty", "成交量"] if c in df.columns), None)
    amount_col = next((c for c in ["amount", "turnover", "成交额"] if c in df.columns), None)
    price_series = pd.to_numeric(df[price_col], errors="coerce") if price_col is not None else pd.Series(0.0, index=df.index)
    if amount_col is not None:
        amount_series = pd.to_numeric(df[amount_col], errors="coerce")
    else:
        amount_series = pd.Series(0.0, index=df.index)
    if volume_col is not None:
        volume_series = pd.to_numeric(df[volume_col], errors="coerce")
    else:
        if (amount_col is not None) and (price_col is not None):
            volume_series = amount_series / price_series.replace(0, pd.NA)
            volume_series = pd.to_numeric(volume_series, errors="coerce").fillna(0.0)
        else:
            volume_series = pd.Series(0.0, index=df.index)
    df["trade_value"] = amount_series.where(amount_series > 0, price_series * volume_series)
    side_col = next((c for c in ["side", "buyorsell", "bsflag"] if c in df.columns), None)
    if side_col is None:
        df["side"] = "N"
    else:
        vals = df[side_col]
        if pd.api.types.is_numeric_dtype(vals):
            s = pd.to_numeric(vals, errors="coerce")
            df["side"] = s.map({0: "B", 1: "S", 2: "N"}).fillna("N")
        else:
            bs = vals.astype(str).str.upper().str.strip()
            df["side"] = bs.map({"B": "B", "S": "S", "N": "N"}).fillna(bs.apply(lambda x: "B" if x.startswith("B") else ("S" if x.startswith("S") else "N")))
    df["is_buy"] = df["side"].astype(str).str.upper().str.startswith("B")
    df["is_sell"] = df["side"].astype(str).str.upper().str.startswith("S")
    total_value = float(df["trade_value"].sum())
    total_volume = float(pd.to_numeric(volume_series, errors="coerce").sum())
    vwap = (float(df["trade_value"].sum()) / max(total_volume, 1e-9)) if total_volume > 0 else 0.0
    trade_cnt = int(len(df))
    buy_cnt = int(df["is_buy"].sum())
    sell_cnt = int(df["is_sell"].sum())
    aggressive_trade_count_ratio = ((buy_cnt - sell_cnt) / float(trade_cnt)) if trade_cnt > 0 else 0.0
    tv = pd.to_numeric(df["trade_value"], errors="coerce").fillna(0.0)
    tv_pos = tv[tv > 0]
    Qg = float(tv_pos.quantile(q)) if len(tv_pos) > 0 else 0.0
    is_big = tv >= Qg
    big_buy_value = float(df.loc[is_big & df["is_buy"], "trade_value"].sum())
    big_sell_value = float(df.loc[is_big & df["is_sell"], "trade_value"].sum())
    big_net_value = big_buy_value - big_sell_value
    big_net_ratio = (big_net_value / max(total_value, 1e-9)) if total_value > 0 else 0.0
    big_buy_ratio = (big_buy_value / max(total_value, 1e-9)) if total_value > 0 else 0.0
    big_sell_ratio = (big_sell_value / max(total_value, 1e-9)) if total_value > 0 else 0.0
    t = df["datetime"].dt.time
    open_win = (t >= pd.to_datetime("09:30").time()) & (t < pd.to_datetime("10:00").time())
    mid_win = (t >= pd.to_datetime("10:00").time()) & (t < pd.to_datetime("14:00").time())
    close_win = (t >= pd.to_datetime("14:00").time()) & (t <= pd.to_datetime("15:00").time())
    tv_open = tv[open_win]
    tv_mid = tv[mid_win]
    tv_close = tv[close_win]
    Q_open = float(tv_open[tv_open > 0].quantile(q)) if len(tv_open) > 0 else Qg
    Q_mid = float(tv_mid[tv_mid > 0].quantile(q)) if len(tv_mid) > 0 else Qg
    Q_close = float(tv_close[tv_close > 0].quantile(q)) if len(tv_close) > 0 else Qg
    is_big_open = (tv >= Q_open) & open_win
    is_big_mid = (tv >= Q_mid) & mid_win
    is_big_close = (tv >= Q_close) & close_win
    big_net_open = float(df.loc[is_big_open & df["is_buy"], "trade_value"].sum() - df.loc[is_big_open & df["is_sell"], "trade_value"].sum())
    big_net_mid = float(df.loc[is_big_mid & df["is_buy"], "trade_value"].sum() - df.loc[is_big_mid & df["is_sell"], "trade_value"].sum())
    big_net_close = float(df.loc[is_big_close & df["is_buy"], "trade_value"].sum() - df.loc[is_big_close & df["is_sell"], "trade_value"].sum())
    buy_value = float(df.loc[df["is_buy"], "trade_value"].sum())
    sell_value = float(df.loc[df["is_sell"], "trade_value"].sum())
    order_flow_imbalance = (buy_value - sell_value) / max(total_value, 1e-9) if total_value > 0 else 0.0
    return {
        "ts_code": ts_code,
        "trade_date": pd.Timestamp(trade_date),
        "total_value": total_value,
        "total_volume": int(total_volume),
        "vwap": vwap,
        "big_buy_value": big_buy_value,
        "big_sell_value": big_sell_value,
        "big_net_value": big_net_value,
        "big_net_ratio": big_net_ratio,
        "big_buy_ratio": big_buy_ratio,
        "big_sell_ratio": big_sell_ratio,
        "big_net_open": big_net_open,
        "big_net_mid": big_net_mid,
        "big_net_close": big_net_close,
        "order_flow_imbalance": order_flow_imbalance,
        "aggressive_trade_count_ratio": aggressive_trade_count_ratio,
    }


def build_microstructure_daily(trade_date: _date, only_active_pool: bool = True, q: float = 0.8) -> int:
    print(f"[build_microstructure_daily] trade_date={trade_date} only_active_pool={only_active_pool}")
    eng = get_engine()
    if only_active_pool:
        df_pool = pd.read_sql_table("stock_active_pool", eng)
        df_pool = df_pool[df_pool["trade_date"].astype("datetime64[ns]").dt.date == trade_date]
        ts_list = [] if df_pool.empty else df_pool["ts_code"].astype(str).tolist()
    else:
        basics = get_all_stock_basics()
        ts_list = [b.ts_code for b in basics] if basics else []
    # 仅处理当日有 tick 的股票，避免大量 0 值
    if ts_list:
        with eng.connect() as conn:
            df_idx = pd.read_sql(
                text("select ts_code, record_cnt from tick_file_index where trade_date=:d"),
                conn,
                params={"d": trade_date},
            )
        if not df_idx.empty:
            idx_map = {str(r["ts_code"]): int(pd.to_numeric(r["record_cnt"], errors="coerce") or 0) for _, r in df_idx.iterrows()}
            ts_list = [ts for ts in ts_list if (ts not in idx_map) or (idx_map.get(ts, 0) > 0)]
    if not ts_list:
        with eng.begin() as conn:
            inspector = inspect(conn)
            if inspector.has_table("stock_microstructure_daily", schema="public"):
                conn.execute(text("delete from stock_microstructure_daily where trade_date=:d"), {"d": trade_date})
        print("[build_microstructure_daily] empty universe")
        return 0
    rows = []
    bar = tqdm(total=len(ts_list), desc="micro_daily", unit="stk")
    for ts in ts_list:
        rows.append(summarize_microstructure_for_stock(ts, trade_date, q))
        bar.update(1)
    bar.close()
    df_out = pd.DataFrame(rows)
    with eng.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("stock_microstructure_daily", schema="public"):
            conn.execute(text("alter table stock_microstructure_daily add column if not exists aggressive_trade_count_ratio double precision"))
            conn.execute(text("delete from stock_microstructure_daily where trade_date=:d"), {"d": trade_date})
        df_out.to_sql("stock_microstructure_daily", conn, if_exists="append", index=False)
    print(f"[build_microstructure_daily] inserted rows={len(df_out)}")
    return len(df_out)




def detect_rise_window(ts_code: str, trade_date: _date, max_lookback: int = 20, max_drawdown: float = 0.06) -> tuple[_date, _date, int]:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text("select trade_date as d, close from stock_daily where ts_code=:ts and trade_date<=:d order by trade_date desc limit :n"),
            conn,
            params={"ts": ts_code, "d": trade_date, "n": max_lookback},
        )
    if df.empty:
        return trade_date, trade_date, 1
    df = df.sort_values("d").reset_index(drop=True)
    closes = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    end_close = float(closes.iloc[-1])
    fwd_min = pd.Series(closes[::-1].cummin()[::-1].values, index=df.index)
    rr = end_close / closes
    dd = fwd_min / closes
    mask = (rr > 1.0) & (dd >= (1.0 - float(max_drawdown)))
    idxs = df.index[mask]
    start_idx = int(idxs.min()) if len(idxs) > 0 else int(max(len(df) - 5, 0))
    start_date = pd.to_datetime(df.loc[start_idx, "d"]).date()
    end_date = pd.to_datetime(df.loc[len(df) - 1, "d"]).date()
    return start_date, end_date, (len(df) - start_idx)


def compute_fund_quality_for_window(ts_code: str, window_start: _date, window_end: _date) -> dict:
    eng = get_engine()
    with eng.connect() as conn:
        df_m = pd.read_sql(
            text("select trade_date as d, total_value, big_net_value from stock_microstructure_daily where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2 order by trade_date"),
            conn,
            params={"ts": ts_code, "d1": window_start, "d2": window_end},
        )
        df_d = pd.read_sql(
            text("select trade_date as d, close, amount from stock_daily where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2 order by trade_date"),
            conn,
            params={"ts": ts_code, "d1": window_start, "d2": window_end},
        )
        cols_basic = []
        df_basic = pd.DataFrame()
        if eng is not None:
            cols_basic = _table_columns(eng, "stock_basic")
        free_float_col = None
        for c in ["free_float_value", "free_float_mv", "float_mv", "free_float_market_cap"]:
            if c in cols_basic:
                free_float_col = c
                break
        free_float_value = 0.0
        if free_float_col is not None:
            df_basic = pd.read_sql(text(f"select {free_float_col} as ff from stock_basic where ts_code=:ts"), conn, params={"ts": ts_code})
            if not df_basic.empty:
                free_float_value = float(pd.to_numeric(df_basic["ff"], errors="coerce").fillna(0.0).iloc[0])
    if df_d.empty:
        return {
            "ts_code": ts_code,
            "window_start": pd.Timestamp(window_start),
            "window_end": pd.Timestamp(window_end),
            "window_days": 0,
            "price_return": 0.0,
            "big_net_sum": 0.0,
            "turnover_ratio": 0.0,
            "pos_days_ratio": 0.0,
            "trend_corr": 0.0,
            "big_cost_gain": float("nan"),
            "support_ratio": 0.0,
            "fund_quality_score": 0.0,
            "created_at": pd.Timestamp.utcnow(),
        }
    df_d["d"] = pd.to_datetime(df_d["d"])
    df_m["d"] = pd.to_datetime(df_m["d"]) if not df_m.empty else pd.Series([], dtype="datetime64[ns]")
    days = len(df_d)
    close = pd.to_numeric(df_d["close"], errors="coerce").ffill().bfill()
    price_return = float(close.iloc[-1] / max(close.iloc[0], 1e-9) - 1.0)
    total_value_sum = float(pd.to_numeric(df_m.get("total_value", 0.0), errors="coerce").sum()) if not df_m.empty else 0.0
    big_net_sum = float(pd.to_numeric(df_m.get("big_net_value", 0.0), errors="coerce").sum()) if not df_m.empty else 0.0
    turnover_ratio = (total_value_sum / max(free_float_value, 1e-9)) if free_float_value > 0 else 0.0
    pos_days_ratio = float((pd.to_numeric(df_m.get("big_net_value", 0.0), errors="coerce") > 0).mean()) if not df_m.empty else 0.0
    cum_big = pd.to_numeric(df_m.get("big_net_value", 0.0), errors="coerce").cumsum() if not df_m.empty else pd.Series([0.0] * days)
    # 对齐到 df_d 的日期索引
    df_join = df_d[["d"]].merge(pd.DataFrame({"d": df_m["d"], "cum_big": cum_big}), on="d", how="left").fillna(0.0)
    trend_corr = float(pd.to_numeric(df_join["cum_big"], errors="coerce").corr(pd.to_numeric(close, errors="coerce"))) if days > 1 else 0.0
    # 回调段支持率：日收益<0 的天中，big_net_value>=0 的比例
    ret = close.pct_change().fillna(0.0)
    m_map = df_m.set_index("d")["big_net_value"] if not df_m.empty else pd.Series([], dtype="float64")
    bn = df_d["d"].map(m_map).fillna(0.0)
    pull_mask = ret < 0
    support_ratio = float((bn[pull_mask] >= 0).mean()) if pull_mask.any() else 0.0
    # big_cost_gain 暂缺（需逐笔 VWAP）；置为空
    big_cost_gain = float("nan")
    # 综合分数（简版）：标准化后加权求和
    # 使用简单缩放以避免跨标的一致性问题
    score = (
        0.25 * max(min(price_return, 1.0), -1.0)
        + 0.25 * (big_net_sum / max(total_value_sum, 1e-9) if total_value_sum > 0 else 0.0)
        + 0.2 * pos_days_ratio
        + 0.2 * (trend_corr if pd.notna(trend_corr) else 0.0)
        + 0.1 * support_ratio
    )
    return {
        "ts_code": ts_code,
        "window_start": pd.Timestamp(window_start),
        "window_end": pd.Timestamp(window_end),
        "window_days": days,
        "price_return": price_return,
        "big_net_sum": big_net_sum,
        "turnover_ratio": turnover_ratio,
        "pos_days_ratio": pos_days_ratio,
        "trend_corr": trend_corr if pd.notna(trend_corr) else 0.0,
        "big_cost_gain": big_cost_gain,
        "support_ratio": support_ratio,
        "fund_quality_score": score,
        "created_at": pd.Timestamp.utcnow(),
    }


def build_fund_quality(trade_date: _date, max_lookback: int = 20) -> int:
    print(f"[build_fund_quality] trade_date={trade_date}")
    eng = get_engine()
    df_pool = pd.read_sql_table("stock_active_pool", eng)
    df_pool = df_pool[df_pool["trade_date"].astype("datetime64[ns]").dt.date == trade_date]
    ts_list = [] if df_pool.empty else df_pool["ts_code"].astype(str).tolist()
    if not ts_list:
        print("[build_fund_quality] empty universe")
        return 0
    rows = []
    bar = tqdm(total=len(ts_list), desc="fund_quality", unit="stk")
    for ts in ts_list:
        ws, we, wd = detect_rise_window(ts, trade_date, max_lookback=max_lookback)
        rows.append(compute_fund_quality_for_window(ts, ws, we))
        bar.update(1)
    bar.close()
    df_out = pd.DataFrame(rows)
    with eng.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("stock_fund_quality", schema="public"):
            conn.execute(text("delete from stock_fund_quality where window_end=:d"), {"d": trade_date})
        df_out.to_sql("stock_fund_quality", conn, if_exists="append", index=False)
    print(f"[build_fund_quality] inserted rows={len(df_out)}")
    return len(df_out)

if __name__ == "__main__":
    from datetime import date as date_cls
    import pandas as pd
    print("请选择功能：")
    print("1 构建股性活跃池")
    print("2 构建日度微结构摘要")
    print("3 运行当日布林更新")
    print("4 回补布林两年窗口")
    
    choice = input("输入选项编号：").strip()
    if choice == "1":
        ds = input("输入测试日期(YYYY-MM-DD，留空为今天)：").strip()
        d = date_cls.today() if ds == "" else pd.to_datetime(ds).date()
        ws = input("窗口天数(默认20)：").strip()
        window_days = 20 if ws == "" else int(ws)
        pt = input("price_position阈值(默认50)：").strip()
        ppos_threshold = 50.0 if pt == "" else float(pt)
        ma = input("日均成交额阈值(默认100000000)：").strip()
        min_avg_amount = 100000000.0 if ma == "" else float(ma)
        n = build_active_pool(d, window_days=window_days, ppos_threshold=ppos_threshold, min_avg_amount=min_avg_amount)
        print(f"active_pool rows: {n}")
    elif choice == "2":
        ds = input("输入测试日期(YYYY-MM-DD，留空为今天)：").strip()
        d = date_cls.today() if ds == "" else pd.to_datetime(ds).date()
        oa = input("仅处理活跃池(Y/N，默认Y)：").strip().upper()
        only_active = (oa == "" or oa == "Y")
        qs = input("分位数q(默认0.8)：").strip()
        q = 0.8 if qs == "" else float(qs)
        n = build_microstructure_daily(d, only_active_pool=only_active, q=q)
        print(f"microstructure_daily rows: {n}")
    elif choice == "3":
        ds = input("输入测试日期(YYYY-MM-DD，留空为今天)：").strip()
        d = date_cls.today() if ds == "" else pd.to_datetime(ds).date()
        job_update_bollinger(d)
    elif choice == "4":
        ws = input("zscore窗口(默认120)：").strip()
        z = 120 if ws == "" else int(ws)
        job_backfill_bollinger_from_db(z_window=z)
    
    else:
        print("无效选项")
    
