from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine
from .utils import ts_to_chan_code, ui_label_to_db_freq
from .types import Bi, Segment, Center, Signal, ChanResult


@dataclass
class ChanViewData:
    kline: pd.DataFrame
    bis: List[Bi]
    segments: List[Segment]
    centers: List[Center]
    signals: List[Signal]


def load_kline(ts_code: str, freq: str, start: datetime, end: datetime) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text(
                """
                select datetime, open, high, low, close, volume
                from stock_kline
                where ts_code=:ts and freq=:fq and datetime>=:d1 and datetime<=:d2
                order by datetime
                """
            ),
            conn,
            params={"ts": ts_code, "fq": freq, "d1": pd.to_datetime(start), "d2": pd.to_datetime(end)},
            parse_dates=["datetime"],
        )
    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].reset_index(drop=True) if not df.empty else pd.DataFrame(columns=["datetime","open","high","low","close","volume"])[:0]


def load_chan_structures(ts_code: str, freq: str, start: datetime, end: datetime) -> ChanResult:
    eng = get_engine()
    with eng.connect() as conn:
        df_bi = pd.read_sql(
            text(
                "select code,freq,bi_id,direction,start_ts,end_ts,start_price,end_price,high,low,start_fx_index,end_fx_index,length_k,amplitude,slope from stock_chan_bi where code=:c and freq=:f and start_ts>=:d1 and end_ts<=:d2 order by start_ts"
            ),
            conn,
            params={"c": ts_code, "f": freq, "d1": pd.to_datetime(start), "d2": pd.to_datetime(end)},
            parse_dates=["start_ts","end_ts"],
        ) if _table_exists("stock_chan_bi") else pd.DataFrame()
        df_seg = pd.read_sql(
            text(
                "select code,freq,seg_id,direction,start_ts,end_ts,high,low,start_bi_index,end_bi_index,bi_count,amplitude,slope,center_count from stock_chan_segment where code=:c and freq=:f and start_ts>=:d1 and end_ts<=:d2 order by start_ts"
            ),
            conn,
            params={"c": ts_code, "f": freq, "d1": pd.to_datetime(start), "d2": pd.to_datetime(end)},
            parse_dates=["start_ts","end_ts"],
        ) if _table_exists("stock_chan_segment") else pd.DataFrame()
        df_cen = pd.read_sql(
            text(
                "select code,freq,center_id,start_ts,end_ts,high,low,related_seg_index,bi_count,enter_times,leave_times from stock_chan_center where code=:c and freq=:f and start_ts>=:d1 and end_ts<=:d2 order by start_ts"
            ),
            conn,
            params={"c": ts_code, "f": freq, "d1": pd.to_datetime(start), "d2": pd.to_datetime(end)},
            parse_dates=["start_ts","end_ts"],
        ) if _table_exists("stock_chan_center") else pd.DataFrame()
        df_sig = pd.read_sql(
            text(
                "select code,freq,signal_id,ts,signal_type,price,related_bi_index,related_seg_index,related_center_index,extra_info from stock_chan_signal where code=:c and freq=:f and ts>=:d1 and ts<=:d2 order by ts"
            ),
            conn,
            params={"c": ts_code, "f": freq, "d1": pd.to_datetime(start), "d2": pd.to_datetime(end)},
            parse_dates=["ts"],
        ) if _table_exists("stock_chan_signal") else pd.DataFrame()
    bis = [
        Bi(
            code=str(r["code"]),
            freq=str(r["freq"]),
            bi_id=int(r["bi_id"]),
            direction=str(r["direction"]),
            start_ts=pd.to_datetime(r["start_ts"]),
            end_ts=pd.to_datetime(r["end_ts"]),
            start_price=float(r["start_price"]),
            end_price=float(r["end_price"]),
            high=float(r["high"]),
            low=float(r["low"]),
            start_fx_index=(None if pd.isna(r.get("start_fx_index")) else int(r.get("start_fx_index"))),
            end_fx_index=(None if pd.isna(r.get("end_fx_index")) else int(r.get("end_fx_index"))),
            length_k=(None if pd.isna(r.get("length_k")) else int(r.get("length_k"))),
            amplitude=(None if pd.isna(r.get("amplitude")) else float(r.get("amplitude"))),
            slope=(None if pd.isna(r.get("slope")) else float(r.get("slope"))),
        )
        for _, r in (df_bi if df_bi is not None else pd.DataFrame()).iterrows()
    ]
    segments = [
        Segment(
            code=str(r["code"]),
            freq=str(r["freq"]),
            seg_id=int(r["seg_id"]),
            direction=str(r["direction"]),
            start_ts=pd.to_datetime(r["start_ts"]),
            end_ts=pd.to_datetime(r["end_ts"]),
            high=float(r["high"]),
            low=float(r["low"]),
            start_bi_index=(None if pd.isna(r.get("start_bi_index")) else int(r.get("start_bi_index"))),
            end_bi_index=(None if pd.isna(r.get("end_bi_index")) else int(r.get("end_bi_index"))),
            bi_count=(None if pd.isna(r.get("bi_count")) else int(r.get("bi_count"))),
            amplitude=(None if pd.isna(r.get("amplitude")) else float(r.get("amplitude"))),
            slope=(None if pd.isna(r.get("slope")) else float(r.get("slope"))),
            center_count=(None if pd.isna(r.get("center_count")) else int(r.get("center_count"))),
        )
        for _, r in (df_seg if df_seg is not None else pd.DataFrame()).iterrows()
    ]
    centers = [
        Center(
            code=str(r["code"]),
            freq=str(r["freq"]),
            center_id=int(r["center_id"]),
            start_ts=pd.to_datetime(r["start_ts"]),
            end_ts=pd.to_datetime(r["end_ts"]),
            high=float(r["high"]),
            low=float(r["low"]),
            related_seg_index=(None if pd.isna(r.get("related_seg_index")) else int(r.get("related_seg_index"))),
            bi_count=(None if pd.isna(r.get("bi_count")) else int(r.get("bi_count"))),
            enter_times=(None if pd.isna(r.get("enter_times")) else int(r.get("enter_times"))),
            leave_times=(None if pd.isna(r.get("leave_times")) else int(r.get("leave_times"))),
        )
        for _, r in (df_cen if df_cen is not None else pd.DataFrame()).iterrows()
    ]
    signals = [
        Signal(
            code=str(r["code"]),
            freq=str(r["freq"]),
            signal_id=int(r["signal_id"]),
            ts=pd.to_datetime(r["ts"]),
            signal_type=str(r["signal_type"]),
            price=(None if pd.isna(r.get("price")) else float(r.get("price"))),
            related_bi_index=(None if pd.isna(r.get("related_bi_index")) else int(r.get("related_bi_index"))),
            related_seg_index=(None if pd.isna(r.get("related_seg_index")) else int(r.get("related_seg_index"))),
            related_center_index=(None if pd.isna(r.get("related_center_index")) else int(r.get("related_center_index"))),
            extra_info=None,
        )
        for _, r in (df_sig if df_sig is not None else pd.DataFrame()).iterrows()
    ]
    return ChanResult(code=ts_code, freq=freq, fractals=[], bis=bis, segments=segments, centers=centers, signals=signals)


def get_chan_view_data(ts_code: str, freq: str, start: datetime, end: datetime) -> ChanViewData:
    kline = load_kline(ts_code, freq, start, end)
    chan = load_chan_structures(ts_code, freq, start, end)
    print(
        "[ChanView]",
        "ts_code=", ts_code,
        "freq=", freq,
        "range=", pd.to_datetime(start), "->", pd.to_datetime(end),
        "kline_rows=", len(kline),
        "bis=", len(chan.bis),
        "segments=", len(chan.segments),
        "centers=", len(chan.centers),
        "signals=", len(chan.signals),
    )
    return ChanViewData(kline=kline, bis=chan.bis, segments=chan.segments, centers=chan.centers, signals=chan.signals)


def get_chan_view_data_realtime(ts_code: str, freq: str, kline_df: pd.DataFrame) -> ChanViewData:
    from .pipeline import run_chan_for_df
    if kline_df is None or kline_df.empty:
        return ChanViewData(kline=pd.DataFrame(columns=["datetime","open","high","low","close","volume"])[:0], bis=[], segments=[], centers=[], signals=[])
    df = kline_df.sort_values("datetime").reset_index(drop=True)
    res_map = run_chan_for_df(ts_code, freq, df)
    res = res_map.get(freq)
    if res is None:
        return ChanViewData(kline=df, bis=[], segments=[], centers=[], signals=[])
    return ChanViewData(kline=df, bis=res.bis, segments=res.segments, centers=res.centers, signals=res.signals)


if __name__ == "__main__":
    ts = "000001.SZ"
    fq = "60m"
    now = pd.Timestamp.utcnow()
    start = now - pd.Timedelta(days=60)
    end = now
    view = get_chan_view_data(ts, fq, start.to_pydatetime(), end.to_pydatetime())
    print("kline rows:", len(view.kline))
    print("bis/seg/center/signal:", len(view.bis), len(view.segments), len(view.centers), len(view.signals))
def _table_exists(table_name: str) -> bool:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text(
                "select 1 from information_schema.tables where table_schema='public' and table_name=:t"
            ),
            conn,
            params={"t": table_name},
        )
    return len(df) > 0