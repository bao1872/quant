from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine
from .types import ChanResult
from .utils import chan_to_ts_code


def _ensure_chan_tables() -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(
            """
            create table if not exists public.stock_chan_bi (
                id bigserial primary key,
                code text not null,
                freq text not null,
                bi_id integer not null,
                direction text not null,
                start_ts timestamp not null,
                end_ts timestamp not null,
                start_price double precision not null,
                end_price double precision not null,
                high double precision not null,
                low double precision not null,
                start_fx_index integer,
                end_fx_index integer,
                length_k integer,
                amplitude double precision,
                slope double precision
            )
            """
        ))
        conn.execute(text(
            "create index if not exists idx_stock_chan_bi_code_freq_start_ts on public.stock_chan_bi(code, freq, start_ts)"
        ))
        conn.execute(text(
            "create index if not exists idx_stock_chan_bi_code_freq_end_ts on public.stock_chan_bi(code, freq, end_ts)"
        ))

        conn.execute(text(
            """
            create table if not exists public.stock_chan_segment (
                id bigserial primary key,
                code text not null,
                freq text not null,
                seg_id integer not null,
                direction text not null,
                start_ts timestamp not null,
                end_ts timestamp not null,
                high double precision not null,
                low double precision not null,
                start_bi_index integer,
                end_bi_index integer,
                bi_count integer,
                amplitude double precision,
                slope double precision,
                center_count integer
            )
            """
        ))
        conn.execute(text(
            "create index if not exists idx_stock_chan_segment_code_freq_start_ts on public.stock_chan_segment(code, freq, start_ts)"
        ))
        conn.execute(text(
            "create index if not exists idx_stock_chan_segment_code_freq_end_ts on public.stock_chan_segment(code, freq, end_ts)"
        ))

        conn.execute(text(
            """
            create table if not exists public.stock_chan_center (
                id bigserial primary key,
                code text not null,
                freq text not null,
                center_id integer not null,
                start_ts timestamp not null,
                end_ts timestamp not null,
                high double precision not null,
                low double precision not null,
                related_seg_index integer,
                bi_count integer,
                enter_times integer,
                leave_times integer
            )
            """
        ))
        conn.execute(text(
            "create index if not exists idx_stock_chan_center_code_freq_start_ts on public.stock_chan_center(code, freq, start_ts)"
        ))

        conn.execute(text(
            """
            create table if not exists public.stock_chan_signal (
                id bigserial primary key,
                code text not null,
                freq text not null,
                signal_id integer not null,
                ts timestamp not null,
                signal_type text not null,
                price double precision,
                related_bi_index integer,
                related_seg_index integer,
                related_center_index integer,
                extra_info text
            )
            """
        ))
        conn.execute(text(
            "create index if not exists idx_stock_chan_signal_code_freq_ts on public.stock_chan_signal(code, freq, ts)"
        ))


def _result_time_range(result: ChanResult) -> Tuple[datetime | None, datetime | None]:
    times: List[datetime] = []
    times += [b.start_ts for b in result.bis] + [b.end_ts for b in result.bis]
    times += [s.start_ts for s in result.segments] + [s.end_ts for s in result.segments]
    times += [c.start_ts for c in result.centers] + [c.end_ts for c in result.centers]
    times += [g.ts for g in result.signals]
    if not times:
        return None, None
    return min(times), max(times)


def save_chan_result_to_db(result: ChanResult) -> None:
    _ensure_chan_tables()
    eng = get_engine()

    min_ts, max_ts = _result_time_range(result)

    with eng.begin() as conn:
        if min_ts is not None and max_ts is not None:
            conn.execute(
                text("delete from stock_chan_bi where code=:c and freq=:f and start_ts>=:d1 and end_ts<=:d2"),
                {"c": result.code, "f": result.freq, "d1": min_ts, "d2": max_ts},
            )
            conn.execute(
                text("delete from stock_chan_segment where code=:c and freq=:f and start_ts>=:d1 and end_ts<=:d2"),
                {"c": result.code, "f": result.freq, "d1": min_ts, "d2": max_ts},
            )
            conn.execute(
                text("delete from stock_chan_center where code=:c and freq=:f and start_ts>=:d1 and end_ts<=:d2"),
                {"c": result.code, "f": result.freq, "d1": min_ts, "d2": max_ts},
            )
            conn.execute(
                text("delete from stock_chan_signal where code=:c and freq=:f and ts>=:d1 and ts<=:d2"),
                {"c": result.code, "f": result.freq, "d1": min_ts, "d2": max_ts},
            )

        if result.bis:
            df_bi = pd.DataFrame([
                {
                    "code": chan_to_ts_code(b.code),
                    "freq": b.freq,
                    "bi_id": b.bi_id,
                    "direction": b.direction,
                    "start_ts": b.start_ts,
                    "end_ts": b.end_ts,
                    "start_price": b.start_price,
                    "end_price": b.end_price,
                    "high": b.high,
                    "low": b.low,
                    "start_fx_index": b.start_fx_index,
                    "end_fx_index": b.end_fx_index,
                    "length_k": b.length_k,
                    "amplitude": b.amplitude,
                    "slope": b.slope,
                }
                for b in result.bis
            ])
            df_bi.to_sql("stock_chan_bi", conn, if_exists="append", index=False)

        if result.segments:
            df_seg = pd.DataFrame([
                {
                    "code": chan_to_ts_code(s.code),
                    "freq": s.freq,
                    "seg_id": s.seg_id,
                    "direction": s.direction,
                    "start_ts": s.start_ts,
                    "end_ts": s.end_ts,
                    "high": s.high,
                    "low": s.low,
                    "start_bi_index": s.start_bi_index,
                    "end_bi_index": s.end_bi_index,
                    "bi_count": s.bi_count,
                    "amplitude": s.amplitude,
                    "slope": s.slope,
                    "center_count": s.center_count,
                }
                for s in result.segments
            ])
            df_seg.to_sql("stock_chan_segment", conn, if_exists="append", index=False)

        if result.centers:
            df_cen = pd.DataFrame([
                {
                    "code": chan_to_ts_code(c.code),
                    "freq": c.freq,
                    "center_id": c.center_id,
                    "start_ts": c.start_ts,
                    "end_ts": c.end_ts,
                    "high": c.high,
                    "low": c.low,
                    "related_seg_index": c.related_seg_index,
                    "bi_count": c.bi_count,
                    "enter_times": c.enter_times,
                    "leave_times": c.leave_times,
                }
                for c in result.centers
            ])
            df_cen.to_sql("stock_chan_center", conn, if_exists="append", index=False)

        if result.signals:
            df_sig = pd.DataFrame([
                {
                    "code": chan_to_ts_code(g.code),
                    "freq": g.freq,
                    "signal_id": g.signal_id,
                    "ts": g.ts,
                    "signal_type": g.signal_type,
                    "price": g.price,
                    "related_bi_index": g.related_bi_index,
                    "related_seg_index": g.related_seg_index,
                    "related_center_index": g.related_center_index,
                    "extra_info": None if g.extra_info is None else str(g.extra_info),
                }
                for g in result.signals
            ])
            df_sig.to_sql("stock_chan_signal", conn, if_exists="append", index=False)