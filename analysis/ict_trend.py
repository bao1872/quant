from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


class TrendKind(IntEnum):
    BEAR = -1
    RANGE = 0
    BULL = 1


@dataclass
class TrendSegment:
    start: pd.Timestamp
    end: pd.Timestamp
    kind: TrendKind


@dataclass
class TrendResult:
    trend_series: pd.Series
    segments: List[TrendSegment]


def compute_trend_from_swings(df_ict: pd.DataFrame, min_swings: int = 4) -> TrendResult:
    sw = df_ict.get("ict_sw_highlow", pd.Series(0, index=df_ict.index)).fillna(0).astype(int)
    lv = pd.to_numeric(df_ict.get("ict_sw_level", pd.Series(np.nan, index=df_ict.index)), errors="coerce")
    dt = pd.to_datetime(df_ict["datetime"]) if "datetime" in df_ict.columns else pd.to_datetime(df_ict.index)
    swings = []
    for i in range(len(df_ict)):
        if sw.iat[i] == 0 or np.isnan(lv.iat[i]):
            continue
        swings.append({"idx": i, "dt": dt.iat[i], "kind": "H" if sw.iat[i] > 0 else "L", "price": float(lv.iat[i])})
    trend_array = np.zeros(len(df_ict), dtype=int)
    segments: List[TrendSegment] = []
    if len(swings) < int(min_swings):
        segs: List[TrendSegment] = []
        if len(df_ict) > 0:
            dt_full = pd.to_datetime(df_ict["datetime"]) if "datetime" in df_ict.columns else pd.to_datetime(df_ict.index)
            segs.append(TrendSegment(start=dt_full.iloc[0], end=dt_full.iloc[-1], kind=TrendKind.RANGE))
        trend_series = pd.Series(trend_array, index=df_ict.index, name="trend_kind")
        return TrendResult(trend_series=trend_series, segments=segs)
    last_two_highs: List[float] = []
    last_two_lows: List[float] = []
    last_trend = TrendKind.RANGE
    for k in range(len(swings) - 1):
        s = swings[k]
        nxt = swings[k + 1]
        if s["kind"] == "H":
            last_two_highs.append(s["price"])
            last_two_highs = last_two_highs[-2:]
        else:
            last_two_lows.append(s["price"])
            last_two_lows = last_two_lows[-2:]
        if len(last_two_highs) == 2 and len(last_two_lows) == 2:
            h1, h2 = last_two_highs
            l1, l2 = last_two_lows
            if h2 > h1 and l2 > l1:
                cur_trend = TrendKind.BULL
            elif h2 < h1 and l2 < l1:
                cur_trend = TrendKind.BEAR
            else:
                cur_trend = TrendKind.RANGE
        else:
            cur_trend = TrendKind.RANGE
        i0, i1 = s["idx"], nxt["idx"]
        trend_array[i0:i1] = int(cur_trend)
        if segments and segments[-1].kind == cur_trend:
            segments[-1].end = nxt["dt"]
        else:
            segments.append(TrendSegment(start=s["dt"], end=nxt["dt"], kind=cur_trend))
        last_trend = cur_trend
    last_idx = swings[-1]["idx"]
    trend_array[last_idx:] = int(last_trend)
    trend_series = pd.Series(trend_array, index=df_ict.index, name="trend_kind")
    return TrendResult(trend_series=trend_series, segments=segments)


if __name__ == "__main__":
    idx = pd.date_range("2024-01-01", periods=8, freq="1D")
    df = pd.DataFrame({
        "datetime": idx,
        "ict_sw_highlow": [1, -1, 1, -1, 1, -1, 1, -1],
        "ict_sw_level": [10, 9.6, 10.4, 9.8, 10.8, 10.0, 11.2, 10.2],
    })
    res = compute_trend_from_swings(df)
    print(len(res.segments))
    print(int(res.trend_series.sum()))