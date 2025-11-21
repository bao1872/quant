from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import pandas as pd
from pyharmonics.technicals import Technicals
from pyharmonics.search import HarmonicSearch


PatternFamily = Literal["ABC", "ABCD", "XABCD"]


@dataclass
class HarmonicPattern:
    family: PatternFamily
    name: str
    bullish: bool
    formed: bool
    x: List[pd.Timestamp]
    y: List[float]
    completion_min_price: float
    completion_max_price: float


def detect_harmonic_patterns(bars: pd.DataFrame, ts_code: str, interval: str) -> List[HarmonicPattern]:
    if "datetime" in bars.columns:
        df = bars.sort_values("datetime").copy()
        df = df.set_index("datetime")
    else:
        df = bars.sort_index().copy()
    req = ["open", "high", "low", "close"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"缺少字段:{miss}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    ohlc = df[["open", "high", "low", "close", "volume"]].copy()
    tech = Technicals(ohlc, ts_code, interval)
    searcher = HarmonicSearch(tech)
    searcher.search()
    raw = searcher.get_patterns()
    fam_map: Dict[PatternFamily, int] = {
        "XABCD": searcher.XABCD,
        "ABCD": searcher.ABCD,
        "ABC": searcher.ABC,
    }
    out: List[HarmonicPattern] = []
    for fam_name, key in fam_map.items():
        pts = raw.get(key, [])
        for p in pts:
            out.append(
                HarmonicPattern(
                    family=fam_name,
                    name=str(getattr(p, "name", fam_name)),
                    bullish=bool(getattr(p, "bullish", True)),
                    formed=bool(getattr(p, "formed", False)),
                    x=list(getattr(p, "x", [])),
                    y=[float(v) for v in getattr(p, "y", [])],
                    completion_min_price=float(getattr(p, "completion_min_price", 0.0)),
                    completion_max_price=float(getattr(p, "completion_max_price", 0.0)),
                )
            )
    return out


if __name__ == "__main__":
    d = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=60, freq="D"),
        "open": pd.Series(range(60)).astype(float) + 10,
        "high": pd.Series(range(60)).astype(float) + 11,
        "low": pd.Series(range(60)).astype(float) + 9,
        "close": pd.Series(range(60)).astype(float) + 10.5,
    })
    res = detect_harmonic_patterns(d, "000001.SZ", "1D")
    print(len(res))