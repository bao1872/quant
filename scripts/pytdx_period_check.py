from __future__ import annotations

import pandas as pd
from data.pytdx_source import PytdxDataSource


def _agg_day(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.copy()
    g["date"] = pd.to_datetime(g["datetime"]).dt.date
    return (
        g.groupby("date")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .reset_index()
    )


def main() -> None:
    code = "688122.SH"
    s = PytdxDataSource()
    with open(".pytdx_period_check.out", "w", encoding="utf-8") as f:
        f.write("init\n")
    dfd = s.get_daily_bars(code, count=360)
    d60 = s.get_minute_bars(code, freq="60m", count=800)
    d30 = s.get_minute_bars(code, freq="30m", count=800)
    d15 = s.get_minute_bars(code, freq="15m", count=800)
    lines = []
    lines.append(f"lens {len(dfd)} {len(d60)} {len(d30)} {len(d15)}")
    with open(".pytdx_period_check.out", "a", encoding="utf-8") as f:
        f.write(lines[-1] + "\n")

    dfd = dfd.assign(date=pd.to_datetime(dfd["datetime"]).dt.date)
    a60 = _agg_day(d60)
    a30 = _agg_day(d30)
    a15 = _agg_day(d15)

    m60 = dfd[["date", "open", "high", "low", "close"]].merge(a60, on="date", how="inner", suffixes=("", "_60"))
    m30 = dfd[["date", "open", "high", "low", "close"]].merge(a30, on="date", how="inner", suffixes=("", "_30"))
    m15 = dfd[["date", "open", "high", "low", "close"]].merge(a15, on="date", how="inner", suffixes=("", "_15"))

    def mism(m: pd.DataFrame, a: str) -> int:
        return int(
            ((m["open"] != m[f"open_{a}"]) | (m["high"] != m[f"high_{a}"]) | (m["low"] != m[f"low_{a}"]) | (m["close"] != m[f"close_{a}"]))
            .sum()
        )

    lines.append(f"mismatch_60m {mism(m60, '60')} {len(m60)}")
    lines.append(f"mismatch_30m {mism(m30, '30')} {len(m30)}")
    lines.append(f"mismatch_15m {mism(m15, '15')} {len(m15)}")
    with open(".pytdx_period_check.out", "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()