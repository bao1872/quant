from __future__ import annotations

import pandas as pd

from data.hybrid_source import HybridDataSource


def _interval_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or ("datetime" not in df.columns):
        return pd.DataFrame()
    d = pd.to_datetime(df["datetime"]).sort_values()
    diff = d.diff().dropna()
    return diff.value_counts().rename_axis("delta").reset_index(name="count")


def main() -> None:
    code = "000001.SZ"
    ds = HybridDataSource()
    d1 = ds.get_daily_bars(code, count=50)
    m1 = ds.get_minute_bars(code, freq="60m", count=50)
    print("Daily rows:", len(d1))
    print("Minute rows:", len(m1))
    print("Daily interval counts:")
    print(_interval_counts(d1).head())
    print("Minute interval counts:")
    print(_interval_counts(m1).head())


if __name__ == "__main__":
    main()