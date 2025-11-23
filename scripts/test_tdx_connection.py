from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from data.pytdx_source import PytdxDataSource


TEST_CODES: List[str] = ["000001.SZ", "600000.SH"]
FREQS: List[str] = ["1d", "60m"]


def _fmt_range(df: pd.DataFrame) -> str:
    if df.empty or ("datetime" not in df.columns):
        return "(empty)"
    return f"{df['datetime'].min()} .. {df['datetime'].max()}"


def main() -> None:
    ds = PytdxDataSource()
    nonempty = False
    ds.connect()
    ip, port = ds.get_current_server()
    for code in TEST_CODES:
        for fq in FREQS:
            print(f"=== {code} {fq} ===")
            print(f"server: {ip}:{port}")
            if fq == "1d":
                df = ds.get_daily_bars(code, count=50)
            else:
                df = ds.get_minute_bars(code, freq=fq, count=200)
            print(f"rows: {len(df)} | datetime range: {_fmt_range(df)}")
            if not df.empty:
                nonempty = True
    print("\nSummary:")
    if nonempty:
        print("TDX connectivity OK for at least one symbol/frequency.")
    else:
        print("All TDX servers returned empty data. Very likely a network / firewall issue, not a code bug.")


if __name__ == "__main__":
    main()