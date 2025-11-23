from __future__ import annotations

from datetime import date
from typing import List

from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.repository import get_all_stock_basics
from db.connection import get_engine
from sqlalchemy import text
from chan.pipeline import run_chan_for_df
from chan.persist import save_chan_result_to_db
import pandas as pd


BEGIN = "2023-01-01"
END = None
FREQS: List[str] = ["1d", "60m", "30m", "15m", "5m"]


def main() -> None:
    eng = get_engine()
    basics = get_all_stock_basics()
    codes = [b.ts_code for b in basics]
    for ts in tqdm(codes, desc="chan_backfill", unit="stk"):
        for fq in FREQS:
            with eng.connect() as conn:
                params = {"ts": ts, "fq": fq}
                q = text(
                    "select datetime, open, high, low, close, volume from stock_kline where ts_code=:ts and freq=:fq order by datetime"
                )
                df = pd.read_sql(q, conn, params=params, parse_dates=["datetime"])  # type: ignore
            if df is None or df.empty:
                continue
            res_map = run_chan_for_df(ts, fq, df)
            r = res_map.get(fq)
            if r is not None:
                save_chan_result_to_db(r)


if __name__ == "__main__":
    main()