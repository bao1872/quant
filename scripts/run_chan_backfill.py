from __future__ import annotations

from datetime import date
from typing import List

from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.repository import get_all_stock_basics
from chan.pipeline import run_chan_for_symbol_to_db
from chan.utils import ts_to_chan_code


BEGIN = "2023-01-01"
END = None
LEVELS: List[str] = ["day", "60m", "30m", "15m", "5m"]


def main() -> None:
    basics = get_all_stock_basics()
    codes = [b.ts_code for b in basics]
    for ts in tqdm(codes, desc="chan_backfill", unit="stk"):
        code_chan = ts_to_chan_code(ts)
        run_chan_for_symbol_to_db(code_chan, LEVELS, begin_time=BEGIN, end_time=END, data_src="baostock", autype="qfq")


if __name__ == "__main__":
    main()