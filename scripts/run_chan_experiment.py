from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from chan.pipeline import run_chan_for_symbol_to_db
from db.connection import get_engine


def _count_table_rows(code: str, freq: str) -> dict[str, int]:
    eng = get_engine()
    with eng.connect() as conn:
        bi = pd.read_sql(text("select count(1) as c from stock_chan_bi where code=:c and freq=:f"), conn, params={"c": code, "f": freq})
        seg = pd.read_sql(text("select count(1) as c from stock_chan_segment where code=:c and freq=:f"), conn, params={"c": code, "f": freq})
        cen = pd.read_sql(text("select count(1) as c from stock_chan_center where code=:c and freq=:f"), conn, params={"c": code, "f": freq})
        sig = pd.read_sql(text("select count(1) as c from stock_chan_signal where code=:c and freq=:f"), conn, params={"c": code, "f": freq})
    return {
        "bi": int(bi["c"].iloc[0]) if not bi.empty else 0,
        "segment": int(seg["c"].iloc[0]) if not seg.empty else 0,
        "center": int(cen["c"].iloc[0]) if not cen.empty else 0,
        "signal": int(sig["c"].iloc[0]) if not sig.empty else 0,
    }


def main() -> None:
    codes = ["SH.688122", "SZ.000426", "SZ.300624"]
    freqs = ["day", "60m"]
    begin = "2023-01-01"
    end = "2023-12-31"

    for code in codes:
        run_chan_for_symbol_to_db(code, freqs, begin, end, data_src="baostock", autype="qfq")
        print(f"=== {code} ===")
        for f in freqs:
            cnts = _count_table_rows(code, f)
            print(f"{f}: bi={cnts['bi']}, segment={cnts['segment']}, center={cnts['center']}, signal={cnts['signal']}")


if __name__ == "__main__":
    main()