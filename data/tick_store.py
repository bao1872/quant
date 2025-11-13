# data/tick_store.py
"""
Tick 文件存储与索引管理。

规则：
- tick 数据只存文件（parquet），不进大 DB。
- 用 TickFileIndex 表记录 ts_code / trade_date -> file_path / 行数 / 起止时间 等信息。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import TICK_BASE_DIR, Settings
from sqlalchemy import text
from db.connection import get_session, get_engine, DummySession
from db.models import TickFileIndex


class TickStore:
    def __init__(self, base_dir: Optional[str] = None, settings: Optional[Settings] = None):
        base = base_dir or (settings.tick_base_dir if settings is not None else TICK_BASE_DIR)
        self.base_dir = Path(base)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, ts_code: str, trade_date: date) -> Path:
        code, exch = ts_code.split(".")
        market = exch.upper()
        return self.base_dir / market / code / f"{trade_date.isoformat()}.parquet"

    # ---------- 写入 & 索引 ----------

    def save_ticks(
        self,
        ts_code: str,
        trade_date: date,
        df: pd.DataFrame,
        already_sorted: bool = False,
    ) -> None:
        """
        将指定日期的 tick DataFrame 写入 parquet 文件，并更新 TickFileIndex。
        """
        file_path = self._file_path(ts_code, trade_date)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(file_path, engine="pyarrow", compression="snappy")

        if df.empty:
            record_cnt = 0
            time_start = ""
            time_end = ""
        else:
            dt_sorted = df if already_sorted else df.sort_values("datetime")
            record_cnt = len(dt_sorted)
            time_start = dt_sorted["datetime"].iloc[0].strftime("%H:%M:%S")
            time_end = dt_sorted["datetime"].iloc[-1].strftime("%H:%M:%S")

        eng = get_engine()
        if eng is None:
            with get_session() as session:
                if isinstance(session, DummySession):
                    return
        else:
            def _table_exists(table_name: str) -> bool:
                q = (
                    "select 1 from information_schema.tables where table_schema='public' and table_name='"
                    + table_name
                    + "'"
                )
                df_chk = pd.read_sql(q, eng)
                return len(df_chk) > 0
            with eng.begin() as conn:
                if _table_exists("tick_file_index"):
                    conn.execute(
                        text(
                            "delete from tick_file_index where ts_code=:ts and trade_date=:d"
                        ),
                        {"ts": ts_code, "d": trade_date},
                    )
                out = pd.DataFrame(
                    [
                        {
                            "ts_code": ts_code,
                            "trade_date": trade_date,
                            "market": ts_code.split(".")[1].upper(),
                            "file_path": str(file_path),
                            "record_cnt": record_cnt,
                            "time_start": time_start,
                            "time_end": time_end,
                            "checksum": None,
                        }
                    ]
                )
                out.to_sql("tick_file_index", eng, if_exists="append", index=False)

    def load_ticks(
        self,
        ts_code: str,
        trade_date: date,
    ) -> pd.DataFrame:
        """
        从 parquet 文件读取某日某股的 tick 数据。
        """
        file_path = self._file_path(ts_code, trade_date)
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    def list_tick_files(self, ts_code: str, start_date: date, end_date: date) -> List[TickFileIndex]:
        eng = get_engine()
        if eng is None:
            return []
        def _table_exists(table_name: str) -> bool:
            q = (
                "select 1 from information_schema.tables where table_schema='public' and table_name='"
                + table_name
                + "'"
            )
            df_chk = pd.read_sql(q, eng)
            return len(df_chk) > 0
        if not _table_exists("tick_file_index"):
            return []
        q = text(
            "select ts_code, trade_date, market, file_path, record_cnt, time_start, time_end, checksum "
            "from tick_file_index where ts_code=:ts and trade_date>=:d1 and trade_date<=:d2 order by trade_date"
        )
        df = pd.read_sql(q, eng, params={"ts": ts_code, "d1": start_date, "d2": end_date})
        rows: List[TickFileIndex] = []
        for _, r in df.iterrows():
            rows.append(
                TickFileIndex(
                    ts_code=str(r["ts_code"]),
                    trade_date=r["trade_date"],
                    market=str(r["market"]),
                    file_path=str(r["file_path"]),
                    record_cnt=int(r["record_cnt"] or 0),
                    time_start=str(r.get("time_start", "")),
                    time_end=str(r.get("time_end", "")),
                    checksum=None,
                )
            )
        return rows


if __name__ == "__main__":
    # 自测：构造一个虚拟 tick df，保存再读出
    from datetime import datetime

    print("[TickStore] self test...")

    store = TickStore()
    ts = "000001.SZ"
    td = date.today()

    df = pd.DataFrame(
        {
            "datetime": [
                datetime(td.year, td.month, td.day, 9, 30),
                datetime(td.year, td.month, td.day, 9, 31),
            ],
            "price": [10.0, 10.1],
            "volume": [100, 200],
            "amount": [1000.0, 2020.0],
        }
    )

    store.save_ticks(ts, td, df)
    df2 = store.load_ticks(ts, td)
    print(df2)
