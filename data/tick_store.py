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
from typing import Optional

import pandas as pd

from config import TICK_BASE_DIR
from sqlalchemy import text
from db.connection import get_session, get_engine, DummySession
from db.models import TickFileIndex


class TickStore:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or TICK_BASE_DIR)
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
    ) -> None:
        """
        将指定日期的 tick DataFrame 写入 parquet 文件，并更新 TickFileIndex。
        """
        file_path = self._file_path(ts_code, trade_date)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(file_path)

        if df.empty:
            record_cnt = 0
            time_start = ""
            time_end = ""
        else:
            # 假设 df 有 datetime 字段
            dt_sorted = df.sort_values("datetime")
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
