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
from db.models import TickFileIndex


class TickStore:
    def __init__(self, base_dir: Optional[str] = None, settings: Optional[Settings] = None):
        base = base_dir or (settings.tick_base_dir if settings is not None else TICK_BASE_DIR)
        self.base_dir = Path(base)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.defer_index: bool = False
        self._pending_index: list[dict] = []

    def _file_path(self, ts_code: str, trade_date: date) -> Path:
        code, exch = ts_code.split(".")
        market = exch.upper()
        return self.base_dir / market / code / f"{trade_date.strftime('%Y%m%d')}_交易数据.parquet"

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

        if "side" not in df.columns:
            if "buyorsell" in df.columns:
                s = pd.to_numeric(df["buyorsell"], errors="coerce")
                df["side"] = s.map({0: "B", 1: "S", 2: "N"}).fillna("N")
            elif "bsflag" in df.columns:
                bs = df["bsflag"].astype(str).str.upper().str.strip()
                df["side"] = bs.map({"B": "B", "S": "S", "N": "N"}).fillna(
                    bs.apply(lambda x: "B" if x.startswith("B") else ("S" if x.startswith("S") else "N"))
                )
            else:
                df["side"] = "N"
        else:
            vals = df["side"]
            if pd.api.types.is_numeric_dtype(vals):
                s = pd.to_numeric(vals, errors="coerce")
                df["side"] = s.map({0: "B", 1: "S", 2: "N"}).fillna("N")
            else:
                df["side"] = vals.astype(str).str.upper().str.strip().replace({"": "N"})

        df.to_parquet(file_path, engine="pyarrow", compression="snappy")
        if (not file_path.exists()) and (len(df) > 0):
            import pyarrow as pa
            import pyarrow.parquet as pq
            tbl = pa.Table.from_pandas(df)
            pq.write_table(tbl, file_path, compression="snappy")

        if df.empty:
            record_cnt = 0
            time_start = ""
            time_end = ""
        else:
            dt_sorted = df if already_sorted else df.sort_values("datetime")
            record_cnt = len(dt_sorted)
            time_start = dt_sorted["datetime"].iloc[0].strftime("%H:%M:%S")
            time_end = dt_sorted["datetime"].iloc[-1].strftime("%H:%M:%S")

        row = {
            "ts_code": ts_code,
            "trade_date": trade_date,
            "market": ts_code.split(".")[1].upper(),
            "file_path": str(file_path),
            "record_cnt": record_cnt,
            "time_start": time_start,
            "time_end": time_end,
            "checksum": None,
        }
        self._pending_index.append(row)

    def flush_index_for_date(self, trade_date: date) -> int:
        if not self._pending_index:
            return 0
        rows = [r for r in self._pending_index if r["trade_date"] == trade_date]
        if not rows:
            return 0
        # 本地模式：不写入数据库索引，直接丢弃待提交列表
        self._pending_index = [r for r in self._pending_index if r["trade_date"] != trade_date]
        return len(rows)

    def normalize_tick_files_for_date(self, target_date: date) -> None:
        for p in self.base_dir.rglob("*.parquet"):
            code = p.parent.name
            market = p.parent.parent.name
            ts = f"{code}.{market}"
            df = pd.read_parquet(p)
            df["trade_date"] = target_date
            df["ts_code"] = ts
            if "side" not in df.columns:
                df["side"] = ""
            if "volume" not in df.columns:
                df["volume"] = 0
            if "amount" not in df.columns:
                df["amount"] = 0.0
            new_name = f"{target_date.strftime('%Y%m%d')}_交易数据.parquet"
            new_path = p.parent / new_name
            if p.name != new_name:
                p = p.rename(new_path)
            df.to_parquet(p, engine="pyarrow", compression="snappy")

    def normalize_file_names_and_update_index(self) -> int:
        n = 0
        renames: list[tuple[str, str, str, date]] = []
        for market_dir in self.base_dir.iterdir():
            if not market_dir.is_dir():
                continue
            for code_dir in market_dir.iterdir():
                if not code_dir.is_dir():
                    continue
                for f in code_dir.glob("*.parquet"):
                    name = f.name
                    if name.endswith("_交易数据.parquet"):
                        s = name.split("_")[0]
                        if len(s) == 8 and s.isdigit():
                            y = int(s[0:4]); m = int(s[4:6]); d = int(s[6:8])
                            dt = date(y, m, d)
                            renames.append((str(f), str(f), f"{code_dir.name}.{market_dir.name}", dt))
                        continue
                    base = name[:-8] if name.endswith(".parquet") else name
                    parts = base.split("-")
                    if len(parts) == 3 and len(parts[0]) == 4:
                        y = int(parts[0]); m = int(parts[1]); d = int(parts[2])
                        dt = date(y, m, d)
                        new_name = f"{dt.strftime('%Y%m%d')}_交易数据.parquet"
                        new_path = f.with_name(new_name)
                        f.rename(new_path)
                        n += 1
                        renames.append((str(f), str(new_path), f"{code_dir.name}.{market_dir.name}", dt))
        return n

    def load_ticks(
        self,
        ts_code: str,
        trade_date: date,
    ) -> pd.DataFrame:
        """
        从 parquet 文件读取某日某股的 tick 数据。
        """
        file_path = self._file_path(ts_code, trade_date)
        if file_path.exists():
            return pd.read_parquet(file_path)
        code, exch = ts_code.split(".")
        ymd = trade_date.strftime("%Y%m%d")
        ymd_dash = trade_date.strftime("%Y-%m-%d")
        candidates_dirs = [
            self.base_dir / exch.upper() / code,
            self.base_dir / exch.lower() / code,
            self.base_dir / code,
        ]
        for d in candidates_dirs:
            if d.exists():
                p1 = d / f"{ymd}_交易数据.parquet"
                if p1.exists():
                    return pd.read_parquet(p1)
                p2 = d / f"{ymd_dash}_交易数据.parquet"
                if p2.exists():
                    return pd.read_parquet(p2)
                files = list(d.glob("*.parquet"))
                if files:
                    sel = None
                    for f in files:
                        name = f.name
                        has_ymd = (ymd in name) or (ymd_dash in name)
                        if has_ymd:
                            sel = f
                            break
                    if sel is None:
                        sel = sorted(files)[-1]
                    return pd.read_parquet(sel)
        # 递归兜底：在任意层级下的 code 目录中查找目标日期文件
        files2 = list(self.base_dir.rglob(f"**/{code}/*.parquet"))
        if files2:
            sel2 = None
            for f in files2:
                name = f.name
                has_ymd = (ymd in name) or (ymd_dash in name)
                if has_ymd:
                    sel2 = f
                    break
            if sel2 is None:
                sel2 = sorted(files2)[-1]
            return pd.read_parquet(sel2)
        return pd.DataFrame()

    def list_tick_files(self, ts_code: str, start_date: date, end_date: date) -> List[TickFileIndex]:
        rows: List[TickFileIndex] = []
        code, exch = ts_code.split(".")
        base_dirs = [
            self.base_dir / exch.upper() / code,
            self.base_dir / exch.lower() / code,
            self.base_dir / code,
        ]
        import pyarrow.parquet as pq
        for d in pd.date_range(start_date, end_date, freq="D"):
            dt = d.date()
            ymd = dt.strftime("%Y%m%d")
            ymd_dash = dt.strftime("%Y-%m-%d")
            found = None
            for bd in base_dirs:
                p1 = bd / f"{ymd}_交易数据.parquet"
                p2 = bd / f"{ymd_dash}_交易数据.parquet"
                if p1.exists():
                    found = p1
                    break
                if p2.exists():
                    found = p2
                    break
            if found is None:
                continue
            try:
                pf = pq.ParquetFile(str(found))
                cnt = int(pf.metadata.num_rows or 0)
            except Exception:
                cnt = 0
            rows.append(
                TickFileIndex(
                    ts_code=ts_code,
                    trade_date=dt,
                    market=exch.upper(),
                    file_path=str(found),
                    record_cnt=cnt,
                    time_start="",
                    time_end="",
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
