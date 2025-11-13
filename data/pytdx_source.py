# data/pytdx_source.py
"""
Pytdx 行情数据源实现。

说明：
- 封装 TdxHq_API，提供 DataSource 标准接口。
- 这里实现简单的“最近 N 条”模式，增量更新时用“取最近若干天 + 覆盖尾部”的方式。
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional

import pandas as pd
from pytdx.hq import TdxHq_API

from config import MARKET_SZ, MARKET_SH
from .base_source import DataSource


class PytdxDataSource(DataSource):
    def __init__(self, host: str = "auto", port: int = 7709):
        self.host = host
        self.port = port
        self.api = TdxHq_API()
        self._connected = False

    def __enter__(self) -> "PytdxDataSource":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # ---------- 基础连接 ----------

    def connect(self) -> None:
        if self._connected:
            return
        ip = self.host
        prt = self.port
        if ip == "auto":
            ip, prt = self._auto_select_host()
            self.host = ip
            self.port = prt
        ok = self.api.connect(ip, prt)
        if not ok:
            raise RuntimeError(f"Failed to connect pytdx {ip}:{prt}")
        self._connected = True

    def _auto_select_host(self) -> tuple[str, int]:
        candidates = [
            ("119.147.164.60", 7709),
            ("180.153.18.171", 7709),
            ("114.80.149.19", 7709),
            ("115.238.90.165", 7709),
            ("123.125.108.23", 7709),
            ("218.108.98.244", 7709),
        ]
        for ip, port in candidates:
            ok = self.api.connect(ip, port)
            if ok:
                self.api.disconnect()
                return ip, port
        return candidates[0]

    def disconnect(self) -> None:
        if self._connected:
            self.api.disconnect()
            self._connected = False

    # ---------- 工具函数 ----------

    @staticmethod
    def ts_code_to_tdx(ts_code: str):
        """
        ts_code: '000001.SZ' -> (market, code)
        """
        code, exch = ts_code.split(".")
        if exch.upper() == "SZ":
            return MARKET_SZ, code
        return MARKET_SH, code

    @staticmethod
    def _bars_to_df(data) -> pd.DataFrame:
        """
        将 pytdx 返回的 list of dict 转换为 DataFrame，并标准化字段名。
        """
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # pytdx 的字段通常有 'open', 'high', 'low', 'close', 'vol', 'amount', 'datetime'
        # 统一改名
        rename_map = {
            "vol": "volume",
        }
        df = df.rename(columns=rename_map)

        # 确保有 datetime 字段为真正的 datetime 类型
        if "datetime" in df.columns:
            # 日线: 'YYYY-MM-DD', 分钟: 'YYYY-MM-DD HH:MM'
            df["datetime"] = df["datetime"].apply(
                lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M")
                if " " in str(x)
                else datetime.strptime(x, "%Y-%m-%d")
            )
        return df

    # ---------- DataSource 实现 ----------

    def get_daily_bars(self, ts_code: str, count: int = 240) -> pd.DataFrame:
        """
        获取最近 count 条日线，按时间升序。
        category=9 是日线。
        """
        self.connect()
        market, code = self.ts_code_to_tdx(ts_code)

        # pytdx get_security_bars 返回从 start 开始的 count 条
        # 为了取最近 count 条，通常从 0 开始，取足够多，然后截 tail。
        raw = self.api.get_security_bars(
            category=9,  # 日线
            market=market,
            code=code,
            start=0,
            count=count,
        )
        df = self._bars_to_df(raw)
        df = df.sort_values("datetime").reset_index(drop=True)
        df["ts_code"] = ts_code
        return df

    def get_minute_bars(
        self,
        ts_code: str,
        freq: str = "1m",
        count: int = 240,
    ) -> pd.DataFrame:
        """
        获取最近 count 条分钟 K。
        pytdx category 定义（部分）：0=1m, 1=5m, 2=15m, 3=30m, 4=60m
        """
        self.connect()
        market, code = self.ts_code_to_tdx(ts_code)

        freq_map = {
            "1m": 0,
            "5m": 1,
            "15m": 2,
            "30m": 3,
            "60m": 4,
        }
        category = freq_map.get(freq, 0)

        raw = self.api.get_security_bars(
            category=category,
            market=market,
            code=code,
            start=0,
            count=count,
        )
        df = self._bars_to_df(raw)
        df = df.sort_values("datetime").reset_index(drop=True)
        df["ts_code"] = ts_code
        df["freq"] = freq
        return df

    def get_ticks(
        self,
        ts_code: str,
        trade_date: Optional[date] = None,
        count: int = 2000,
    ) -> pd.DataFrame:
        self.connect()
        market, code = self.ts_code_to_tdx(ts_code)

        raw = self.api.get_transaction_data(
            market=market,
            code=code,
            start=0,
            count=count,
        )
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        if "vol" in df.columns:
            df = df.rename(columns={"vol": "volume"})

        if "buyorsell" in df.columns:
            map_side = {0: "B", 1: "S"}
            df["side"] = df["buyorsell"].map(map_side).fillna("N")
        elif "bsflag" in df.columns:
            df["side"] = df["bsflag"].astype(str)
        else:
            df["side"] = "N"

        if "amount" not in df.columns and {"price", "volume"}.issubset(df.columns):
            df["amount"] = df["price"] * df["volume"]

        base_date = trade_date or date.today()
        if "time" in df.columns:
            def _parse_time(t: str) -> datetime:
                t = str(t)
                if len(t) == 5:
                    fmt = "%Y-%m-%d %H:%M"
                elif len(t) == 8:
                    fmt = "%Y-%m-%d %H:%M:%S"
                else:
                    t = t[:5]
                    fmt = "%Y-%m-%d %H:%M"
                return datetime.strptime(f"{base_date.strftime('%Y-%m-%d')} {t}", fmt)
            df["datetime"] = df["time"].apply(_parse_time)
        else:
            df["datetime"] = datetime.combine(base_date, datetime.min.time())

        df["ts_code"] = ts_code
        wanted = ["ts_code", "datetime", "price", "volume", "amount", "time", "side"]
        cols = [c for c in wanted if c in df.columns or c in ["ts_code", "datetime"]]
        return df[cols]

    def fetch_all_stock_list(self) -> pd.DataFrame:
        self.connect()
        all_rows = []
        for market, exch in [(MARKET_SZ, "SZ"), (MARKET_SH, "SH")]:
            step = 1000
            start = 0
            while True:
                data = self.api.get_security_list(market=market, start=start)
                if not data:
                    break
                df = pd.DataFrame(data)
                if df.empty:
                    break
                df["exchange"] = exch
                cols = [c for c in ["code", "name", "exchange"] if c in df.columns]
                all_rows.append(df[cols])
                if len(df) < step:
                    break
                start += step
        if not all_rows:
            return pd.DataFrame(columns=["code", "name", "exchange"])
        out = pd.concat(all_rows, ignore_index=True)
        out["code"] = out["code"].astype(str)
        out["name"] = out["name"].astype(str)
        out["exchange"] = out["exchange"].astype(str)
        return out

    def fetch_finance_info(self, ts_code: str) -> pd.DataFrame:
        self.connect()
        market, code = self.ts_code_to_tdx(ts_code)
        data = self.api.get_finance_info(market, code)
        if not data:
            return pd.DataFrame(columns=["ts_code", "total_shares", "float_shares"])
        df = pd.DataFrame([data])
        total = df.get("zongguben")
        floatable = df.get("liutongguben")
        total_val = float(total.iloc[0]) * 10000 if total is not None and len(total) > 0 else None
        float_val = float(floatable.iloc[0]) * 10000 if floatable is not None and len(floatable) > 0 else None
        return pd.DataFrame({"ts_code": [ts_code], "total_shares": [total_val], "float_shares": [float_val]})

    def get_ticks_full_day(self, ts_code: str, trade_date: date) -> pd.DataFrame:
        self.connect()
        market, code = self.ts_code_to_tdx(ts_code)
        all_rows: list = []
        start = 0
        step = 1000
        while True:
            raw = self.api.get_history_transaction_data(market, code, trade_date, start, step)
            if not raw:
                break
            df = pd.DataFrame(raw)
            if df.empty:
                break
            if "vol" in df.columns:
                df = df.rename(columns={"vol": "volume"})
            if "buyorsell" in df.columns:
                df["side"] = df["buyorsell"].map({0: "B", 1: "S"}).fillna("N")
            elif "bsflag" in df.columns:
                df["side"] = df["bsflag"].astype(str)
            else:
                df["side"] = "N"
            if "amount" not in df.columns and {"price", "volume"}.issubset(df.columns):
                df["amount"] = df["price"] * df["volume"]
            base_date = trade_date
            if "time" in df.columns:
                def _parse_time(t: str) -> datetime:
                    t = str(t)
                    if len(t) == 5:
                        fmt = "%Y-%m-%d %H:%M"
                    elif len(t) == 8:
                        fmt = "%Y-%m-%d %H:%M:%S"
                    else:
                        t = t[:5]
                        fmt = "%Y-%m-%d %H:%M"
                    return datetime.strptime(f"{base_date.strftime('%Y-%m-%d')} {t}", fmt)
                df["datetime"] = df["time"].apply(_parse_time)
            else:
                df["datetime"] = datetime.combine(base_date, datetime.min.time())
            df["ts_code"] = ts_code
            all_rows.append(df[["ts_code", "datetime", "price", "volume", "amount", "time", "side"]].copy())
            if len(df) < step:
                break
            start += step
        if not all_rows:
            return pd.DataFrame(columns=["ts_code", "datetime", "price", "volume", "amount", "time", "side"])
        out = pd.concat(all_rows, ignore_index=True)
        out = out.sort_values("datetime").reset_index(drop=True)
        return out


if __name__ == "__main__":
    # 简单自测：尝试连一次 pytdx 并获取一只股票的日线 & 分钟线
    src = PytdxDataSource()
    try:
        df_daily = src.get_daily_bars("000001.SZ", count=5)
        print("Daily bars sample:")
        print(df_daily.head())

        df_min = src.get_minute_bars("000001.SZ", freq="1m", count=5)
        print("Minute bars sample:")
        print(df_min.head())

        df_tick = src.get_ticks("000001.SZ", count=10)
        print("Ticks sample:")
        print(df_tick.head())
    finally:
        src.disconnect()
