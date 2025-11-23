from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from .base_source import DataSource
from .pytdx_source import PytdxDataSource
from . import repository


class HybridDataSource(DataSource):
    def __init__(self, inner: Optional[PytdxDataSource] = None) -> None:
        self._inner = inner or PytdxDataSource()

    def get_daily_bars(self, ts_code: str, count: int = 240) -> pd.DataFrame:
        freq = "1d"
        df_db = repository.get_recent_kline(ts_code, freq, count)
        if len(df_db) >= count:
            return df_db
        df_remote = self._inner.get_daily_bars(ts_code, count=count)
        if not df_remote.empty:
            repository.upsert_stock_kline(ts_code, freq, df_remote)
            df_db = repository.get_recent_kline(ts_code, freq, count)
            if not df_db.empty:
                return df_db
        return df_remote

    def get_minute_bars(self, ts_code: str, freq: str = "1m", count: int = 240) -> pd.DataFrame:
        df_db = repository.get_recent_kline(ts_code, freq, count)
        if len(df_db) >= count:
            return df_db
        df_remote = self._inner.get_minute_bars(ts_code, freq=freq, count=count)
        if not df_remote.empty:
            repository.upsert_stock_kline(ts_code, freq, df_remote)
            df_db = repository.get_recent_kline(ts_code, freq, count)
            if not df_db.empty:
                return df_db
        return df_remote

    def get_ticks(self, ts_code: str, trade_date: Optional[date] = None, count: int = 2000) -> pd.DataFrame:
        return self._inner.get_ticks(ts_code, trade_date=trade_date, count=count)