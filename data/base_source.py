# data/base_source.py
"""
数据源抽象定义。

目标：
- 定义统一的行情访问接口（DataSource），回测/实盘/数据更新都只依赖这个抽象。
- 当前阶段由 PytdxDataSource 实现，未来可以替换为 xtquant、tushare 等。

注意：
- 所有方法都返回 pandas.DataFrame，字段命名尽量统一（open/high/low/close/volume/amount/datetime）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional

import pandas as pd


class DataSource(ABC):
    """
    行情数据源抽象接口。
    """

    @abstractmethod
    def get_daily_bars(self, ts_code: str, count: int = 240) -> pd.DataFrame:
        """
        获取最近 count 条日线数据，按时间升序返回。
        至少包含字段：
        - datetime (python datetime)
        - open, high, low, close, volume
        - amount（若数据源无，可填 0 或 None）
        """
        raise NotImplementedError

    @abstractmethod
    def get_minute_bars(
        self,
        ts_code: str,
        freq: str = "1m",
        count: int = 240,
    ) -> pd.DataFrame:
        """
        获取最近 count 条分钟线数据（默认 1 分钟）。
        freq 示例：'1m', '5m' 等。
        字段同日线，多一个 'freq' 可以放在 extra 中。
        """
        raise NotImplementedError

    @abstractmethod
    def get_ticks(
        self,
        ts_code: str,
        trade_date: Optional[date] = None,
        count: int = 2000,
    ) -> pd.DataFrame:
        """
        获取某日的部分 tick 数据。不同数据源实现可以不同：
        - trade_date 为空时，默认取最近交易日。
        - count 表示最多返回多少条（数据源限制为准）。
        返回字段：
        - datetime, price, volume, amount
        - 其他字段根据数据源决定（可附加）。
        """
        raise NotImplementedError


if __name__ == "__main__":
    # 简单自测：仅验证接口存在 & 能被子类继承。
    class DummySource(DataSource):
        def get_daily_bars(self, ts_code: str, count: int = 240) -> pd.DataFrame:
            return pd.DataFrame({"ts_code": [ts_code], "dummy": [1]})

        def get_minute_bars(
            self,
            ts_code: str,
            freq: str = "1m",
            count: int = 240,
        ) -> pd.DataFrame:
            return pd.DataFrame({"ts_code": [ts_code], "freq": [freq]})

        def get_ticks(
            self,
            ts_code: str,
            trade_date=None,
            count: int = 2000,
        ) -> pd.DataFrame:
            return pd.DataFrame({"ts_code": [ts_code], "tick": [1]})

    src = DummySource()
    print("Daily:", src.get_daily_bars("000001.SZ").head())
    print("Minute:", src.get_minute_bars("000001.SZ").head())
    print("Ticks:", src.get_ticks("000001.SZ").head())

