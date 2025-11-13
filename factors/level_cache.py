# factors/level_cache.py
"""
回测专用：带缓存的 PriceLevelProvider。

目标：
- 使用与实盘完全相同的 PriceLevelProvider（默认 AbuPriceLevelProvider）
- 在回测开始前一次性预热需要的 (ts_code, trade_date) 的关键位
- 回测主循环里只查内存，避免频繁访问 DB，保证效率
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

from strategy.base import PriceLevel, PriceLevelProvider
from .abu_price_levels import AbuPriceLevelProvider


class CachedPriceLevelProvider(PriceLevelProvider):
    def __init__(self, base: Optional[PriceLevelProvider] = None) -> None:
        self.base = base or AbuPriceLevelProvider()
        self._cache: Dict[Tuple[str, date], List[PriceLevel]] = {}

    def precompute(self, trade_date: date) -> None:
        self.base.precompute(trade_date)

    def warmup_range(self, ts_codes: Iterable[str], dates: Iterable[date]) -> None:
        ts_list = list(ts_codes)
        date_list = list(dates)
        for ts in ts_list:
            for d in date_list:
                key = (ts, d)
                if key in self._cache:
                    continue
                levels = self.base.get_levels(ts, d)
                self._cache[key] = levels

    def get_levels(self, ts_code: str, trade_date: date) -> List[PriceLevel]:
        key = (ts_code, trade_date)
        if key in self._cache:
            return self._cache[key]
        levels = self.base.get_levels(ts_code, trade_date)
        self._cache[key] = levels
        return levels


if __name__ == "__main__":
    from datetime import timedelta

    print("[level_cache] self test...")

    base = AbuPriceLevelProvider()
    cached = CachedPriceLevelProvider(base)

    today = date.today()
    days = [today - timedelta(days=i) for i in range(5)]
    ts_codes = ["000001.SZ", "000002.SZ"]

    print("Warmup...")
    cached.warmup_range(ts_codes, days)

    print("Get levels after warmup:")
    for ts in ts_codes:
        for d in days:
            lv = cached.get_levels(ts, d)
            print(ts, d, "levels:", len(lv))

