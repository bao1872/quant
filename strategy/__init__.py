# strategy/__init__.py
"""
strategy 包导出常用抽象与注册函数。

外部模块应该尽量只从这里导入：
- BaseStrategy / PriceLevelProvider / MicrostructureProvider 抽象
- TradeSignal / Bar / Tick / PriceLevel / MicrostructureFeatures 数据结构
- create_strategy_instances / register_strategy 等
"""

from .base import (
    BaseStrategy,
    PriceLevelProvider,
    MicrostructureProvider,
    StrategyContext,
    Bar,
    Tick,
    PriceLevel,
    MicrostructureFeatures,
    TradeSignal,
)
from .registry import StrategyRegistry

__all__ = [
    "BaseStrategy",
    "PriceLevelProvider",
    "MicrostructureProvider",
    "StrategyContext",
    "Bar",
    "Tick",
    "PriceLevel",
    "MicrostructureFeatures",
    "TradeSignal",
    "StrategyRegistry",
]


if __name__ == "__main__":
    print("strategy package self-test")
    reg = StrategyRegistry()
    print("Available strategies:", ["abu_key_level"]) 
