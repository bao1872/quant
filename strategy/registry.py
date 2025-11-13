# strategy/registry.py
"""
策略注册表：

目标：
- 通过一个统一的“注册中心”来管理所有策略实现（包括阿布价格策略、其他策略）。
- 回测引擎 / 实盘引擎 / 工具脚本，只需要通过策略名称来获取对应实现。
- 以后切换策略，只需在这里修改注册配置，而不需要改全局业务逻辑。

本模块只关注：
- name -> (StrategyClass, PriceLevelProviderClass, MicrostructureProviderClass) 的映射
- 简单的工厂方法：根据名称创建实例

注意：
- 这里可以先注册 DummyStrategy，后续再注册 AbuStrategy 等真实策略。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from strategy.base import PriceLevelProvider
from .abu_level_strategy import AbuKeyLevelStrategy


StrategyFactory = Callable[..., Any]


class StrategyRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, StrategyFactory] = {}
        self._register_builtin()

    def _register_builtin(self) -> None:
        self._factories["abu_key_level"] = self._create_abu_key_level

    def register(self, name: str, factory: StrategyFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> Any:
        if name not in self._factories:
            raise KeyError(f"Unknown strategy: {name}")
        return self._factories[name](**kwargs)

    def _create_abu_key_level(
        self,
        ts_code: str,
        price_level_provider: Optional[PriceLevelProvider] = None,
        **kwargs: Any,
    ) -> AbuKeyLevelStrategy:
        if price_level_provider is None:
            from factors import AbuPriceLevelProvider
            plp = AbuPriceLevelProvider()
        else:
            plp = price_level_provider
        from microstructure import AbuMicrostructureAnalyzer
        micro = AbuMicrostructureAnalyzer()
        return AbuKeyLevelStrategy(
            ts_code=ts_code,
            price_level_provider=plp,
            micro_analyzer=micro,
            **kwargs,
        )


if __name__ == "__main__":
    registry = StrategyRegistry()
    strat = registry.create("abu_key_level", ts_code="000001.SZ")
    print("Created strategy:", type(strat), strat.name, strat.ts_code)
