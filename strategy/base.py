# strategy/base.py
"""
策略层核心抽象 & 通用数据结构。

本文件定义：
- 标准数据结构：Bar、Tick、PriceLevel、MicrostructureFeatures、TradeSignal
- 策略上下文协议：StrategyContext
- 抽象接口：PriceLevelProvider、MicrostructureProvider、BaseStrategy

设计目标：
1. 上层（回测/实盘/前端）只依赖这些抽象，不直接依赖具体策略实现。
2. 以后更换策略（从阿布价格理论换成别的逻辑）时，只需要实现新的
   PriceLevelProvider/MicrostructureProvider/BaseStrategy，即可复用整条流水线。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# =========================
# 通用数据结构定义
# =========================

@dataclass
class Bar:
    """
    标准 K 线结构（可用于日线/分钟线），回测和实盘统一用这个结构给策略喂数据。
    """
    ts_code: str
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tick:
    """
    标准 Tick 结构，承载逐笔成交或逐笔行情。
    """
    ts_code: str
    dt: datetime
    price: float
    volume: float
    amount: float = 0.0
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriceLevel:
    """
    关键价位结构：由 PriceLevelProvider 计算/提供。
    """
    ts_code: str
    trade_date: date
    level_price: float
    level_type: str              # 例如: swing_high, swing_low, gap, fibo, ...
    direction: str               # support / resistance / neutral
    strength: int                # 0–100
    source_flags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MicrostructureFeatures:
    """
    微观结构特征，用于在关键位附近判断“接受/拒绝”等。
    """
    imbalance: float
    buy_volume: float
    sell_volume: float
    large_buy_amount: float
    large_sell_amount: float
    price_change: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    """
    策略产生的交易信号，回测和实盘都使用这个结构承载。
    """
    ts_code: str
    dt: datetime
    side: str                     # BUY / SELL / FLAT / EXIT
    price: float
    reason: str
    suggested_qty: Optional[int] = None
    level: Optional[PriceLevel] = None
    micro: Optional[MicrostructureFeatures] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# =========================
# 策略上下文 Protocol
# =========================

@runtime_checkable
class StrategyContext(Protocol):
    """
    策略上下文接口：
    - 回测引擎和实盘引擎各自实现这个协议，策略层通过该上下文获取账户信息 / 持仓 / 日志功能。
    """

    def get_position(self, ts_code: str) -> float:
        """返回当前持仓数量（股数），没有则为 0。"""
        ...

    def get_cash(self) -> float:
        """返回当前可用现金。"""
        ...

    def log(self, msg: str) -> None:
        """策略日志输出接口。"""
        ...

    def get_config(self) -> Dict[str, Any]:
        """返回当前策略配置（如参数等）。"""
        ...


# =========================
# 抽象 Providers
# =========================

class PriceLevelProvider(ABC):
    """
    关键位提供者抽象：
    - 阿布价格理论的关键位模块会实现这个接口；
    - 将来换成缠论价格结构、其他价格理论时，也只需要实现该接口。
    """

    @abstractmethod
    def precompute(self, trade_date: date) -> None:
        """
        盘后批量任务：为指定 trade_date 计算全市场（或部分）关键位并落库。
        """
        raise NotImplementedError

    @abstractmethod
    def get_levels(self, ts_code: str, trade_date: date) -> List[PriceLevel]:
        """
        获取某只股票在指定交易日可用的关键位列表。
        一般从数据库 `price_levels_daily` 中读取之前离线算好的结果。
        """
        raise NotImplementedError


class MicrostructureProvider(ABC):
    """
    微观结构提供者抽象：
    - 基于 tick 数据，在指定时间窗口内计算订单流/盘口结构等特征。
    """

    @abstractmethod
    def analyze_window(
        self,
        ts_code: str,
        trade_date: date,
        start_time: time,
        end_time: time,
    ) -> Optional[MicrostructureFeatures]:
        """
        分析 [start_time, end_time] 窗口内的 tick 数据，返回微观结构特征。
        如无数据或窗口异常，返回 None。
        """
        raise NotImplementedError


# =========================
# BaseStrategy 抽象
# =========================

class BaseStrategy(ABC):
    """
    策略抽象基类：

    - on_bar: 接收 K 线（bar）事件，用于日线/分钟级决策。
    - on_tick: 接收 tick 事件，用于逐笔微观结构决策。
    - 两者都可以返回 0 个或多个 TradeSignal。
    """

    def __init__(
        self,
        name: str,
        price_level_provider: PriceLevelProvider,
        micro_provider: Optional[MicrostructureProvider] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.price_level_provider = price_level_provider
        self.micro_provider = micro_provider
        self.config = config or {}

    @abstractmethod
    def on_bar(
        self,
        bar: Bar,
        context: StrategyContext,
    ) -> List[TradeSignal]:
        """
        处理一根 K 线（可为日线/分钟线），返回产生的交易信号列表。
        """
        raise NotImplementedError

    @abstractmethod
    def on_tick(
        self,
        tick: Tick,
        context: StrategyContext,
    ) -> List[TradeSignal]:
        """
        处理一条 tick 数据，返回产生的交易信号列表。
        """
        raise NotImplementedError


# =========================
# Dummy 实现（仅用于自测 / 示例）
# =========================

class DummyPriceLevelProvider(PriceLevelProvider):
    """
    一个非常简单的 PriceLevelProvider 实现，仅用于自测和示例。
    不会用于真实策略。
    """

    def precompute(self, trade_date: date) -> None:
        # 自测场景下什么都不做
        print(f"[DummyPriceLevelProvider] precompute called for {trade_date}")

    def get_levels(self, ts_code: str, trade_date: date) -> List[PriceLevel]:
        # 返回一个固定的价位，方便测试
        return [
            PriceLevel(
                ts_code=ts_code,
                trade_date=trade_date,
                level_price=10.0,
                level_type="dummy",
                direction="support",
                strength=50,
                source_flags=["dummy"],
            )
        ]


class DummyMicrostructureProvider(MicrostructureProvider):
    """
    一个简单的微观结构 Provider，用于测试链路是否跑通。
    """

    def analyze_window(
        self,
        ts_code: str,
        trade_date: date,
        start_time: time,
        end_time: time,
    ) -> Optional[MicrostructureFeatures]:
        # 返回一些固定特征，方便测试
        return MicrostructureFeatures(
            imbalance=0.2,
            buy_volume=1000,
            sell_volume=800,
            large_buy_amount=500000.0,
            large_sell_amount=200000.0,
            price_change=0.05,
            extra={"source": "dummy"},
        )


class DummyContext:
    """
    一个极简的上下文实现，仅用于本文件的 __main__ 自测。
    """

    def __init__(self, cash: float = 1_000_000.0) -> None:
        self._cash = cash
        self._positions: Dict[str, float] = {}

    def get_position(self, ts_code: str) -> float:
        return self._positions.get(ts_code, 0.0)

    def get_cash(self) -> float:
        return self._cash

    def log(self, msg: str) -> None:
        print(f"[DummyContext] {msg}")

    def get_config(self) -> Dict[str, Any]:
        return {}


class DummyStrategy(BaseStrategy):
    """
    一个简单的策略示例：
    - on_bar: 如果收盘价低于 dummy 支撑位，就发出 BUY 信号。
    - on_tick: 不做任何事，只打印。
    """

    def on_bar(
        self,
        bar: Bar,
        context: StrategyContext,
    ) -> List[TradeSignal]:
        levels = self.price_level_provider.get_levels(bar.ts_code, bar.dt.date())
        signals: List[TradeSignal] = []

        context.log(f"Processing bar: {bar.ts_code} @ {bar.dt}, close={bar.close}")
        for lvl in levels:
            context.log(f"Found level: {lvl.level_type} {lvl.level_price}")
            if bar.close < lvl.level_price:
                signals.append(
                    TradeSignal(
                        ts_code=bar.ts_code,
                        dt=bar.dt,
                        side="BUY",
                        price=bar.close,
                        reason="dummy_close_below_level",
                        level=lvl,
                    )
                )

        return signals

    def on_tick(
        self,
        tick: Tick,
        context: StrategyContext,
    ) -> List[TradeSignal]:
        context.log(
            f"Processing tick: {tick.ts_code} @ {tick.dt}, price={tick.price}"
        )
        # Dummy 策略在 on_tick 不做事
        return []


# =========================
# main 自测入口
# =========================

if __name__ == "__main__":
    # 简单自测：构造一个 DummyStrategy，喂一根 bar，看是否能产生信号。
    from datetime import timedelta

    ts_code = "000001.SZ"
    trade_dt = datetime.now()

    dummy_pl = DummyPriceLevelProvider()
    dummy_micro = DummyMicrostructureProvider()
    dummy_ctx = DummyContext()

    strategy = DummyStrategy(
        name="dummy",
        price_level_provider=dummy_pl,
        micro_provider=dummy_micro,
        config={"example_param": 123},
    )

    # 构造一根 close < 10 的 bar，应该触发一个 BUY 信号
    test_bar = Bar(
        ts_code=ts_code,
        dt=trade_dt,
        open=9.5,
        high=9.8,
        low=9.3,
        close=9.4,
        volume=1_000_000,
    )

    signals = strategy.on_bar(test_bar, dummy_ctx)
    print("Generated signals from on_bar:")
    for sig in signals:
        print(sig)

    # 构造一个 tick，on_tick 这里只会打日志
    test_tick = Tick(
        ts_code=ts_code,
        dt=trade_dt + timedelta(seconds=1),
        price=9.45,
        volume=1000,
        amount=9_450.0,
    )
    _ = strategy.on_tick(test_tick, dummy_ctx)

