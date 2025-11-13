# backtest/__init__.py
"""
回测相关模块入口。
"""

from .engine import TickBacktester, BacktestContext, BacktestResult, TradeRecord

__all__ = [
    "TickBacktester",
    "BacktestContext",
    "BacktestResult",
    "TradeRecord",
]


if __name__ == "__main__":
    print("backtest package self-test:", __all__)

