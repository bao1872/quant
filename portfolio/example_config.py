from __future__ import annotations

from datetime import date
from portfolio.models import PortfolioConfig, StrategyInstanceConfig

EXAMPLE_PORTFOLIO_CONFIG = PortfolioConfig(
    name="example_abu_portfolio",
    instances=[
        StrategyInstanceConfig(
            name="abu_key_level",
            ts_code="000001.SZ",
            initial_cash=100_000,
            strategy_params={
                "min_signal_score": 60.0,
                "risk_per_trade_pct": 0.01,
                "max_position_pct": 0.3,
            },
        ),
        StrategyInstanceConfig(
            name="abu_key_level",
            ts_code="000002.SZ",
            initial_cash=150_000,
            strategy_params={
                "min_signal_score": 65.0,
                "risk_per_trade_pct": 0.008,
                "max_position_pct": 0.25,
            },
        ),
    ],
    start_date=date(2024, 1, 2),
    end_date=date(2024, 1, 10),
    meta={
        "description": "示例组合：两只股票跑阿布关键位策略",
    },
)


if __name__ == "__main__":
    print("EXAMPLE_PORTFOLIO_CONFIG:")
    print(EXAMPLE_PORTFOLIO_CONFIG)

