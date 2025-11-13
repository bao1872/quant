from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class StrategyInstanceConfig:
    name: str
    ts_code: str
    initial_cash: float = 100_000.0
    strategy_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    name: str
    instances: List[StrategyInstanceConfig]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    meta: Dict[str, Any] = field(default_factory=dict)


if __name__ == "__main__":
    from datetime import date as _date
    cfg = PortfolioConfig(
        name="demo_portfolio",
        instances=[
            StrategyInstanceConfig(
                name="abu_key_level",
                ts_code="000001.SZ",
                initial_cash=100_000,
                strategy_params={"min_signal_score": 60.0},
            )
        ],
        start_date=_date(2024, 1, 1),
        end_date=_date(2024, 1, 31),
    )
    print("PortfolioConfig self-test:")
    print(cfg)

