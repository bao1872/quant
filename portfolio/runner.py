from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from portfolio.models import PortfolioConfig, StrategyInstanceConfig
from backtest.engine import TickBacktester, BacktestResult
from live.live_engine import LiveEngine
from live.broker_qmt import QmtBroker
from live.broker_base import BrokerBase
from factors import AbuPriceLevelProvider
from microstructure import AbuMicrostructureAnalyzer
from strategy.registry import StrategyRegistry


@dataclass
class PortfolioBacktestResult:
    portfolio_name: str
    start_date: date
    end_date: date
    instance_results: Dict[str, BacktestResult]

    def to_equity_dataframe(self) -> pd.DataFrame:
        rows = []
        for key, r in self.instance_results.items():
            rows.append(
                {
                    "key": key,
                    "ts_code": r.ts_code,
                    "start_date": r.start_date,
                    "end_date": r.end_date,
                    "initial_cash": r.initial_cash,
                    "final_equity": r.final_equity,
                    "total_pnl": r.total_pnl,
                    "trade_count": r.trade_count,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "key",
                    "ts_code",
                    "start_date",
                    "end_date",
                    "initial_cash",
                    "final_equity",
                    "total_pnl",
                    "trade_count",
                ]
            )
        return pd.DataFrame(rows)


class PortfolioBacktestRunner:
    def __init__(self, cfg: PortfolioConfig) -> None:
        self.cfg = cfg

    def _build_date_range(self, start: date, end: date) -> List[date]:
        cur = start
        out: List[date] = []
        while cur <= end:
            out.append(cur)
            cur = cur + timedelta(days=1)
        return out

    def run(self) -> PortfolioBacktestResult:
        if self.cfg.start_date is None or self.cfg.end_date is None:
            raise ValueError("PortfolioConfig.start_date / end_date 不能为空")
        trade_dates = self._build_date_range(self.cfg.start_date, self.cfg.end_date)
        results: Dict[str, BacktestResult] = {}
        for inst in self.cfg.instances:
            print(
                f"[PortfolioBacktestRunner] run backtest for {inst.ts_code} with initial_cash={inst.initial_cash}"
            )
            bt = TickBacktester(
                ts_code=inst.ts_code,
                initial_cash=inst.initial_cash,
            )
            result = bt.run(
                trade_dates=trade_dates,
                strategy_config=inst.strategy_params,
            )
            key = f"{inst.name}:{inst.ts_code}"
            results[key] = result
        return PortfolioBacktestResult(
            portfolio_name=self.cfg.name,
            start_date=self.cfg.start_date,
            end_date=self.cfg.end_date,
            instance_results=results,
        )


class PortfolioLiveReplayRunner:
    def __init__(
        self,
        cfg: PortfolioConfig,
        broker_factory: Optional[callable] = None,
        strategy_registry: Optional[StrategyRegistry] = None,
    ) -> None:
        self.cfg = cfg
        self.registry = strategy_registry or StrategyRegistry()
        self.broker_factory = broker_factory or (lambda: QmtBroker(dry_run=True, config={"initial_cash": 1_000_000}))
        self._engines: Dict[str, LiveEngine] = {}

    def _build_date_range(self, start: date, end: date) -> List[date]:
        cur = start
        out: List[date] = []
        while cur <= end:
            out.append(cur)
            cur = cur + timedelta(days=1)
        return out

    def _ensure_engines(self) -> None:
        if self._engines:
            return
        for inst in self.cfg.instances:
            broker: BrokerBase = self.broker_factory()
            plp = AbuPriceLevelProvider()
            strategy = self.registry.create(
                "abu_key_level",
                ts_code=inst.ts_code,
                price_level_provider=plp,
                config=inst.strategy_params,
            )
            engine = LiveEngine(
                strategy=strategy,
                broker=broker,
            )
            key = f"{inst.name}:{inst.ts_code}"
            self._engines[key] = engine
            print(f"[PortfolioLiveReplayRunner] engine created for {key}")

    def run_replay(self) -> Dict[str, pd.DataFrame]:
        if self.cfg.start_date is None or self.cfg.end_date is None:
            raise ValueError("PortfolioConfig.start_date / end_date 不能为空")
        self._ensure_engines()
        trade_dates = self._build_date_range(self.cfg.start_date, self.cfg.end_date)
        all_orders: Dict[str, pd.DataFrame] = {}
        for key, engine in self._engines.items():
            ts_code = key.split(":", 1)[1]
            print(f"[PortfolioLiveReplayRunner] run replay for {key}")
            df_orders = engine.run_replay(ts_code=ts_code, trade_dates=trade_dates)
            all_orders[key] = df_orders
        return all_orders


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
            ),
            StrategyInstanceConfig(
                name="abu_key_level",
                ts_code="000002.SZ",
                initial_cash=200_000,
                strategy_params={"min_signal_score": 65.0},
            ),
        ],
        start_date=_date(2024, 1, 2),
        end_date=_date(2024, 1, 5),
    )
    print("=== PortfolioBacktestRunner self-test ===")
    bt_runner = PortfolioBacktestRunner(cfg)
    bt_result = bt_runner.run()
    print(bt_result.to_equity_dataframe())
    print("=== PortfolioLiveReplayRunner self-test ===")
    live_runner = PortfolioLiveReplayRunner(cfg)
    orders_dict = live_runner.run_replay()
    for k, df in orders_dict.items():
        print(f"orders for {k}:")
        print(df.head())
