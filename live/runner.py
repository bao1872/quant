from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional
import time

import pandas as pd

from data.tick_store import TickStore
from factors import AbuPriceLevelProvider
from microstructure import AbuMicrostructureAnalyzer
from strategy.abu_level_strategy import AbuKeyLevelStrategy
from .qmt_client import QmtClient
from .context import LiveContext


class LiveRunner:
    def __init__(self, ts_code: str, dry_run: bool = True, qmt_config: Optional[Dict[str, Any]] = None, strategy_config: Optional[Dict[str, Any]] = None) -> None:
        self.ts_code = ts_code
        self.dry_run = dry_run
        self.qmt_config = qmt_config or {}
        self.strategy_config = strategy_config or {}
        self.tick_store = TickStore()
        self.price_level_provider = AbuPriceLevelProvider()
        self.micro_analyzer = AbuMicrostructureAnalyzer()
        self.client = QmtClient(dry_run=self.dry_run, extra_config=self.qmt_config)
        self.client.connect()
        self.ctx = LiveContext(client=self.client, ts_code=self.ts_code)
        self.strategy = AbuKeyLevelStrategy(ts_code=self.ts_code, price_level_provider=self.price_level_provider, micro_analyzer=self.micro_analyzer, config=self.strategy_config)

    def run_replay(self, trade_date: date, speed: float = 0.0) -> None:
        self.client.connect()
        df = self.tick_store.load_ticks(self.ts_code, trade_date)
        if df is None or df.empty:
            print(f"[LiveRunner] no tick data for {self.ts_code} {trade_date}")
            self.client.disconnect()
            return
        df = df.sort_values("datetime").reset_index(drop=True)
        print(f"[LiveRunner] start replay {self.ts_code} {trade_date}, ticks={len(df)}, dry_run={self.dry_run}")
        for _, row in df.iterrows():
            tick = self._row_to_tick(row)
            self.strategy.on_tick(tick, self.ctx)
            if speed > 0:
                time.sleep(speed)
        print(f"[LiveRunner] replay finished. Trades:")
        print(self.ctx.trades_to_dataframe())
        self.client.disconnect()

    def _row_to_tick(self, row: pd.Series) -> Dict[str, Any]:
        dt = row["datetime"]
        if not isinstance(dt, datetime):
            dt = pd.to_datetime(dt).to_pydatetime()
        return {"ts_code": self.ts_code, "datetime": dt, "price": float(row["price"]), "volume": int(row.get("volume", 0)), "side": row.get("side", "N")}


if __name__ == "__main__":
    from datetime import timedelta
    ts = "000001.SZ"
    today = date.today()
    d = today - timedelta(days=1)
    runner = LiveRunner(ts_code=ts, dry_run=True, qmt_config={"initial_cash": 100_000}, strategy_config={"min_signal_score": 60.0})
    runner.run_replay(trade_date=d, speed=0.0)
