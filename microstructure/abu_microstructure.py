from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional

import math
import pandas as pd

from strategy.base import PriceLevel


DEFAULT_MICRO_CONFIG: Dict[str, Any] = {
    "window_seconds": 60,
    "min_total_volume": 1000,
    "min_bs_ratio": 1.5,
    "max_penetration_pct": 0.003,
    "level_tolerance_pct": 0.002,
}


@dataclass
class MicroSignal:
    ts_code: str
    trade_date: date
    dt: datetime
    side: str
    level_price: float
    level_type: str
    score: float
    reason: str
    meta: Dict[str, Any]


class AbuMicrostructureAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = DEFAULT_MICRO_CONFIG.copy()
        if config:
            self.config.update(config)

    def analyze_near_level(
        self,
        ts_code: str,
        trade_date: date,
        ticks: pd.DataFrame,
        level: PriceLevel,
        now_dt: datetime,
    ) -> Optional[MicroSignal]:
        if ticks.empty:
            return None

        cfg = self.config
        window_seconds = int(cfg["window_seconds"])
        tol_pct = float(cfg["level_tolerance_pct"])
        max_penetration = float(cfg["max_penetration_pct"])
        min_total_vol = int(cfg["min_total_volume"])
        min_bs_ratio = float(cfg["min_bs_ratio"])

        start_time = now_dt - timedelta(seconds=window_seconds)
        win = ticks[(ticks["datetime"] >= start_time) & (ticks["datetime"] <= now_dt)]
        if win.empty:
            return None

        last_price = float(win["price"].iloc[-1])
        level_price = float(level.level_price)
        dist_pct = abs(last_price - level_price) / max(level_price, 1e-6)
        if dist_pct > tol_pct:
            return None

        total_vol = float(win["volume"].sum())
        if total_vol < min_total_vol:
            return None

        if "side" in win.columns:
            buy_vol = float(win.loc[win["side"] == "B", "volume"].sum())
            sell_vol = float(win.loc[win["side"] == "S", "volume"].sum())
        else:
            return None

        eff_buy = buy_vol if buy_vol > 0 else 1.0
        eff_sell = sell_vol if sell_vol > 0 else 1.0

        min_price = float(win["price"].min())
        max_price = float(win["price"].max())

        if level.direction == "support":
            penetration_pct = (min_price - level_price) / max(level_price, 1e-6)
        elif level.direction == "resistance":
            penetration_pct = (max_price - level_price) / max(level_price, 1e-6)
        else:
            return None

        if penetration_pct < -max_penetration:
            return None

        side: Optional[str] = None
        score = 0.0
        reason = ""

        if level.direction == "support":
            bs_ratio = eff_buy / eff_sell
            if bs_ratio < min_bs_ratio:
                return None
            side = "long"
            score = self._score_support(level_price, last_price, penetration_pct, bs_ratio, total_vol)
            reason = f"support_buy_dominate bs_ratio={bs_ratio:.2f}, pen={penetration_pct:.4f}"
        elif level.direction == "resistance":
            sb_ratio = eff_sell / eff_buy
            if sb_ratio < min_bs_ratio:
                return None
            side = "short"
            score = self._score_resistance(level_price, last_price, penetration_pct, sb_ratio, total_vol)
            reason = f"resist_sell_dominate sb_ratio={sb_ratio:.2f}, pen={penetration_pct:.4f}"

        if side is None:
            return None

        meta = {
            "total_vol": total_vol,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "penetration_pct": penetration_pct,
            "dist_pct": dist_pct,
        }

        return MicroSignal(
            ts_code=ts_code,
            trade_date=trade_date,
            dt=now_dt,
            side=side,
            level_price=level_price,
            level_type=level.level_type,
            score=score,
            reason=reason,
            meta=meta,
        )

    def _score_support(self, level_price: float, last_price: float, penetration_pct: float, bs_ratio: float, total_vol: float) -> float:
        score = 60.0
        score += max(0.0, 10.0 * (1.0 + penetration_pct))
        score += min(15.0, (bs_ratio - 1.0) * 5.0)
        score += min(15.0, math.log10(max(total_vol, 1.0) + 9.0))
        return max(0.0, min(100.0, score))

    def _score_resistance(self, level_price: float, last_price: float, penetration_pct: float, sb_ratio: float, total_vol: float) -> float:
        score = 60.0
        score += max(0.0, 10.0 * (1.0 + penetration_pct))
        score += min(15.0, (sb_ratio - 1.0) * 5.0)
        score += min(15.0, math.log10(max(total_vol, 1.0) + 9.0))
        return max(0.0, min(100.0, score))

if __name__ == "__main__":
    from strategy.base import PriceLevel

    now = datetime.now()
    tds = [now - timedelta(seconds=60) + timedelta(seconds=i) for i in range(60)]
    prices = [10.0 - 0.02 * (59 - i) for i in range(60)]
    volumes = [100 + i * 5 for i in range(60)]
    sides = ["S"] * 40 + ["B"] * 20
    df = pd.DataFrame({"datetime": tds, "price": prices, "volume": volumes, "side": sides})
    level = PriceLevel(ts_code="000001.SZ", trade_date=now.date(), level_price=10.0, level_type="range_low", direction="support", strength=80, source_flags=["range"], meta={})
    analyzer = AbuMicrostructureAnalyzer()
    sig = analyzer.analyze_near_level(ts_code="000001.SZ", trade_date=now.date(), ticks=df, level=level, now_dt=now)
    print("Micro signal:", sig)

