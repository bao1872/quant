# factors/abu_price_levels.py
"""
阶段 2：基于阿布价格理论/价格行为的关键位模块。

核心要点：
- 只从价格结构里提取关键位（区间、趋势 pivot、缺口、突破、测量目标位）
- 不把 MA / 20日高低 等指标直接当关键位，只能作为共振信息写进 meta（目前先不加）
- 支持：
    - precompute(trade_date): 批量计算并写入 price_levels_daily
    - get_levels(ts_code, trade_date): 从 DB 读出关键位
- 回测与实盘调用的是同一套逻辑，保证一致性
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

import json
import math

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine
from data.repository import get_all_stock_basics
from strategy.base import PriceLevel, PriceLevelProvider


# ---------------- 工具：获取 SQLAlchemy engine ----------------

def _get_engine():
    return get_engine()

def _table_exists(eng, table_name: str) -> bool:
    q = (
        "select 1 from information_schema.tables where table_schema='public' and table_name='"
        + table_name
        + "'"
    )
    df = pd.read_sql(q, eng)
    return len(df) > 0


# ---------------- 配置 ----------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "lookback_days": 250,
    "swing_window": 2,
    "min_range_pivots": 4,
    "min_range_bars": 10,
    "max_range_width_pct": 0.08,
    "merge_pct": 0.002,
    "merge_min_tick": 0.01,
    "recent_window": 40,
    "min_major_swing_pct": 0.04,
    "measured_move_lookback": 120,
    "max_levels_per_stock": 40,
    "stock_pool_limit": None,
    "debug_print": False,
}


# ---------------- 内部数据结构 ----------------

@dataclass
class SwingPoint:
    idx: int
    trade_date: date
    price: float
    kind: str


@dataclass
class RangeSegment:
    start_idx: int
    end_idx: int
    high: float
    low: float
    tests_high: int
    tests_low: int


@dataclass
class BreakoutInfo:
    idx: int
    trade_date: date
    price: float
    direction: str
    range_high: float
    range_low: float


@dataclass
class MeasuredMoveTarget:
    trade_date: date
    price: float
    direction: str
    from_idx: int
    to_idx: int


@dataclass
class _LevelCandidate:
    ts_code: str
    trade_date: date
    price: float
    level_type: str
    direction: str
    strength: float
    source_flags: List[str]
    src_index: int
    meta: Dict[str, Any]


class AbuPriceStructureExtractor:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

    def extract_levels_for_history(
        self,
        ts_code: str,
        df: pd.DataFrame,
        target_date: date,
    ) -> List[PriceLevel]:
        if df.empty:
            return []

        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df = df.sort_values("trade_date").reset_index(drop=True)
        df = df[df["trade_date"] <= target_date]
        if df.empty:
            return []

        target_date = df["trade_date"].iloc[-1]

        swings = self._find_swings(df)
        ranges = self._detect_ranges(df, swings)
        major_pivots = self._detect_major_pivots(df, swings, ranges)
        gaps = self._detect_gaps(df)
        breakouts = self._detect_breakouts(df, ranges)
        measured = self._detect_measured_moves(major_pivots)

        cands: List[_LevelCandidate] = []
        cands.extend(self._build_range_levels(ts_code, target_date, df, ranges))
        cands.extend(self._build_pivot_levels(ts_code, target_date, df, major_pivots))
        cands.extend(self._build_gap_levels(ts_code, target_date, gaps))
        cands.extend(self._build_breakout_levels(ts_code, target_date, df, breakouts))
        cands.extend(self._build_measured_move_levels(ts_code, target_date, measured))

        if not cands:
            return []

        merged = self._merge_candidates(cands)
        levels = [self._candidate_to_level(c) for c in merged]
        max_n = self.config.get("max_levels_per_stock") or 40
        levels = sorted(levels, key=lambda x: x.strength, reverse=True)[:max_n]
        return levels

    def _find_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        win = int(self.config.get("swing_window", 2))
        n = len(df)
        if n < 2 * win + 1:
            return []

        highs = df["high"].values
        lows = df["low"].values
        tds = df["trade_date"].values

        swings: List[SwingPoint] = []
        for i in range(win, n - win):
            is_high = all(highs[i] >= highs[i - k] for k in range(1, win + 1)) and \
                      all(highs[i] > highs[i + k] for k in range(1, win + 1))
            is_low = all(lows[i] <= lows[i - k] for k in range(1, win + 1)) and \
                     all(lows[i] < lows[i + k] for k in range(1, win + 1))
            if is_high:
                swings.append(SwingPoint(idx=i, trade_date=tds[i], price=float(highs[i]), kind="H"))
            if is_low:
                swings.append(SwingPoint(idx=i, trade_date=tds[i], price=float(lows[i]), kind="L"))

        swings.sort(key=lambda s: s.idx)
        return swings

    def _detect_ranges(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[RangeSegment]:
        min_pivots = int(self.config.get("min_range_pivots", 4))
        min_bars = int(self.config.get("min_range_bars", 10))
        max_width_pct = float(self.config.get("max_range_width_pct", 0.08))

        ranges: List[RangeSegment] = []
        if len(swings) < min_pivots:
            return ranges

        current: List[SwingPoint] = [swings[0]]

        def make_segment(pivots: List[SwingPoint]) -> Optional[RangeSegment]:
            if len(pivots) < min_pivots:
                return None
            start_idx = pivots[0].idx
            end_idx = pivots[-1].idx
            if end_idx - start_idx + 1 < min_bars:
                return None
            prices = [p.price for p in pivots]
            high = max(prices)
            low = min(prices)
            mid = (high + low) / 2.0
            if mid <= 0:
                return None
            width_pct = (high - low) / mid
            if width_pct > max_width_pct:
                return None

            band = (high - low)
            high_band = high - band * 0.2
            low_band = low + band * 0.2
            tests_high = sum(1 for p in pivots if p.price >= high_band)
            tests_low = sum(1 for p in pivots if p.price <= low_band)

            return RangeSegment(
                start_idx=start_idx,
                end_idx=end_idx,
                high=float(high),
                low=float(low),
                tests_high=tests_high,
                tests_low=tests_low,
            )

        for sp in swings[1:]:
            tmp = current + [sp]
            prices = [p.price for p in tmp]
            high = max(prices)
            low = min(prices)
            mid = (high + low) / 2.0
            width_pct = (high - low) / mid if mid > 0 else 0.0

            if width_pct <= max_width_pct * 1.2:
                current.append(sp)
            else:
                seg = make_segment(current)
                if seg:
                    ranges.append(seg)
                current = [sp]

        seg = make_segment(current)
        if seg:
            ranges.append(seg)

        if not ranges:
            return ranges

        ranges.sort(key=lambda r: r.start_idx)
        merged: List[RangeSegment] = [ranges[0]]
        for seg in ranges[1:]:
            last = merged[-1]
            if seg.start_idx <= last.end_idx:
                merged[-1] = RangeSegment(
                    start_idx=last.start_idx,
                    end_idx=max(last.end_idx, seg.end_idx),
                    high=max(last.high, seg.high),
                    low=min(last.low, seg.low),
                    tests_high=last.tests_high + seg.tests_high,
                    tests_low=last.tests_low + seg.tests_low,
                )
            else:
                merged.append(seg)

        return merged

    def _detect_major_pivots(self, df: pd.DataFrame, swings: List[SwingPoint], ranges: List[RangeSegment]) -> List[SwingPoint]:
        if not swings:
            return []

        min_pct = float(self.config.get("min_major_swing_pct", 0.04))

        def in_range(idx: int) -> bool:
            for r in ranges:
                if r.start_idx <= idx <= r.end_idx:
                    return True
            return False

        major: List[SwingPoint] = []
        prev = swings[0]
        for sp in swings[1:]:
            if sp.kind == prev.kind:
                prev = sp
                continue

            if prev.kind == "L" and sp.kind == "H":
                pct = (sp.price - prev.price) / prev.price if prev.price > 0 else 0.0
            elif prev.kind == "H" and sp.kind == "L":
                pct = (prev.price - sp.price) / prev.price if prev.price > 0 else 0.0
            else:
                pct = 0.0

            if pct >= min_pct:
                if not in_range(prev.idx):
                    major.append(prev)
                if not in_range(sp.idx):
                    major.append(sp)

            prev = sp

        uniq: Dict[int, SwingPoint] = {}
        for sp in major:
            uniq[sp.idx] = sp
        res = list(uniq.values())
        res.sort(key=lambda s: s.idx)
        return res

    def _detect_gaps(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        n = len(df)
        if n < 2:
            return []
        gaps: List[Dict[str, Any]] = []
        for i in range(1, n):
            prev = df.iloc[i - 1]
            cur = df.iloc[i]
            prev_high, prev_low = prev["high"], prev["low"]
            cur_high, cur_low = cur["high"], cur["low"]

            if cur_low > prev_high * 1.001:
                gaps.append({"idx": i, "trade_date": cur["trade_date"], "type": "up", "low": float(prev_high), "high": float(cur_low)})

            if cur_high < prev_low * 0.999:
                gaps.append({"idx": i, "trade_date": cur["trade_date"], "type": "down", "low": float(cur_high), "high": float(prev_low)})
        return gaps

    def _detect_breakouts(self, df: pd.DataFrame, ranges: List[RangeSegment]) -> List[BreakoutInfo]:
        n = len(df)
        if not ranges or n == 0:
            return []
        breakouts: List[BreakoutInfo] = []
        for seg in ranges:
            for i in range(seg.end_idx + 1, n):
                row = df.iloc[i]
                td = row["trade_date"]
                high = row["high"]
                low = row["low"]
                close = row["close"]

                if close > seg.high and high > seg.high:
                    breakouts.append(BreakoutInfo(idx=i, trade_date=td, price=float(high), direction="up", range_high=seg.high, range_low=seg.low))
                    break

                if close < seg.low and low < seg.low:
                    breakouts.append(BreakoutInfo(idx=i, trade_date=td, price=float(low), direction="down", range_high=seg.high, range_low=seg.low))
                    break
        return breakouts

    def _detect_measured_moves(self, major_pivots: List[SwingPoint]) -> List[MeasuredMoveTarget]:
        res: List[MeasuredMoveTarget] = []
        if len(major_pivots) < 3:
            return res

        for a, b, c in zip(major_pivots[:-2], major_pivots[1:-1], major_pivots[2:]):
            if a.kind == "L" and b.kind == "H" and c.kind == "L" and c.price > a.price:
                height = b.price - a.price
                target_price = c.price + height
                res.append(MeasuredMoveTarget(trade_date=c.trade_date, price=float(target_price), direction="up", from_idx=a.idx, to_idx=c.idx))

            if a.kind == "H" and b.kind == "L" and c.kind == "H" and c.price < a.price:
                height = a.price - b.price
                target_price = c.price - height
                res.append(MeasuredMoveTarget(trade_date=c.trade_date, price=float(target_price), direction="down", from_idx=a.idx, to_idx=c.idx))

        return res

    def _apply_recency_bonus(self, base: float, bars_ago: int) -> float:
        recent_win = int(self.config.get("recent_window", 40))
        if bars_ago <= 0:
            bonus = 10.0
        elif bars_ago >= recent_win:
            bonus = 0.0
        else:
            bonus = 10.0 * (1.0 - bars_ago / recent_win)
        return max(0.0, min(100.0, base + bonus))

    def _build_range_levels(self, ts_code: str, target_date: date, df: pd.DataFrame, ranges: List[RangeSegment]) -> List[_LevelCandidate]:
        if not ranges:
            return []
        n = len(df)
        cands: List[_LevelCandidate] = []
        for seg in ranges:
            bars_ago = n - 1 - seg.end_idx
            base_high = 70.0 + min(seg.tests_high, 5) * 2.0
            base_low = 70.0 + min(seg.tests_low, 5) * 2.0

            cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(seg.high), level_type="range_high", direction="resistance", strength=self._apply_recency_bonus(base_high, bars_ago), source_flags=["range"], src_index=seg.end_idx, meta={"tests_high": seg.tests_high, "range_low": seg.low}))
            cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(seg.low), level_type="range_low", direction="support", strength=self._apply_recency_bonus(base_low, bars_ago), source_flags=["range"], src_index=seg.end_idx, meta={"tests_low": seg.tests_low, "range_high": seg.high}))
        return cands

    def _build_pivot_levels(self, ts_code: str, target_date: date, df: pd.DataFrame, pivots: List[SwingPoint]) -> List[_LevelCandidate]:
        if not pivots:
            return []
        n = len(df)
        cands: List[_LevelCandidate] = []
        for sp in pivots:
            bars_ago = n - 1 - sp.idx
            if sp.kind == "H":
                direction = "resistance"
                level_type = "trend_pivot_high"
            else:
                direction = "support"
                level_type = "trend_pivot_low"
            base = 60.0
            cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(sp.price), level_type=level_type, direction=direction, strength=self._apply_recency_bonus(base, bars_ago), source_flags=["trend_pivot"], src_index=sp.idx, meta={}))
        return cands

    def _build_gap_levels(self, ts_code: str, target_date: date, gaps: List[Dict[str, Any]]) -> List[_LevelCandidate]:
        if not gaps:
            return []
        cands: List[_LevelCandidate] = []
        for g in gaps:
            idx = g["idx"]
            if g["type"] == "up":
                base_low = 65.0
                base_high = 55.0
                cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(g["low"]), level_type="gap_up_low", direction="support", strength=base_low, source_flags=["gap"], src_index=idx, meta={"gap_type": "up"}))
                cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(g["high"]), level_type="gap_up_high", direction="neutral", strength=base_high, source_flags=["gap"], src_index=idx, meta={"gap_type": "up"}))
            else:
                base_high = 65.0
                base_low = 55.0
                cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(g["high"]), level_type="gap_down_high", direction="resistance", strength=base_high, source_flags=["gap"], src_index=idx, meta={"gap_type": "down"}))
                cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(g["low"]), level_type="gap_down_low", direction="neutral", strength=base_low, source_flags=["gap"], src_index=idx, meta={"gap_type": "down"}))
        return cands

    def _build_breakout_levels(self, ts_code: str, target_date: date, df: pd.DataFrame, breakouts: List[BreakoutInfo]) -> List[_LevelCandidate]:
        if not breakouts:
            return []
        n = len(df)
        cands: List[_LevelCandidate] = []
        for b in breakouts:
            bars_ago = n - 1 - b.idx
            if b.direction == "up":
                cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(b.price), level_type="breakout_high", direction="resistance", strength=self._apply_recency_bonus(70.0, bars_ago), source_flags=["breakout"], src_index=b.idx, meta={"range_high": b.range_high, "range_low": b.range_low}))
            else:
                cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(b.price), level_type="breakout_low", direction="support", strength=self._apply_recency_bonus(70.0, bars_ago), source_flags=["breakout"], src_index=b.idx, meta={"range_high": b.range_high, "range_low": b.range_low}))
        return cands

    def _build_measured_move_levels(self, ts_code: str, target_date: date, measured: List[MeasuredMoveTarget]) -> List[_LevelCandidate]:
        if not measured:
            return []
        cands: List[_LevelCandidate] = []
        for m in measured:
            direction = "resistance" if m.direction == "up" else "support"
            cands.append(_LevelCandidate(ts_code=ts_code, trade_date=target_date, price=float(m.price), level_type="measured_move_target", direction=direction, strength=55.0, source_flags=["measured"], src_index=m.to_idx, meta={"direction": m.direction}))
        return cands

    def _merge_candidates(self, cands: Sequence[_LevelCandidate]) -> List[_LevelCandidate]:
        if not cands:
            return []

        merge_pct = float(self.config.get("merge_pct", 0.002))
        min_tick = float(self.config.get("merge_min_tick", 0.01))

        cands_sorted = sorted(cands, key=lambda c: c.price)
        clusters: List[List[_LevelCandidate]] = []
        current: List[_LevelCandidate] = [cands_sorted[0]]

        def can_merge(center: float, price: float) -> bool:
            radius = max(min_tick, center * merge_pct)
            return abs(price - center) <= radius

        for cand in cands_sorted[1:]:
            center = sum(c.price for c in current) / len(current)
            if can_merge(center, cand.price):
                current.append(cand)
            else:
                clusters.append(current)
                current = [cand]
        clusters.append(current)

        merged: List[_LevelCandidate] = []
        for cluster in clusters:
            merged.append(self._merge_cluster(cluster))
        return merged

    def _merge_cluster(self, cluster: Sequence[_LevelCandidate]) -> _LevelCandidate:
        if not cluster:
            raise ValueError("empty cluster")

        ts_code = cluster[0].ts_code
        trade_date = cluster[0].trade_date

        total_strength = sum(c.strength for c in cluster) or 1.0
        price = sum(c.price * c.strength for c in cluster) / total_strength

        support_votes = sum(1 for c in cluster if c.direction == "support")
        resist_votes = sum(1 for c in cluster if c.direction == "resistance")
        if support_votes > resist_votes:
            direction = "support"
        elif resist_votes > support_votes:
            direction = "resistance"
        else:
            direction = "neutral"

        strength_avg = sum(c.strength for c in cluster) / len(cluster)
        strength = min(100.0, strength_avg + 5.0 * (len(cluster) - 1))

        types = sorted({c.level_type for c in cluster})
        level_type = "+".join(types)
        flags: List[str] = []
        for c in cluster:
            for f in c.source_flags:
                if f not in flags:
                    flags.append(f)

        src_index = min(c.src_index for c in cluster)

        meta: Dict[str, Any] = {}
        for c in cluster:
            for k, v in c.meta.items():
                if k not in meta:
                    meta[k] = v

        return _LevelCandidate(ts_code=ts_code, trade_date=trade_date, price=float(price), level_type=level_type, direction=direction, strength=strength, source_flags=flags, src_index=src_index, meta=meta)

    def _candidate_to_level(self, c: _LevelCandidate) -> PriceLevel:
        return PriceLevel(ts_code=c.ts_code, trade_date=c.trade_date, level_price=float(c.price), level_type=c.level_type, direction=c.direction, strength=int(round(c.strength)), source_flags=c.source_flags, meta=c.meta)


class AbuPriceLevelProvider(PriceLevelProvider):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        self.extractor = AbuPriceStructureExtractor(self.config)

    def precompute(self, trade_date: date) -> None:
        eng = _get_engine()

        basics = get_all_stock_basics()
        ts_codes = [b.ts_code for b in basics]
        limit = self.config.get("stock_pool_limit")
        if isinstance(limit, int) and limit > 0:
            ts_codes = ts_codes[:limit]

        if not ts_codes:
            print("[AbuPriceLevelProvider] stock_basic is empty.")
            return

        total = len(ts_codes)
        debug = bool(self.config.get("debug_print"))

        with eng.begin() as conn:
            for i, ts_code in enumerate(ts_codes, start=1):
                df_hist = self._load_history(conn, ts_code, trade_date)
                if df_hist.empty:
                    if debug and i <= 5:
                        print(f"[AbuPriceLevelProvider] {ts_code}: no history for {trade_date}")
                    continue

                levels = self.extractor.extract_levels_for_history(ts_code, df_hist, trade_date)
                if not levels:
                    continue

                self._write_levels_for_stock(conn, levels)

                if i % 50 == 0 or (debug and i <= 5):
                    print(f"[AbuPriceLevelProvider] {i}/{total} {ts_code}: {len(levels)} levels generated.")

        print(f"[AbuPriceLevelProvider] precompute done for {trade_date}, stocks={total}")

    def get_levels(self, ts_code: str, trade_date: date) -> List[PriceLevel]:
        eng = _get_engine()
        if not _table_exists(eng, "price_levels_daily"):
            return []
        sql_txt = (
            "select ts_code, trade_date, level_price, level_type, "
            "direction, strength, source_flags, meta "
            "from price_levels_daily "
            "where ts_code = %(ts)s and trade_date = %(d)s"
        )
        df = pd.read_sql(sql_txt, eng, params={"ts": ts_code, "d": trade_date})
        if df.empty:
            return []

        levels: List[PriceLevel] = []
        for _, row in df.iterrows():
            td = row["trade_date"]
            if isinstance(td, datetime):
                td = td.date()

            flags_raw = row.get("source_flags")
            if isinstance(flags_raw, str) and flags_raw:
                try:
                    source_flags = json.loads(flags_raw)
                    if not isinstance(source_flags, list):
                        source_flags = [str(source_flags)]
                except Exception:
                    source_flags = [flags_raw]
            elif isinstance(flags_raw, list):
                source_flags = flags_raw
            else:
                source_flags = []

            meta_raw = row.get("meta")
            if isinstance(meta_raw, str) and meta_raw:
                try:
                    meta = json.loads(meta_raw)
                    if not isinstance(meta, dict):
                        meta = {"value": meta}
                except Exception:
                    meta = {}
            elif isinstance(meta_raw, dict):
                meta = meta_raw
            else:
                meta = {}

            levels.append(PriceLevel(ts_code=row["ts_code"], trade_date=td, level_price=float(row["level_price"]), level_type=row["level_type"], direction=row["direction"], strength=int(row["strength"]), source_flags=source_flags, meta=meta))

        return levels

    def _load_history(self, conn, ts_code: str, trade_date: date) -> pd.DataFrame:
        lookback_days = int(self.config.get("lookback_days", 250))
        extra = 20
        total_limit = lookback_days + extra

        sql_txt = (
            "select trade_date, open, high, low, close, volume, amount "
            f"from stock_daily where ts_code = '{ts_code}' "
            f"and trade_date <= '{trade_date.isoformat()}' "
            "order by trade_date desc "
            f"limit {total_limit}"
        )
        df = pd.read_sql(sql_txt, conn)
        if df.empty:
            return df

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    def _write_levels_for_stock(self, conn, levels: List[PriceLevel]) -> None:
        if not levels:
            return

        ts_code = levels[0].ts_code
        trade_date = levels[0].trade_date

        rows = []
        for lv in levels:
            rows.append({"ts_code": lv.ts_code, "trade_date": lv.trade_date, "level_price": lv.level_price, "level_type": lv.level_type, "direction": lv.direction, "strength": int(lv.strength), "source_flags": json.dumps(lv.source_flags, ensure_ascii=False), "meta": json.dumps(lv.meta or {}, ensure_ascii=False)})
        df = pd.DataFrame(rows)

        if _table_exists(conn, "price_levels_daily"):
            conn.execute(text("delete from price_levels_daily where ts_code = :ts and trade_date = :d"), {"ts": ts_code, "d": trade_date})

        df.to_sql("price_levels_daily", con=conn, if_exists="append", index=False)


if __name__ == "__main__":
    from datetime import timedelta
    import random

    print("[abu_price_levels] self test...")

    today = date.today()
    price = 10.0
    records = []
    for i in range(120):
        d = today - timedelta(days=119 - i)
        if 20 < i < 40:
            delta = random.uniform(-0.2, 0.2)
        elif 60 < i < 80:
            delta = random.uniform(-0.3, 0.3)
        else:
            delta = random.uniform(-0.1, 0.4)
        price = max(3.0, price + delta)
        high = price + random.uniform(0.0, 0.2)
        low = price - random.uniform(0.0, 0.2)
        open_p = (high + low) / 2
        close = price
        vol = 100000 + i * 500
        records.append({"trade_date": d, "open": open_p, "high": high, "low": low, "close": close, "volume": vol, "amount": vol * close})

    hist_df = pd.DataFrame(records)
    extractor = AbuPriceStructureExtractor()
    levels = extractor.extract_levels_for_history("000001.SZ", hist_df, today)

    print(f"Computed {len(levels)} levels for 000001.SZ:")
    for lv in levels:
        print(f"{lv.level_type:22s} {lv.direction:10s} price={lv.level_price:8.2f} strength={lv.strength}")

    provider = AbuPriceLevelProvider()
    provider.precompute(today)
