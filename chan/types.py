from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union


@dataclass
class Fractal:
    """分型结构。暂不从 CChan 抽取，保持接口占位。"""
    code: str
    freq: str
    ts: datetime
    fx_type: str
    high: float
    low: float
    k_index_start: int
    k_index_end: int


@dataclass
class Bi:
    """缠论笔。来源于 CChan 的 CBi。"""
    code: str
    freq: str
    bi_id: int
    direction: str
    start_ts: datetime
    end_ts: datetime
    start_price: float
    end_price: float
    high: float
    low: float
    start_fx_index: Optional[int]
    end_fx_index: Optional[int]
    length_k: Optional[int]
    amplitude: Optional[float]
    slope: Optional[float]


@dataclass
class Segment:
    """线段。来源于 CChan 的 CSeg。"""
    code: str
    freq: str
    seg_id: int
    direction: str
    start_ts: datetime
    end_ts: datetime
    high: float
    low: float
    start_bi_index: Optional[int]
    end_bi_index: Optional[int]
    bi_count: Optional[int]
    amplitude: Optional[float]
    slope: Optional[float]
    center_count: Optional[int]


@dataclass
class Center:
    """中枢。来源于 CChan 的 CZS。"""
    code: str
    freq: str
    center_id: int
    start_ts: datetime
    end_ts: datetime
    high: float
    low: float
    related_seg_index: Optional[int]
    bi_count: Optional[int]
    enter_times: Optional[int]
    leave_times: Optional[int]


@dataclass
class Signal:
    """买卖点/缠论信号。来源于 CChan 的 CBS_Point。"""
    code: str
    freq: str
    signal_id: int
    ts: datetime
    signal_type: str
    price: Optional[float]
    related_bi_index: Optional[int]
    related_seg_index: Optional[int]
    related_center_index: Optional[int]
    extra_info: Optional[Dict[str, Union[float, int, str]]]


@dataclass
class ChanResult:
    """某标的+级别的缠论聚合结果。"""
    code: str
    freq: str
    fractals: List[Fractal]
    bis: List[Bi]
    segments: List[Segment]
    centers: List[Center]
    signals: List[Signal]