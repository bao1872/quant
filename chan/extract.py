from __future__ import annotations

from datetime import datetime
from typing import Dict
import sys

from .types import Bi, Center, ChanResult, Segment, Signal
from .engine import KL_TYPE, build_cchan_by_run_config, ChanRunConfig

PROJECT_ROOT_ADDED = False
if not PROJECT_ROOT_ADDED:
    PROJECT_ROOT_ADDED = True


def _ctime_to_dt(ct) -> datetime:
    return datetime(ct.year, ct.month, ct.day, ct.hour, ct.minute, ct.second)


def kl_type_to_freq_str(kl_type) -> str:
    m = {
        KL_TYPE.K_DAY: "1d",
        KL_TYPE.K_60M: "60m",
        KL_TYPE.K_30M: "30m",
        KL_TYPE.K_15M: "15m",
        KL_TYPE.K_5M: "5m",
        KL_TYPE.K_1M: "1m",
        KL_TYPE.K_WEEK: "week",
        KL_TYPE.K_MON: "month",
        KL_TYPE.K_YEAR: "year",
        KL_TYPE.K_3M: "3m",
        KL_TYPE.K_QUARTER: "quarter",
    }
    return m.get(kl_type, str(kl_type))


def _bi_direction_str(bi) -> str:
    return "UP" if bi.is_up() else "DOWN"


def _seg_direction_str(seg) -> str:
    return "UP" if seg.is_up() else "DOWN"


def _signal_type_str(bsp) -> str:
    t = bsp.type[0].value
    if t == "1":
        return "BUY1" if bsp.is_buy else "SELL1"
    if t == "1p":
        return "BUY1P" if bsp.is_buy else "SELL1P"
    if t == "2":
        return "BUY2" if bsp.is_buy else "SELL2"
    if t == "2s":
        return "BUY2S" if bsp.is_buy else "SELL2S"
    if t == "3a":
        return "BUY3A" if bsp.is_buy else "SELL3A"
    if t == "3b":
        return "BUY3B" if bsp.is_buy else "SELL3B"
    return bsp.type2str()


def extract_level_structures(chan, lv) -> ChanResult:
    kl = chan[lv]
    code = str(chan.code)
    freq = kl_type_to_freq_str(lv)

    bis = [
        Bi(
            code=code,
            freq=freq,
            bi_id=bi.idx,
            direction=_bi_direction_str(bi),
            start_ts=_ctime_to_dt(bi.get_begin_klu().time),
            end_ts=_ctime_to_dt(bi.get_end_klu().time),
            start_price=bi.get_begin_val(),
            end_price=bi.get_end_val(),
            high=bi._high(),
            low=bi._low(),
            start_fx_index=None,
            end_fx_index=None,
            length_k=bi.get_klu_cnt(),
            amplitude=bi.amp(),
            slope=(bi.get_end_val() - bi.get_begin_val()) / max(bi.get_klu_cnt(), 1),
        )
        for bi in kl.bi_list.bi_list
    ]

    segments = [
        Segment(
            code=code,
            freq=freq,
            seg_id=seg.idx,
            direction=_seg_direction_str(seg),
            start_ts=_ctime_to_dt(seg.get_begin_klu().time),
            end_ts=_ctime_to_dt(seg.get_end_klu().time),
            high=seg._high(),
            low=seg._low(),
            start_bi_index=seg.start_bi.idx,
            end_bi_index=seg.end_bi.idx,
            bi_count=seg.cal_bi_cnt(),
            amplitude=seg.amp(),
            slope=(seg.get_end_val() - seg.get_begin_val()) / max(seg.get_klu_cnt(), 1),
            center_count=len(seg.zs_lst),
        )
        for seg in kl.seg_list
    ]

    centers = [
        Center(
            code=code,
            freq=freq,
            center_id=i,
            start_ts=_ctime_to_dt(zs.begin.time),
            end_ts=_ctime_to_dt(zs.end.time),
            high=zs.high,
            low=zs.low,
            related_seg_index=zs.begin_bi.seg_idx,
            bi_count=len(zs.bi_lst) if hasattr(zs, "bi_lst") else None,
            enter_times=None,
            leave_times=None,
        )
        for i, zs in enumerate(kl.zs_list)
    ]

    signals = [
        Signal(
            code=code,
            freq=freq,
            signal_id=i,
            ts=_ctime_to_dt(bsp.klu.time),
            signal_type=_signal_type_str(bsp),
            price=bsp.klu.close,
            related_bi_index=bsp.bi.idx if hasattr(bsp, "bi") else None,
            related_seg_index=(bsp.bi.parent_seg.idx if getattr(bsp.bi, "parent_seg", None) is not None else None),
            related_center_index=None,
            extra_info=dict(bsp.features.items()) if hasattr(bsp, "features") else None,
        )
        for i, bsp in enumerate(kl.bs_point_lst.getSortedBspList())
    ]

    return ChanResult(
        code=code,
        freq=freq,
        fractals=[],
        bis=bis,
        segments=segments,
        centers=centers,
        signals=signals,
    )


def extract_all_levels(chan) -> Dict[str, ChanResult]:
    res: Dict[str, ChanResult] = {}
    for lv in chan.lv_list:
        cr = extract_level_structures(chan, lv)
        res[cr.freq] = cr
    return res


if __name__ == "__main__":
    from Common.CEnum import DATA_FIELD
    from Common.CTime import CTime
    from KLine.KLine_Unit import CKLine_Unit
    from .engine import KL_TYPE

    run = ChanRunConfig(code="SH.600000", lv_list=["day"], autype="qfq", data_src="csv")
    chan = build_cchan_by_run_config(run, chan_cfg_overrides={"trigger_step": True, "bi_algo": "fx", "bi_strict": False})
    base = [10.0, 11.0, 10.5, 12.0, 9.5, 13.0, 10.0, 13.5, 9.0, 14.0, 10.5, 14.5]
    klu_day = []
    for i, b in enumerate(base):
        y, m, d = 2023, 1, 1 + i
        t = CTime(y, m, d, 0, 0)
        high = b + (0.2 + 0.1 * (i % 3))
        low = b - (0.2 + 0.1 * (i % 3))
        kl_dict = {
            DATA_FIELD.FIELD_TIME: t,
            DATA_FIELD.FIELD_OPEN: b,
            DATA_FIELD.FIELD_HIGH: high,
            DATA_FIELD.FIELD_LOW: low,
            DATA_FIELD.FIELD_CLOSE: b,
        }
        klu_day.append(CKLine_Unit(kl_dict))
    chan.trigger_load({KL_TYPE.K_DAY: klu_day})
    out = extract_all_levels(chan)
    for f, r in out.items():
        print(f, len(r.bis), len(r.segments), len(r.centers), len(r.signals))