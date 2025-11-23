from __future__ import annotations

from typing import Dict, List

from .engine import ChanRunConfig, build_cchan_by_run_config
from .extract import extract_all_levels
from .types import ChanResult
from .utils import ts_to_chan_code
from .persist import save_chan_result_to_db


def run_chan_for_symbol(
    code: str,
    lv_list: List[str],
    begin_time: str | None = None,
    end_time: str | None = None,
    data_src: str = "baostock",
    autype: str = "qfq",
) -> Dict[str, ChanResult]:
    run_cfg = ChanRunConfig(
        code=ts_to_chan_code(code),
        begin_time=begin_time,
        end_time=end_time,
        lv_list=lv_list,
        autype=autype,
        data_src=data_src,
    )
    chan = build_cchan_by_run_config(run_cfg)
    return extract_all_levels(chan)


def run_chan_for_symbol_to_db(
    code: str,
    lv_list: List[str],
    begin_time: str | None = None,
    end_time: str | None = None,
    data_src: str = "baostock",
    autype: str = "qfq",
) -> None:
    res = run_chan_for_symbol(code, lv_list, begin_time, end_time, data_src, autype)
    for _, r in res.items():
        save_chan_result_to_db(r)


def run_chan_for_df(
    ts_code: str,
    freq: str,
    kline_df,
) -> Dict[str, ChanResult]:
    import pandas as pd
    from Common.CEnum import DATA_FIELD, KL_TYPE
    from Common.CTime import CTime
    from KLine.KLine_Unit import CKLine_Unit

    lv_alias = {
        "1d": "day",
        "60m": "60m",
        "30m": "30m",
        "15m": "15m",
        "5m": "5m",
        "1m": "1m",
    }
    freq_key = str(freq).lower()
    lv_name = lv_alias.get(freq_key, freq_key)
    run_cfg = ChanRunConfig(code=ts_to_chan_code(ts_code), lv_list=[lv_name], autype="qfq", data_src="csv")
    chan = build_cchan_by_run_config(run_cfg, chan_cfg_overrides={"trigger_step": True})

    df = kline_df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    kl_list = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row["datetime"])  # type: ignore
        t = CTime(int(dt.year), int(dt.month), int(dt.day), int(dt.hour), int(dt.minute))
        kl_dict = {
            DATA_FIELD.FIELD_TIME: t,
            DATA_FIELD.FIELD_OPEN: float(row["open"]),
            DATA_FIELD.FIELD_HIGH: float(row["high"]),
            DATA_FIELD.FIELD_LOW: float(row["low"]),
            DATA_FIELD.FIELD_CLOSE: float(row["close"]),
        }
        kl_list.append(CKLine_Unit(kl_dict))
    m = {
        "1d": KL_TYPE.K_DAY,
        "60m": KL_TYPE.K_60M,
        "30m": KL_TYPE.K_30M,
        "15m": KL_TYPE.K_15M,
        "5m": KL_TYPE.K_5M,
        "1m": KL_TYPE.K_1M,
    }
    chan.trigger_load({m[freq_key]: kl_list})
    for lv in chan.lv_list:
        chan.kl_datas[lv].cal_seg_and_zs()
    return extract_all_levels(chan)

if __name__ == "__main__":
    from Common.CEnum import DATA_FIELD
    from Common.CTime import CTime
    from KLine.KLine_Unit import CKLine_Unit
    from .engine import KL_TYPE

    run_cfg = ChanRunConfig(code="SH.600000", lv_list=["day"], data_src="csv")
    chan = build_cchan_by_run_config(run_cfg, chan_cfg_overrides={
        "trigger_step": True,
        "bi_algo": "fx",
        "bi_strict": False,
        "bi_fx_check": "loss",
        "gap_as_kl": True,
    })

    base = [
        10.0, 11.0, 10.5, 12.0, 9.5, 13.0, 10.0, 13.5, 9.0, 14.0,
        10.5, 14.5, 10.0, 15.0, 10.0, 15.5, 10.0, 16.0, 10.0, 16.5,
    ]
    klu_day = []
    for i, b in enumerate(base):
        y, m, d = 2023, 1 + (i // 20), 1 + i
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
    for _, r in out.items():
        save_chan_result_to_db(r)
    print("written:", ", ".join([f for f in out.keys()]))