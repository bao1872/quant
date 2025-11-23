from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANPY_ROOT = PROJECT_ROOT / "third_party" / "chan.py"

if str(CHANPY_ROOT) not in sys.path:
    sys.path.insert(0, str(CHANPY_ROOT))

from Chan import CChan  # type: ignore
from ChanConfig import CChanConfig  # type: ignore
from Common.CEnum import AUTYPE, KL_TYPE, DATA_SRC  # type: ignore

from .config_loader import build_cchan_config

LV_ALIAS = {
    "year": KL_TYPE.K_YEAR,
    "quarter": KL_TYPE.K_QUARTER,
    "mon": KL_TYPE.K_MON,
    "month": KL_TYPE.K_MON,
    "week": KL_TYPE.K_WEEK,
    "day": KL_TYPE.K_DAY,
    "d": KL_TYPE.K_DAY,
    "60m": KL_TYPE.K_60M,
    "30m": KL_TYPE.K_30M,
    "15m": KL_TYPE.K_15M,
    "5m": KL_TYPE.K_5M,
    "3m": KL_TYPE.K_3M,
    "1m": KL_TYPE.K_1M,
}

AUTYPE_ALIAS = {
    "qfq": AUTYPE.QFQ,
    "hfq": AUTYPE.HFQ,
    "none": AUTYPE.NONE,
}

DATA_SRC_ALIAS = {
    "baostock": DATA_SRC.BAO_STOCK,
    "csv": DATA_SRC.CSV,
    "ccxt": DATA_SRC.CCXT,
}


@dataclass
class ChanRunConfig:
    code: str
    begin_time: Optional[str] = None
    end_time: Optional[str] = None
    lv_list: Sequence[str] = ("day",)
    autype: str = "qfq"
    data_src: str = "baostock"


def _normalize_lv_list(lv_list: Iterable[str]) -> List[KL_TYPE]:
    res: List[KL_TYPE] = []
    for lv in lv_list:
        key = str(lv).lower()
        if key not in LV_ALIAS:
            raise KeyError(key)
        res.append(LV_ALIAS[key])
    return res


def _normalize_autype(autype: str) -> AUTYPE:
    key = autype.lower()
    if key not in AUTYPE_ALIAS:
        raise KeyError(key)
    return AUTYPE_ALIAS[key]


def _normalize_data_src(data_src: str):
    key = data_src.lower()
    if key in DATA_SRC_ALIAS:
        return DATA_SRC_ALIAS[key]
    return data_src


def build_cchan(
    code: str,
    begin_time: Optional[str] = None,
    end_time: Optional[str] = None,
    lv_list: Sequence[str] = ("day"),
    autype: str = "qfq",
    data_src: str = "baostock",
    chan_cfg_path: Optional[str | Path] = None,
    chan_cfg_overrides: Optional[dict] = None,
) -> CChan:
    lv_enum_list = _normalize_lv_list(lv_list)
    autype_enum = _normalize_autype(autype)
    data_src_obj = _normalize_data_src(data_src)
    config: CChanConfig = build_cchan_config(
        path=str(chan_cfg_path) if chan_cfg_path is not None else None,
        extra_overrides=chan_cfg_overrides,
    )
    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src_obj,
        lv_list=lv_enum_list,
        config=config,
        autype=autype_enum,
    )
    return chan


def build_cchan_by_run_config(
    run_cfg: ChanRunConfig,
    chan_cfg_path: Optional[str | Path] = None,
    chan_cfg_overrides: Optional[dict] = None,
) -> CChan:
    return build_cchan(
        code=run_cfg.code,
        begin_time=run_cfg.begin_time,
        end_time=run_cfg.end_time,
        lv_list=run_cfg.lv_list,
        autype=run_cfg.autype,
        data_src=run_cfg.data_src,
        chan_cfg_path=chan_cfg_path,
        chan_cfg_overrides=chan_cfg_overrides,
    )


if __name__ == "__main__":
    chan = build_cchan(code="SH.600000", lv_list=("day",), data_src="csv")
    kl = chan[0]
    bi_cnt = len(getattr(kl, "bi_list", []))
    seg_cnt = len(getattr(kl, "seg_list", []))
    zs_cnt = len(getattr(kl, "zs_list", []))
    bsp_cnt = len(getattr(kl, "bs_point_lst", []))
    print("bi_list:", bi_cnt)
    print("seg_list:", seg_cnt)
    print("zs_list:", zs_cnt)
    print("bs_point:", bsp_cnt)