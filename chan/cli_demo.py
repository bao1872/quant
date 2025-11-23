from __future__ import annotations

import argparse
from typing import Sequence
from pathlib import Path

from .engine import ChanRunConfig, build_cchan_by_run_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run chan.py calculation demo inside quant project.")
    p.add_argument("--code", required=True)
    p.add_argument("--begin", dest="begin_time", default=None)
    p.add_argument("--end", dest="end_time", default=None)
    p.add_argument("--lv", dest="lv_list", nargs="+", default=["day"]) 
    p.add_argument("--autype", default="qfq")
    p.add_argument("--data-src", dest="data_src", default="baostock")
    p.add_argument("--chan-config", dest="chan_cfg_path", default=None)
    return p.parse_args()


def _len_attr(x, attr):
    if x is None:
        return -1
    v = getattr(x, attr, x)
    return len(v)


def main() -> None:
    args = parse_args()
    run_cfg = ChanRunConfig(
        code=args.code,
        begin_time=args.begin_time,
        end_time=args.end_time,
        lv_list=args.lv_list,
        autype=args.autype,
        data_src=args.data_src,
    )
    chan = build_cchan_by_run_config(run_cfg, chan_cfg_path=args.chan_cfg_path)
    print(f"=== CChan created for {args.code} ===")
    for i, lv in enumerate(chan.lv_list):
        cur = chan[lv] if lv in chan.kl_datas else chan[i]
        bi_list = getattr(cur, "bi_list", None)
        seg_list = getattr(cur, "seg_list", None)
        zs_list = getattr(cur, "zs_list", None)
        bsp_list = getattr(cur, "bs_point_lst", None)
        print(f"--- Level: {lv} ---")
        if bi_list is not None:
            print("  bi_list:", _len_attr(bi_list, "bi_list"))
        if seg_list is not None:
            print("  seg_list:", _len_attr(seg_list, "seg_list"))
        if zs_list is not None:
            print("  zs_list:", _len_attr(zs_list, "zs_list"))
        if bsp_list is not None:
            print("  bs_point:", _len_attr(bsp_list, "bs_point_list"))
    print("Done.")


if __name__ == "__main__":
    main()