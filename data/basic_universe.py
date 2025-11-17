from __future__ import annotations

import pandas as pd

from config import MARKET_SZ, MARKET_SH
from db.connection import get_engine
from sqlalchemy import text, inspect
from .pytdx_source import PytdxDataSource
from tqdm import tqdm


def load_a_share_universe() -> pd.DataFrame:
    with PytdxDataSource() as ds:
        df_all = ds.fetch_all_stock_list()
    if df_all.empty:
        return pd.DataFrame(columns=["ts_code", "code", "name", "market", "board"])
    df = df_all.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["exchange"] = df["exchange"].astype(str).str.upper()
    df["name"] = df["name"].astype(str)
    is_sz = df["exchange"].eq("SZ")
    is_sh = df["exchange"].eq("SH")
    prefix = df["code"].str.slice(0, 3)
    sz_ok = is_sz & prefix.isin(["000", "001", "002", "003", "300", "301"])
    sh_ok = is_sh & prefix.isin(["600", "601", "603", "605", "688", "689"])
    name_upper = df["name"].str.upper()
    is_st = name_upper.str.contains("ST") | df["name"].str.contains("退")
    df = df[(sz_ok | sh_ok) & (~is_st)].copy()
    df["market"] = df["exchange"]
    df["ts_code"] = df["code"] + "." + df["market"]
    sz_main = df["market"].eq("SZ") & df["code"].str.startswith(tuple(["000", "001", "003"]))
    sz_smeb = df["market"].eq("SZ") & df["code"].str.startswith("002")
    sz_chinext = df["market"].eq("SZ") & df["code"].str.startswith(tuple(["300", "301"]))
    sh_main = df["market"].eq("SH") & df["code"].str.startswith(tuple(["600", "601", "603", "605"]))
    sh_star = df["market"].eq("SH") & df["code"].str.startswith(tuple(["688", "689"]))
    board = pd.Series([""] * len(df))
    board = board.mask(sz_main, "SZ_MAIN")
    board = board.mask(sz_smeb, "SZ_SMEB")
    board = board.mask(sz_chinext, "SZ_CHINEXT")
    board = board.mask(sh_main, "SH_MAIN")
    board = board.mask(sh_star, "SH_STAR")
    df["board"] = board
    df = df[df["board"] != ""].copy()
    df = df[["ts_code", "code", "name", "market", "board"]].drop_duplicates(subset=["ts_code"]).sort_values(["market", "code"]).reset_index(drop=True)
    return df


def ensure_stock_basic_a_share() -> int:
    df = load_a_share_universe()
    eng = get_engine()
    insp = inspect(eng)
    has_table = insp.has_table("stock_basic", schema="public")
    cols: list[str] = []
    if has_table:
        df_existing = pd.read_sql_table("stock_basic", eng)
        existing_ts = set(df_existing["ts_code"].astype(str).tolist()) if not df_existing.empty else set()
    else:
        existing_ts = set()
    # enrich shares info
    with PytdxDataSource() as ds:
        shares_rows = []
        for ts in tqdm(df["ts_code"].tolist(), desc="finance", unit="stk"):
            dfi = ds.fetch_finance_info(ts)
            if not dfi.empty:
                shares_rows.append(dfi)
        if shares_rows:
            df_shares = pd.concat(shares_rows, ignore_index=True)
            df = df.merge(df_shares, on="ts_code", how="left")
    cols = list(df.columns) if not has_table else df_existing.columns.astype(str).tolist()
    df_out = df.copy()
    df_out["exchange"] = df_out["market"].astype(str)
    if "is_st" in cols:
        df_out["is_st"] = False
    if "source" in cols:
        df_out["source"] = "pytdx"
    if "updated_at" in cols:
        from pandas import Timestamp
        df_out["updated_at"] = Timestamp.now()
    if "security_type" in cols:
        df_out["security_type"] = "stock"
    if "currency" in cols:
        df_out["currency"] = "CNY"
    if "total_shares" in cols and "total_shares" not in df_out.columns:
        df_out["total_shares"] = None
    if "float_shares" in cols and "float_shares" not in df_out.columns:
        df_out["float_shares"] = None
    out_cols = [c for c in [
        "ts_code","code","exchange","market","name","board","total_shares","float_shares","is_st","source","updated_at","security_type","currency"
    ] if c in cols]
    df_out = df_out[out_cols]
    if existing_ts:
        df_out = df_out[~df_out["ts_code"].isin(existing_ts)]
    df_out.to_sql("stock_basic", eng, if_exists="append", index=False)
    return len(df_out)


if __name__ == "__main__":
    import os
    os.environ["USE_REAL_DB"] = "1"
    cnt = ensure_stock_basic_a_share()
    print(cnt)
