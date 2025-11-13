from __future__ import annotations

import pandas as pd

from config import MARKET_SZ, MARKET_SH
from db.connection import get_engine
from sqlalchemy import text
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
    def _classify_board(code: str, market_name: str) -> str:
        if market_name == "SZ":
            if code.startswith(("000", "001", "003")):
                return "SZ_MAIN"
            if code.startswith("002"):
                return "SZ_SMEB"
            if code.startswith(("300", "301")):
                return "SZ_CHINEXT"
            return ""
        if code.startswith(("600", "601", "603", "605")):
            return "SH_MAIN"
        if code.startswith(("688", "689")):
            return "SH_STAR"
        return ""
    df["board"] = df.apply(lambda r: _classify_board(str(r["code"]), str(r["market"])), axis=1)
    df = df[df["board"] != ""].copy()
    df = df[["ts_code", "code", "name", "market", "board"]].drop_duplicates(subset=["ts_code"]).sort_values(["market", "code"]).reset_index(drop=True)
    return df


def ensure_stock_basic_a_share() -> int:
    df = load_a_share_universe()
    eng = get_engine()
    if eng is None:
        return 0
    def _table_exists(table_name: str) -> bool:
        q = (
            "select 1 from information_schema.tables where table_schema='public' and table_name='"
            + table_name
            + "'"
        )
        chk = pd.read_sql(q, eng)
        return len(chk) > 0
    def _table_columns(table_name: str) -> list[str]:
        q = (
            "select column_name from information_schema.columns where table_schema='public' and table_name='"
            + table_name
            + "' order by ordinal_position"
        )
        df_cols = pd.read_sql(q, eng)
        return [str(c) for c in df_cols["column_name"].tolist()]
    if _table_exists("stock_basic"):
        with eng.begin() as conn:
            conn.execute(text("truncate table stock_basic"))
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
    cols = _table_columns("stock_basic") if _table_exists("stock_basic") else ["ts_code", "code", "exchange", "name"]
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
    df_out.to_sql("stock_basic", eng, if_exists="append", index=False)
    return len(df_out)


if __name__ == "__main__":
    import os
    os.environ["USE_REAL_DB"] = "1"
    cnt = ensure_stock_basic_a_share()
    print(cnt)
