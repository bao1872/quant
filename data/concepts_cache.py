from __future__ import annotations

import importlib.util
import pandas as pd
from typing import List, Tuple

from db.connection import get_engine
from sqlalchemy import text


def _has_pywencai() -> bool:
    return importlib.util.find_spec("pywencai") is not None


def _format_ts_code(code: str) -> str:
    s = str(code)
    if "." in s:
        return s.upper()
    p = s.zfill(6)
    return (p + ".SZ") if p.startswith(("0", "3")) else (p + ".SH")


def _table_exists(eng, table_name: str) -> bool:
    q = (
        "select 1 from information_schema.tables where table_schema='public' and table_name='"
        + table_name
        + "'"
    )
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)
        return len(df) > 0


def _delete_by_ts_codes(eng, table_name: str, ts_codes: List[str]) -> None:
    if not ts_codes:
        return
    with eng.begin() as conn:
        payload = [{"ts": c} for c in ts_codes]
        conn.execute(text(f"delete from {table_name} where ts_code=:ts"), payload)


def _upsert_df(eng, table_name: str, df: pd.DataFrame, key: str = "ts_code") -> int:
    if df is None or df.empty:
        return 0
    if _table_exists(eng, table_name):
        with eng.connect() as conn:
            existing = pd.read_sql(f"select {key} from {table_name}", conn)
        if not existing.empty and key in existing.columns:
            exists = existing[key].astype(str).tolist()
            to_del = df[df[key].astype(str).isin(exists)][key].astype(str).tolist()
            _delete_by_ts_codes(eng, table_name, to_del)
    df.to_sql(table_name, eng, if_exists="append", index=False)
    return len(df)


def _transform_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code", "name", "concepts", "industry", "popularity_rank", "market_cap"])[:0]
    cols = df.columns.astype(str).tolist()
    concept_col = next((c for c in cols if "所属概念" in c), None)
    industry_col = next((c for c in cols if "所属同花顺行业" in c), None)
    rank_col = next((c for c in cols if ("个股热度排名" in c or "人气" in c)), None)
    mkt_col = next((c for c in cols if ("流通" in c or "限售" in c or "市值" in c)), None)
    # 必须保留所有数据，不因人气值缺失或高低进行删除；人气列可选。
    need = ["股票代码", "股票简称"]
    if any(x is None for x in need) or concept_col is None:
        return pd.DataFrame(columns=["ts_code", "name", "concepts", "industry", "popularity_rank", "market_cap"])[:0]
    base = df[["股票代码", "股票简称", concept_col]].copy()
    if industry_col:
        base[industry_col] = df[industry_col]
    if rank_col:
        base[rank_col] = df[rank_col]
    if mkt_col:
        base[mkt_col] = df[mkt_col]
    rename = {"股票简称": "name"}
    rename[concept_col] = "concepts"
    if rank_col:
        rename[rank_col] = "popularity_rank"
    if mkt_col:
        rename[mkt_col] = "market_cap"
    if industry_col:
        rename[industry_col] = "industry"
    base = base.rename(columns=rename)
    base["ts_code"] = base["股票代码"].astype(str).map(_format_ts_code)
    # 数值转换但不因人气而过滤；缺失值保留
    if "popularity_rank" in base.columns:
        base["popularity_rank"] = pd.to_numeric(base["popularity_rank"], errors="coerce")
    if "market_cap" in base.columns:
        base["market_cap"] = pd.to_numeric(base["market_cap"], errors="coerce")
    out_cols = ["ts_code", "name", "concepts"]
    if "market_cap" in base.columns:
        out_cols.append("market_cap")
    if "popularity_rank" in base.columns:
        out_cols.append("popularity_rank")
    if "industry" in base.columns:
        out_cols.insert(3, "industry")
    return base[out_cols]


def update_concepts_cache() -> int:
    eng = get_engine()
    if not _has_pywencai():
        print("[concepts_cache] missing dependency: please install pywencai")
        return 0
    import pywencai as wc
    res = wc.get(query="非st，主板或创业板或科创板，所属概念，行业分类，人气排名，流通市值", loop=True, sleep=2, cookie=None)
    if res is None or res.empty:
        return 0
    df = _transform_raw_data(res)
    if df is None or df.empty:
        return 0
    n = _upsert_df(eng, "concepts_cache", df, key="ts_code")
    return n


def _transform_hk_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code", "name", "hk_industry"])[:0]
    cols = df.columns.astype(str).tolist()
    col = next((c for c in cols if "所属恒生行业" in c), None)
    if col is None:
        return pd.DataFrame(columns=["ts_code", "name", "hk_industry"])[:0]
    base = df[["股票代码", "股票简称", col]].copy()
    base = base.rename(columns={"股票简称": "name", col: "hk_industry"})
    def _fmt_hk(code: str) -> str:
        s = str(code)
        if "." in s:
            return s.upper()
        return s.zfill(5) + ".HK"
    base["ts_code"] = base["股票代码"].astype(str).map(_fmt_hk)
    base = base.dropna(subset=["hk_industry"]).copy()
    return base[["ts_code", "name", "hk_industry"]]


def update_hk_industry_cache() -> int:
    eng = get_engine()
    if not _has_pywencai():
        print("[hk_industry_cache] missing dependency: please install pywencai")
        return 0
    import pywencai as wc
    opts: List[Tuple[str, str]] = [
        ("所属恒生行业", "hkstock"),
        ("行业分类", "hkstock"),
        ("行业", "hkstock"),
        ("", "hkstock"),
    ]
    res = None
    for q, qt in opts:
        r = wc.get(query=q, query_type=qt, loop=True, sleep=2, cookie=None)
        if r is not None and not r.empty:
            res = r
            break
    if res is None or res.empty:
        return 0
    df = _transform_hk_raw_data(res)
    if df is None or df.empty:
        return 0
    n = _upsert_df(eng, "hk_industry_cache", df, key="ts_code")
    return n


def validate_concepts_cache_count() -> None:
    eng = get_engine()
    if not _table_exists(eng, "concepts_cache"):
        print("[concepts_cache] table not exists")
        return
    with eng.connect() as conn:
        dfc = pd.read_sql("select count(distinct ts_code) as cnt from concepts_cache", conn)
    from data import repository
    basics = repository.get_all_stock_basics()
    total = len(basics)
    cached = int(dfc["cnt"].iloc[0]) if not dfc.empty else 0
    if cached != total:
        print(f"[concepts_cache] WARNING: cached {cached} != universe {total}")
    else:
        print(f"[concepts_cache] OK: cached {cached} matches universe {total}")


def reconcile_concepts_cache_with_universe() -> int:
    eng = get_engine()
    if not _table_exists(eng, "concepts_cache"):
        return 0
    with eng.connect() as conn:
        dfc = pd.read_sql("select ts_code, name from concepts_cache", conn)
    from data import repository
    basics = repository.get_all_stock_basics()
    codes = [b.ts_code for b in basics]
    dfu = pd.DataFrame({"ts_code": pd.Series(codes, dtype=str)})
    # 尝试从 concepts_cache 中匹配名称；若无则留空
    name_map = {}
    if not dfc.empty and "ts_code" in dfc.columns and "name" in dfc.columns:
        name_map = dict(zip(dfc["ts_code"].astype(str), dfc["name"].astype(str)))
    missing = sorted(set(dfu["ts_code"].astype(str)) - set(dfc["ts_code"].astype(str)))
    if not missing:
        return 0
    to_insert = pd.DataFrame({
        "ts_code": missing,
        "name": [name_map.get(ts, "") for ts in missing],
        "concepts": ["" for _ in missing],
        "industry": [None for _ in missing],
        "market_cap": [None for _ in missing],
    })
    to_insert.to_sql("concepts_cache", eng, if_exists="append", index=False)
    return len(to_insert)


def purge_non_universe_from_concepts_cache() -> int:
    eng = get_engine()
    if not _table_exists(eng, "concepts_cache"):
        return 0
    with eng.connect() as conn:
        dfc = pd.read_sql("select ts_code from concepts_cache", conn)
    from data import repository
    basics = repository.get_all_stock_basics()
    codes = set(b.ts_code for b in basics)
    extras = sorted(set(dfc["ts_code"].astype(str)) - codes)
    if not extras:
        return 0
    _delete_by_ts_codes(eng, "concepts_cache", extras)
    return len(extras)


if __name__ == "__main__":
    sample = pd.DataFrame({
        "股票代码": ["000001", "600000"],
        "股票简称": ["平安银行", "浦发银行"],
        "所属概念": ["银行;数字货币", "银行"],
        "个股热度排名": [100, 200],
        "流通市值": [1e11, 2e11],
        "所属同花顺行业": ["银行", "银行"],
    })
    df = _transform_raw_data(sample)
    print(len(df))
