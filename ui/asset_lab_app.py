from __future__ import annotations

import os
import sys
from typing import List, Dict, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data.repository import get_all_stock_basics
from data.source_factory import get_data_source
from db.connection import get_engine
from sqlalchemy import text
from data.pytdx_source import PytdxDataSource
from factors.ict_smc import compute_ict_structures, ICTConfig
from factors.harmonic_patterns import detect_harmonic_patterns
from backtest.bar_backtest import run_backtest_one_unit
from strategy.ict_rr3_simple import generate_rr3_long_signals, backtest_fullsize_rr3
import analysis.ob_swing_tuner as ob_swing_tuner
from strategy.ict_mtf_lab import IctMtfConfig, run_ict_mtf_backtest, attach_entry_signals, compute_daily_trend_with_fallback, TrendState
from datetime import date
from typing import Optional
from analysis.multi_tf_key_levels import CompositeLevel, build_composite_levels_for_symbol, SingleTFLevel, detect_single_tf_levels, detect_tf_levels_ict_ob_liq, merge_levels_within_tf


ASSET_LABEL_TO_CODE: Dict[str, str] = {
    "股票 (A股)": "stock",
    "股指期货 (占位)": "index_future",
    "国债期货 (占位)": "gov_bond",
}

SUPPORTED_FREQS = ["周线", "日线", "60分钟", "30分钟", "15分钟", "5分钟"]

_KEYLEVEL_CACHE: Dict[str, Dict[str, object]] = {}
_KEYLEVEL_CACHE_TTL_MINUTES: int = 60


def _load_stock_universe() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    eng = get_engine()
    df = pd.read_sql_table("stock_basic", eng) if eng is not None else pd.DataFrame(columns=["ts_code","name"])[:0]
    if not df.empty and ("ts_code" not in df.columns):
        market_map = {0: "SZ", 1: "SH", "SZ": "SZ", "SH": "SH"}
        if "market" in df.columns:
            exch = df["market"].map(market_map).fillna("SZ").astype(str)
        elif "exchange" in df.columns:
            exch = df["exchange"].map(market_map).fillna("SZ").astype(str)
        else:
            exch = pd.Series(["SZ"] * len(df))
        df["ts_code"] = df["code"].astype(str) + "." + exch
    if df.empty or ("ts_code" not in df.columns):
        basics = get_all_stock_basics()
        if not basics:
            return [], {}, {}
        ts_codes: List[str] = [b.ts_code for b in basics]
        names: List[str] = [getattr(b, "name", b.ts_code) for b in basics]
        df = pd.DataFrame({"ts_code": ts_codes, "name": names})
    import importlib, importlib.util
    has_py = importlib.util.find_spec("pypinyin") is not None
    if has_py:
        from pypinyin import lazy_pinyin
        clean = df["name"].astype(str).str.replace(r"\s+", "", regex=True)
        df["abbr"] = clean.map(lambda x: "".join([(s[0] if s else "") for s in lazy_pinyin(str(x))]).upper())
    else:
        df["abbr"] = ""
    df = df.sort_values(["abbr", "name"]).reset_index(drop=True)
    code_to_name = dict(zip(df["ts_code"], df["name"]))
    code_to_abbr = dict(zip(df["ts_code"], df["abbr"]))
    return df["ts_code"].tolist(), code_to_name, code_to_abbr


def _load_bars(asset_type: str, ts_code: str, freq_label: str, bar_count: int) -> pd.DataFrame:
    freq_map = {
        "日线": "1d",
        "60分钟": "60m",
        "30分钟": "30m",
        "15分钟": "15m",
        "5分钟": "5m",
    }
    fq = freq_map.get(freq_label, "1d")
    eng = get_engine()
    def _table_exists_local(eng, table_name: str) -> bool:
        if eng is None:
            return False
        with eng.connect() as conn:
            df = pd.read_sql(text("select 1 from information_schema.tables where table_schema='public' and table_name=:t"), conn, params={"t": table_name})
            return not df.empty
    df_db = pd.DataFrame()
    if _table_exists_local(eng, "stock_kline"):
        q = text(f"select datetime, open, high, low, close, volume from stock_kline where ts_code=:ts and freq=:fq order by datetime desc limit {int(bar_count)}")
        with eng.connect() as conn:
            df_db = pd.read_sql(q, conn, params={"ts": ts_code, "fq": fq}, parse_dates=["datetime"]) if eng is not None else pd.DataFrame()
    df_db = df_db.sort_values("datetime").reset_index(drop=True) if not df_db.empty else pd.DataFrame(columns=["datetime","open","high","low","close","volume"])[:0]
    if len(df_db) >= int(bar_count):
        if fq != "1d" and "datetime" in df_db.columns:
            per_day = df_db.groupby(pd.to_datetime(df_db["datetime"]).dt.date).size()
            exp = {"60m": 4, "30m": 8, "15m": 16, "5m": 48}.get(fq)
            if exp is not None:
                med = int(per_day.tail(10).median()) if len(per_day) > 0 else 0
                if med != exp:
                    df_db = pd.DataFrame(columns=["datetime","open","high","low","close","volume"])[:0]
        else:
            last_dt = pd.to_datetime(df_db["datetime"]).max() if not df_db.empty else None
            last_day = last_dt.date() if hasattr(last_dt, "date") else None
            if last_day is not None and last_day >= date.today():
                return df_db.tail(int(bar_count)).reset_index(drop=True)
    ds = PytdxDataSource()
    if fq == "1d":
        df_src = ds.get_daily_bars(ts_code, count=bar_count)
    else:
        # 对分钟频率，按与日线同区间进行分页抓取，保证日内根数正确
        df_day = ds.get_daily_bars(ts_code, count=max(bar_count, 240))
        if df_day is None or df_day.empty:
            df_src = ds.get_minute_bars(ts_code, freq=fq, count=bar_count)
        else:
            start_date = pd.to_datetime(df_day["datetime"]).dt.date.iloc[0]
            end_date = pd.to_datetime(df_day["datetime"]).dt.date.iloc[-1]
            df_src = ds.get_bars_range(ts_code, fq, start_date, end_date, page=600)
            keep = [c for c in ["datetime","open","high","low","close","volume"] if c in df_src.columns]
            df_src = df_src[keep].copy()
    if df_src is None or df_src.empty:
        return df_db.reset_index(drop=True)
    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df_src.columns]
    df_src = df_src[keep].copy()
    df_src["datetime"] = pd.to_datetime(df_src["datetime"]) if "datetime" in df_src.columns else pd.to_datetime([])
    if not df_db.empty:
        have = set(pd.to_datetime(df_db["datetime"]).tolist())
        df_src = df_src[~df_src["datetime"].isin(have)]
    df_all = pd.concat([df_db, df_src], ignore_index=True)
    df_all = df_all.sort_values("datetime").reset_index(drop=True)
    if len(df_all) > int(bar_count):
        df_all = df_all.tail(int(bar_count)).reset_index(drop=True)
    return df_all


def _build_demo_signal(df: pd.DataFrame) -> pd.Series:
    if "ict_choch_flag" not in df.columns:
        return pd.Series(0, index=df.index, name="signal")
    raw = df["ict_choch_flag"].fillna(0)
    pos_list: List[int] = []
    pos = 0
    for v in raw:
        if v > 0:
            pos = 1
        elif v < 0:
            pos = 0
        pos_list.append(pos)
    return pd.Series(pos_list, index=df.index, name="signal")


def _daily_trend_allow_long(df: pd.DataFrame) -> bool:
    bos = df.get("ict_bos_flag", pd.Series()).fillna(0)
    if bos.ne(0).any():
        last_bos = bos[bos.ne(0)].iloc[-1]
        if float(last_bos) > 0:
            return True
    sw = df.get("ict_sw_highlow", pd.Series()).fillna(0)
    lv = pd.to_numeric(df.get("ict_sw_level", pd.Series()), errors="coerce")
    highs = lv.where(sw > 0).dropna()
    lows = lv.where(sw < 0).dropna()
    if len(highs) >= 2 and len(lows) >= 2:
        hh = float(highs.iloc[-1]) > float(highs.iloc[-2])
        hl = float(lows.iloc[-1]) > float(lows.iloc[-2])
        return bool(hh and hl)
    return False


def _daily_recent_high(df: pd.DataFrame) -> float:
    lv = pd.to_numeric(df.get("ict_sw_level", pd.Series()), errors="coerce")
    sw = df.get("ict_sw_highlow", pd.Series()).fillna(0)
    highs = lv.where(sw > 0).dropna()
    if len(highs) > 0:
        return float(highs.iloc[-1])
    close = pd.to_numeric(df.get("close", pd.Series()), errors="coerce").dropna()
    return float(close.iloc[-1]) if len(close) > 0 else float("nan")


def _plot_main_chart(df: pd.DataFrame, ts_code: str, show_ict: bool, show_harmonics: bool, cycle_levels: Optional[List[SingleTFLevel]] = None):
    """
    主图绘制（保留既有力度显示），并在任意周期上叠加同一套“全周期综合关键位”。
    防回归说明：当前 UI 中关键位力度（S=total_strength）的显示已通过测试，请不要随意修改可视化格式和含义。
    """
    if "datetime" in df.columns:
        x = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        x = df.index.astype(str)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color="red",
            decreasing_line_color="green",
            increasing_fillcolor="rgba(255,0,0,0.7)",
            decreasing_fillcolor="rgba(0,128,0,0.7)",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    # 趋势分区可视化已移除
    if show_ict:
        if "ict_choch_flag" in df.columns:
            bull_idx = df.index[df["ict_choch_flag"] > 0]
            bear_idx = df.index[df["ict_choch_flag"] < 0]
            if len(bull_idx) > 0:
                fig.add_trace(go.Scatter(x=[x[i] for i in bull_idx], y=df.loc[bull_idx, "close"], mode="markers", marker=dict(size=10, symbol="triangle-up", color="red"), name="Bull CHOCH"), row=1, col=1)
            if len(bear_idx) > 0:
                fig.add_trace(go.Scatter(x=[x[i] for i in bear_idx], y=df.loc[bear_idx, "close"], mode="markers", marker=dict(size=10, symbol="triangle-down", color="green"), name="Bear CHOCH"), row=1, col=1)
        if "ict_bos_flag" in df.columns:
            bos_up_idx = df.index[df["ict_bos_flag"] > 0]
            bos_dn_idx = df.index[df["ict_bos_flag"] < 0]
            if len(bos_up_idx) > 0:
                fig.add_trace(go.Scatter(x=[x[i] for i in bos_up_idx], y=df.loc[bos_up_idx, "close"], mode="markers", marker=dict(size=10, symbol="square-open", line=dict(width=2)), name="Bull BOS"), row=1, col=1)
            if len(bos_dn_idx) > 0:
                fig.add_trace(go.Scatter(x=[x[i] for i in bos_dn_idx], y=df.loc[bos_dn_idx, "close"], mode="markers", marker=dict(size=10, symbol="square-open", line=dict(width=2)), name="Bear BOS"), row=1, col=1)
            if "ict_bos_level" in df.columns:
                recent_levels = df["ict_bos_level"].where(df["ict_bos_level"].notna()).dropna().tail(10)
                for y in recent_levels:
                    fig.add_hline(y=float(y), line_width=1, line_dash="dot", annotation_text="BOS", annotation_position="right", opacity=0.4, row=1, col=1)
        if "ict_ob_top" in df.columns and "ict_ob_flag" in df.columns:
            ob_idx = df.index[df["ict_ob_flag"].fillna(0) != 0]
            for i in ob_idx:
                y_val = df.at[i, "ict_ob_top"]
                fig.add_hline(y=y_val, line_width=2, line_color="blue", annotation_text="OB", annotation_position="top left", row=1, col=1)
        if "bull_entry_signal" in df.columns:
            bull_sig_idx = df.index[df["bull_entry_signal"] == 1]
            if len(bull_sig_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[x[i] for i in bull_sig_idx],
                        y=(df.loc[bull_sig_idx, "low"] * 0.995),
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=8),
                        name="多头信号",
                    ),
                    row=1,
                    col=1,
                )
        if "bear_entry_signal" in df.columns:
            bear_sig_idx = df.index[df["bear_entry_signal"] == 1]
            if len(bear_sig_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[x[i] for i in bear_sig_idx],
                        y=(df.loc[bear_sig_idx, "high"] * 1.005),
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=8),
                        name="空头信号",
                    ),
                    row=1,
                    col=1,
                )
    if show_harmonics and "harmonic_patterns" in df.attrs:
        patterns = df.attrs["harmonic_patterns"]
        for p in patterns:
            if not getattr(p, "formed", False):
                continue
            fig.add_trace(go.Scatter(x=[ts.strftime("%Y-%m-%d %H:%M:%S") for ts in p.x], y=p.y, mode="lines+markers", name=f"{p.family}-{p.name}", opacity=0.7), row=1, col=1)
    vol = df["volume"] if "volume" in df.columns else pd.Series([0] * len(df))
    up = df["close"] >= df["open"]
    vol_colors = ["red" if bool(u) else "green" for u in up]
    fig.add_trace(go.Bar(x=x, y=vol, name="成交量", marker_color=vol_colors), row=2, col=1)
    close = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    macd = (diff - dea) * 2.0
    macd_colors = ["red" if float(v) >= 0 else "green" for v in macd]
    fig.add_trace(go.Scatter(x=x, y=diff, name="DIFF", line=dict(color="#FF0000")), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=dea, name="DEA", line=dict(color="#0000FF")), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=macd, name="MACD", marker_color=macd_colors), row=3, col=1)
    fig.update_xaxes(type="category", row=1, col=1)
    fig.update_xaxes(type="category", row=2, col=1)
    fig.update_xaxes(type="category", row=3, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=900)
    fig.update_layout(hovermode="x unified")
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", yanchor="bottom"))
    fig.add_trace(go.Scatter(x=[x.iloc[0] if hasattr(x, "iloc") else (x[0] if len(x)>0 else "")], y=[df["close"].iloc[0] if "close" in df.columns else 0], name="周主导（粗实线）", mode="lines", line=dict(color="#1f77b4", width=3, dash="solid"), visible="legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x.iloc[0] if hasattr(x, "iloc") else (x[0] if len(x)>0 else "")], y=[df["close"].iloc[0] if "close" in df.columns else 0], name="日主导（中虚线）", mode="lines", line=dict(color="#1f77b4", width=2, dash="dash"), visible="legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x.iloc[0] if hasattr(x, "iloc") else (x[0] if len(x)>0 else "")], y=[df["close"].iloc[0] if "close" in df.columns else 0], name="60m主导（细点线）", mode="lines", line=dict(color="#1f77b4", width=1, dash="dot"), visible="legendonly"), row=1, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across")
    fig.update_yaxes(showspikes=True)
    if cycle_levels:
        y_min = float(pd.to_numeric(df.get("low", pd.Series(index=df.index)), errors="coerce").min()) if "low" in df.columns else float(pd.to_numeric(df.get("close", pd.Series(index=df.index)), errors="coerce").min())
        y_max = float(pd.to_numeric(df.get("high", pd.Series(index=df.index)), errors="coerce").max()) if "high" in df.columns else float(pd.to_numeric(df.get("close", pd.Series(index=df.index)), errors="coerce").max())
        strengths = [float(getattr(l, "strength_tf", 0.0)) for l in cycle_levels]
        levels_draw = list(cycle_levels)
        MAX_LEVELS_ON_CHART = 10
        PERCENTILE_FILTER = 30
        if strengths:
            thr = float(np.percentile(strengths, PERCENTILE_FILTER))
            levels_draw = [l for l in levels_draw if float(getattr(l, "strength_tf", 0.0)) >= thr]
            levels_draw = sorted(levels_draw, key=lambda z: float(getattr(z, "strength_tf", 0.0)), reverse=True)[:MAX_LEVELS_ON_CHART]
            s_vals = [float(getattr(l, "strength_tf", 0.0)) for l in levels_draw]
            s_min = float(min(s_vals)) if s_vals else 0.0
            s_max = float(max(s_vals)) if s_vals else 1.0
            s_span = float(max(s_max - s_min, 1e-6))
        import os
        if os.getenv("LAB_DEBUG") == "1":
            print("DEBUG composite levels on chart:", len(levels_draw), "S range:", (min(strengths) if strengths else None), "->", (max(strengths) if strengths else None))
        if levels_draw:
            in_view = []
            for lvl in levels_draw:
                y = float(lvl.price)
                if (y < y_min * 0.9) or (y > y_max * 1.1):
                    continue
                in_view.append(lvl)
            current_price = float(pd.to_numeric(df.get("close", pd.Series(index=df.index)), errors="coerce").iloc[-1])
            in_view_sorted = sorted(in_view, key=lambda z: float(getattr(z, "strength_tf", 0.0)), reverse=True)[:MAX_LEVELS_ON_CHART]
            labels_sorted = sorted(in_view_sorted, key=lambda z: abs(float(z.price) - current_price))[:5]
            for lvl in in_view_sorted:
                y = float(lvl.price)
                s_norm = (float(getattr(lvl, "strength_tf", 0.0)) - s_min) / s_span if s_span > 0 else 0.0
                base_map = {"OB": 2.2, "LIQ": 1.6, "FVG": 1.2}
                dash_map = {"OB": "solid", "LIQ": "dash", "FVG": "dot"}
                dom = str(getattr(lvl, "source", "")).upper()
                dom = dom if dom in base_map else "OB"
                base_w = float(base_map[dom])
                width = base_w + 1.0 * s_norm
                alpha = 0.3 + 0.4 * s_norm
                dash = dash_map[dom]
                fig.add_hline(y=y, line=dict(color="#1f77b4", width=width, dash=dash), opacity=alpha, row=1, col=1)
            x_text = x.iloc[-1] if hasattr(x, "iloc") else (x[-1] if len(x) > 0 else None)
            last_y = None
            for lvl in labels_sorted:
                y0 = float(lvl.price)
                y = y0
                if (last_y is not None) and (abs(y0 - last_y) < (y_max - y_min) * 0.02):
                    y = y0 + (y_max - y_min) * 0.02
                last_y = y
                if x_text is not None:
                    src = str(getattr(lvl, "source", "")).upper()
                    if src not in ("OB", "LIQ", "FVG"):
                        src = "OB"
                    fig.add_annotation(x=x_text, y=y, text=f"[{src}] {y0:.2f} | S={float(getattr(lvl,'strength_tf',0.0)):.1f}", showarrow=False, xanchor="right", xshift=-20, yanchor="middle", font=dict(size=8), opacity=0.8, row=1, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToAdd": ["toggleSpikelines"], "scrollZoom": True})


def _plot_equity_curve(equity: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(xaxis_title="时间", yaxis_title="资金", height=300)
    st.plotly_chart(fig, use_container_width=True)


def _auto_choose_swing_min(stats_map: Dict[int, object], target_width: float = 0.01, default_L: int = 5) -> int:
    if not stats_map:
        return int(default_L)
    items = sorted(stats_map.items(), key=lambda x: x[0])
    for L, s in items:
        if float(getattr(s, "min_width", 0.0)) >= float(target_width):
            return int(L)
    best_L = int(default_L)
    best_min = -1.0
    for L, s in items:
        mw = float(getattr(s, "min_width", 0.0))
        if mw > best_min:
            best_min = mw
            best_L = int(L)
    return best_L
def _to_weekly_bars_from_daily(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """
    仅在 UI 使用：将日线 bars 聚合为周线 bars。
    规则：每周第一根 open、最高 high、最低 low、最后一根 close，volume 为一周之和。
    不改 data 层和 DB。
    """
    if daily_bars is None or daily_bars.empty:
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    df = daily_bars.copy()
    if "datetime" not in df.columns:
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    g = df.set_index("datetime").resample("W-FRI")
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum() if "volume" in df.columns else None,
    }).dropna(subset=["open","high","low","close"]).reset_index()
    return out

def composite_levels_to_df(composite_levels: List[CompositeLevel]) -> pd.DataFrame:
    """
    把 CompositeLevel 列表转成 DataFrame 以展示周期贡献：
    列：price, kind, total_strength, strength_W, strength_D, strength_60m, members。
    """
    if not composite_levels:
        return pd.DataFrame(columns=["price","kind","total_strength","strength_W","strength_D","strength_60m","members"])[:0]
    rows = []
    for lvl in composite_levels:
        contrib_W = float(getattr(lvl, "strength_W", 0.0))
        contrib_D = float(getattr(lvl, "strength_D", 0.0))
        contrib_60 = float(getattr(lvl, "strength_H1", 0.0))
        dom_attr = getattr(lvl, "dominant_tf", None)
        dom = dom_attr if dom_attr in ("W", "D", "60m") else max([("W", contrib_W), ("D", contrib_D), ("60m", contrib_60)], key=lambda x: x[1])[0]
        rows.append({
            "price": float(lvl.price),
            "kind": str(lvl.kind),
            "total_strength": float(lvl.total_strength),
            "strength_W": float(contrib_W),
            "strength_D": float(contrib_D),
            "strength_H1": float(contrib_60),
            "dominant_tf": dom,
            "members": int(len(getattr(lvl, "members", []) or [])),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("price").reset_index(drop=True)

def single_levels_to_df(levels: List[SingleTFLevel]) -> pd.DataFrame:
    if not levels:
        return pd.DataFrame(columns=["price","kind","freq","strength_tf","touch_count","avg_reaction","span_bars","role_switch_count"])[:0]
    rows = [
        {
            "price": float(getattr(l, "price", np.nan)),
            "kind": str(getattr(l, "kind", "")),
            "freq": str(getattr(l, "freq", "")),
            "strength_tf": float(getattr(l, "strength_tf", 0.0)),
            "touch_count": int(getattr(l, "touch_count", 0)),
            "avg_reaction": float(getattr(l, "avg_reaction", 0.0)),
            "span_bars": int(getattr(l, "span_bars", 0)),
            "role_switch_count": int(getattr(l, "role_switch_count", 0)),
            "source": str(getattr(l, "source", "")),
        }
        for l in levels
    ]
    df = pd.DataFrame(rows)
    return df.sort_values(["strength_tf","price"], ascending=[False, True]).reset_index(drop=True)

def _cache_valid(entry: Dict[str, object]) -> bool:
    ts = entry.get("timestamp") if entry is not None else None
    if ts is None:
        return False
    now = pd.Timestamp.utcnow()
    dt = now - pd.Timestamp(ts)
    return dt <= pd.Timedelta(minutes=int(_KEYLEVEL_CACHE_TTL_MINUTES))

def get_single_tf_levels_for_ui(ts_code: str, n_bars: int) -> Tuple[List[SingleTFLevel], List[SingleTFLevel], List[SingleTFLevel]]:
    entry = _KEYLEVEL_CACHE.get(ts_code)
    if entry is not None and _cache_valid(entry):
        lw = entry.get("single_W") or []
        ld = entry.get("single_D") or []
        lh = entry.get("single_60m") or []
        return list(lw), list(ld), list(lh)
    daily_bars = _load_bars("stock", ts_code, "日线", max(240, int(n_bars)))
    h1_bars = _load_bars("stock", ts_code, "60分钟", max(480, int(n_bars) * 2))
    weekly_bars = _to_weekly_bars_from_daily(daily_bars)
    lw = detect_tf_levels_ict_ob_liq(weekly_bars, "W", swing_length=5)
    ld = detect_tf_levels_ict_ob_liq(daily_bars, "D", swing_length=5)
    lh = detect_single_tf_levels(h1_bars, "60m", 3)
    _KEYLEVEL_CACHE[ts_code] = {
        "timestamp": pd.Timestamp.utcnow(),
        "single_W": lw,
        "single_D": ld,
        "single_60m": lh,
        "bars_daily_len": len(daily_bars),
        "bars_h1_len": len(h1_bars),
    }
    return lw, ld, lh

def get_composite_levels_for_ui(ts_code: str, n_bars: int, ref_end_time: Optional[pd.Timestamp] = None) -> List[CompositeLevel]:
    """
    UI 层统一入口：对当前股票统一计算一次全周期综合关键位（W/D/60m），供所有周期复用。
    代码级复现：设置环境变量 LAB_SELF_TEST=1 后运行 python -m ui.asset_lab_app 打印数量与表格样例。
    UI 手动验证：选择股票后依次切换 周线/日线/60/30/15/5，观察同一套横线与 S 值保持一致；表格数据不随周期改变。
    测试标的建议：000001.SZ、600519.SH。
    """
    import os
    entry = _KEYLEVEL_CACHE.get(ts_code)
    if entry is not None and _cache_valid(entry) and (entry.get("composite") is not None):
        return list(entry.get("composite") or [])
    daily_bars = _load_bars("stock", ts_code, "日线", max(240, int(n_bars)))
    h1_bars = _load_bars("stock", ts_code, "60分钟", max(480, int(n_bars) * 2))
    weekly_bars = _to_weekly_bars_from_daily(daily_bars)
    comp_levels = build_composite_levels_for_symbol(weekly_bars=weekly_bars, daily_bars=daily_bars, h1_bars=h1_bars)
    _KEYLEVEL_CACHE[ts_code] = {
        "timestamp": pd.Timestamp.utcnow(),
        "single_W": detect_tf_levels_ict_ob_liq(weekly_bars, "W", swing_length=5),
        "single_D": detect_tf_levels_ict_ob_liq(daily_bars, "D", swing_length=5),
        "single_60m": detect_single_tf_levels(h1_bars, "60m", 3),
        "composite": comp_levels,
        "bars_daily_len": len(daily_bars),
        "bars_h1_len": len(h1_bars),
    }
    if os.getenv("LAB_DEBUG") == "1":
        hval = hash(tuple(round(float(l.price), 3) for l in comp_levels)) if comp_levels else 0
        print("DEBUG composite_levels hash:", ts_code, int(n_bars), hval)
    return comp_levels

def get_cycle_levels_for_ui(asset_type: str, ts_code: str, freq_label: str, n_bars: int) -> List[SingleTFLevel]:
    key = f"{ts_code}:{freq_label}"
    entry = _KEYLEVEL_CACHE.get(key)
    if entry is not None and _cache_valid(entry):
        return list(entry.get("cycle_levels", []) or [])
    if freq_label == "周线":
        df_day = _load_bars(asset_type, ts_code, "日线", max(500, n_bars))
        df = _to_weekly_bars_from_daily(df_day)
        lv = detect_tf_levels_ict_ob_liq(df, "W", swing_length=5, include_fvg=False)
    elif freq_label == "日线":
        df = _load_bars(asset_type, ts_code, "日线", max(500, n_bars))
        lv = detect_tf_levels_ict_ob_liq(df, "D", swing_length=5, include_fvg=False)
    else:
        fq_map = {"60分钟": "60m", "30分钟": "30m", "15分钟": "15m", "5分钟": "5m"}
        df = _load_bars(asset_type, ts_code, freq_label, max(800, n_bars))
        lv = detect_tf_levels_ict_ob_liq(df, "60m", swing_length=5, include_fvg=True)
    lv_m = merge_levels_within_tf(lv, pct=0.03)
    _KEYLEVEL_CACHE[key] = {"timestamp": pd.Timestamp.utcnow(), "cycle_levels": lv_m}
    return lv_m

def main():
    st.set_page_config(page_title="单品种 ICT + 谐波 实验室", layout="wide")
    st.title("单品种 ICT + 谐波 实验室")
    asset_label = st.sidebar.selectbox("品种类型", options=list(ASSET_LABEL_TO_CODE.keys()))
    asset_type = ASSET_LABEL_TO_CODE[asset_label]
    freq_label = st.sidebar.selectbox("周期", options=SUPPORTED_FREQS)
    bar_count = st.sidebar.slider("K 线数量（最近 N 根）", min_value=100, max_value=1000, value=240, step=50)
    if asset_type != "stock":
        st.sidebar.info("当前仅实现股票数据，期货/国债为占位")
        return
    ts_codes, code_to_name, code_to_abbr = _load_stock_universe()
    if not ts_codes:
        st.error("stock_basic 为空")
        return
    ts = st.sidebar.selectbox(
        "选择股票",
        options=ts_codes,
        index=None,
        placeholder="输入名称或拼音首字母",
        format_func=lambda x: (f"{code_to_abbr.get(x, '')} {code_to_name.get(x, x)}").strip(),
        key="ts_code_select",
    )
    if ts is None:
        st.info("请选择股票")
        return
    show_ict = st.sidebar.checkbox("叠加 ICT 结构", value=True)
    show_harm = st.sidebar.checkbox("叠加谐波形态", value=True)
    run_rr3_bt = st.sidebar.checkbox("运行 ICT R:R≥3 策略回测", value=False)
    with st.spinner("拉取K线..."):
        if freq_label == "周线":
            df_day = _load_bars(asset_type, ts, "日线", max(500, bar_count))
            df = _to_weekly_bars_from_daily(df_day)
        else:
            df = _load_bars(asset_type, ts, freq_label, bar_count)
    
    L_list = [3, 4, 5, 6, 8, 10, 12]
    stats_map = ob_swing_tuner.evaluate_swing_lengths(df, L_list)
    best_L = (
        ob_swing_tuner.auto_choose_swing_length_min_based(stats_map, target_width=0.01, default_L=5)
        if hasattr(ob_swing_tuner, "auto_choose_swing_length_min_based")
        else _auto_choose_swing_min(stats_map, target_width=0.01, default_L=5)
    )
    swing_default = int(best_L)
    swing_length = st.sidebar.slider("ICT 摆动长度 (swing_length)", min_value=1, max_value=50, value=swing_default, step=1)
    cfg_strategy = ICTConfig(swing_length=int(best_L) if best_L is not None else swing_length)
    if show_ict:
        with st.spinner("计算ICT结构..."):
            df = compute_ict_structures(df, cfg_strategy)
            df = attach_entry_signals(df)
            # 趋势分区计算已移除
    if show_harm:
        interval_map = {
            "日线": "1D",
            "60分钟": "60m",
            "30分钟": "30m",
            "15分钟": "15m",
            "5分钟": "5m",
        }
        interval = interval_map.get(freq_label, "1D")
        with st.spinner("检测谐波形态..."):
            pats = detect_harmonic_patterns(df, ts, interval)
            df.attrs["harmonic_patterns"] = pats
    cycle_levels = get_cycle_levels_for_ui(asset_type, ts, freq_label, bar_count)
    _plot_main_chart(df, ts, show_ict, show_harm, cycle_levels=cycle_levels)
    show_lv_debug = st.checkbox("显示关键位调试信息（周/日/60m）", value=True)
    if show_lv_debug:
        st.subheader("关键位调试信息（当前周期）")
        levels_draw = list(cycle_levels or [])
        strengths = [float(getattr(l, "strength_tf", 0.0)) for l in levels_draw]
        MAX_LEVELS_ON_CHART = 10
        PERCENTILE_FILTER = 30
        if strengths:
            thr = float(np.percentile(strengths, PERCENTILE_FILTER))
            levels_draw = [l for l in levels_draw if float(getattr(l, "strength_tf", 0.0)) >= thr]
            levels_draw = sorted(levels_draw, key=lambda z: float(getattr(z, "strength_tf", 0.0)), reverse=True)[:MAX_LEVELS_ON_CHART]
        in_view = []
        if len(df) > 0:
            y_min = float(pd.to_numeric(df.get("low", pd.Series(index=df.index)), errors="coerce").min()) if "low" in df.columns else float(pd.to_numeric(df.get("close", pd.Series(index=df.index)), errors="coerce").min())
            y_max = float(pd.to_numeric(df.get("high", pd.Series(index=df.index)), errors="coerce").max()) if "high" in df.columns else float(pd.to_numeric(df.get("close", pd.Series(index=df.index)), errors="coerce").max())
            for lvl in levels_draw:
                y = float(getattr(lvl, "price", 0.0))
                if (y >= y_min * 0.9) and (y <= y_max * 1.1):
                    in_view.append(lvl)
        st.caption(f"当前图窗范围内关键位（与主图一致）：{len(in_view)} 条")
        st.dataframe(single_levels_to_df(in_view), use_container_width=True)
    

    if run_rr3_bt:
        ds = get_data_source(asset_type)
        daily_bars = ds.get_daily_bars(ts, count=max(240, bar_count))
        keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in daily_bars.columns]
        daily_bars = daily_bars[keep].copy().reset_index(drop=True)
        cfg_mtf = IctMtfConfig(
            swing_length_daily=20,
            swing_length_exec=int(best_L) if best_L is not None else int(swing_length),
            risk_per_trade_pct=0.01,
        )
        with st.spinner("运行 ICT 多周期回测..."):
            res = run_ict_mtf_backtest(daily_bars, df, cfg_mtf)
        st.subheader("ICT 多周期回测资金曲线")
        x = res.equity.index
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=x, y=res.equity.values, mode="lines", name="Equity"))
        fig_eq.update_layout(xaxis_title="时间", yaxis_title="资金", height=300)
        st.plotly_chart(fig_eq, use_container_width=True)
        show_debug = st.checkbox("显示调试明细（当前周期）", value=False)
        if show_debug:
            debug_df = res.debug.copy()
            view_mode = st.selectbox(
                "查看模式",
                ["全部bar", "只看有入场机会的bar", "只看实际开仓的bar", "只看平仓bar"],
                index=0,
            )
            if view_mode == "只看有入场机会的bar":
                debug_df = debug_df[(debug_df["can_open_long"]) | (debug_df["can_open_short"])]
            elif view_mode == "只看实际开仓的bar":
                debug_df = debug_df[(debug_df["opened_long"]) | (debug_df["opened_short"])]
            elif view_mode == "只看平仓bar":
                debug_df = debug_df[debug_df["closed_pos"]]
            st.write(f"共 {len(debug_df)} 行")
            st.dataframe(debug_df.tail(300))
        if res.trades:
            rows = [
                {
                    "entry_dt": t.entry_time,
                    "exit_dt": t.exit_time,
                    "entry_idx": None,
                    "exit_idx": None,
                    "entry_price": float(t.entry_price),
                    "exit_price": float(t.exit_price),
                    "stop_price": float(t.stop_price),
                    "target_price": float(t.target_price),
                    "qty": int(t.qty),
                    "rr": float(t.rr),
                    "pnl": float(t.pnl_after_fee),
                    "side": t.side,
                }
                for t in res.trades
            ]
            st.subheader("回测交易历史")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            df.attrs["bt_trades"] = rows

    


if __name__ == "__main__":
    import os
    if os.getenv("LAB_SELF_TEST") == "1":
        ts_code = os.getenv("LAB_TEST_TS", "000001.SZ")
        comp_levels = get_composite_levels_for_ui(ts_code, 240)
        print(f"Total composite levels: {len(comp_levels)}")
        df_comp = composite_levels_to_df(comp_levels)
        print(df_comp.head().to_string(index=False))
        df_day_dbg = _load_bars("stock", ts_code, "日线", 500)
        df_h1_dbg = _load_bars("stock", ts_code, "60分钟", 1000)
        df_w_dbg = _to_weekly_bars_from_daily(df_day_dbg)
        lv_w_cnt = len(detect_single_tf_levels(df_w_dbg, "W", 3))
        lv_d_cnt = len(detect_single_tf_levels(df_day_dbg, "D", 3))
        lv_h_cnt = len(detect_single_tf_levels(df_h1_dbg, "60m", 3))
        print("Single TF counts W/D/60m:", lv_w_cnt, lv_d_cnt, lv_h_cnt)
    else:
        main()
