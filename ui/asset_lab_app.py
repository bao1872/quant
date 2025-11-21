from __future__ import annotations

import os
import sys
from typing import List, Dict, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.repository import get_all_stock_basics
from data.source_factory import get_data_source
from factors.ict_smc import compute_ict_structures, ICTConfig
from factors.harmonic_patterns import detect_harmonic_patterns
from backtest.bar_backtest import run_backtest_one_unit


ASSET_LABEL_TO_CODE: Dict[str, str] = {
    "股票 (A股)": "stock",
    "股指期货 (占位)": "index_future",
    "国债期货 (占位)": "gov_bond",
}

SUPPORTED_FREQS = ["日线", "60分钟", "30分钟", "15分钟", "5分钟"]


def _load_stock_universe() -> Tuple[List[str], Dict[str, str]]:
    basics = get_all_stock_basics()
    if not basics:
        return [], {}
    ts_codes: List[str] = [b.ts_code for b in basics]
    names: List[str] = [getattr(b, "name", b.ts_code) for b in basics]
    code_to_name = dict(zip(ts_codes, names))
    return ts_codes, code_to_name


def _load_bars(asset_type: str, ts_code: str, freq_label: str, bar_count: int) -> pd.DataFrame:
    ds = get_data_source(asset_type)
    if freq_label == "日线":
        bars = ds.get_daily_bars(ts_code, count=bar_count)
    else:
        freq_map = {
            "60分钟": "60m",
            "30分钟": "30m",
            "15分钟": "15m",
            "5分钟": "5m",
        }
        freq = freq_map.get(freq_label)
        bars = ds.get_minute_bars(ts_code, freq=freq, count=bar_count)
    if bars is None or bars.empty:
        raise RuntimeError(f"{ts_code} 在 {freq_label} 周期没有数据")
    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in bars.columns]
    bars = bars[keep].copy()
    if "datetime" in bars.columns:
        bars = bars.sort_values("datetime")
    return bars.reset_index(drop=True)


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


def _plot_main_chart(df: pd.DataFrame, ts_code: str, show_ict: bool, show_harmonics: bool):
    x = df["datetime"] if "datetime" in df.columns else df.index
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K线"))
    if show_ict:
        if "ict_choch_flag" in df.columns:
            bull_idx = df.index[df["ict_choch_flag"] > 0]
            bear_idx = df.index[df["ict_choch_flag"] < 0]
            if len(bull_idx) > 0:
                fig.add_trace(go.Scatter(x=x.iloc[bull_idx] if hasattr(x, "iloc") else [x[i] for i in bull_idx], y=df.loc[bull_idx, "close"], mode="markers", marker=dict(size=8, symbol="triangle-up"), name="Bull CHOCH"))
            if len(bear_idx) > 0:
                fig.add_trace(go.Scatter(x=x.iloc[bear_idx] if hasattr(x, "iloc") else [x[i] for i in bear_idx], y=df.loc[bear_idx, "close"], mode="markers", marker=dict(size=8, symbol="triangle-down"), name="Bear CHOCH"))
        if "ict_ob_top" in df.columns and "ict_ob_flag" in df.columns:
            ob_idx = df.index[df["ict_ob_flag"].fillna(0) != 0]
            ob_idx = list(ob_idx)[-5:]
            for i in ob_idx:
                y = df.at[i, "ict_ob_top"]
                fig.add_hline(y=y, line_width=1, annotation_text="OB", annotation_position="top left")
    if show_harmonics and "harmonic_patterns" in df.attrs:
        pats = df.attrs["harmonic_patterns"]
        for p in pats:
            if not p.formed:
                continue
            fig.add_trace(go.Scatter(x=p.x, y=p.y, mode="lines+markers", name=f"{p.family}-{p.name}"))
    fig.update_layout(xaxis_title="时间", yaxis_title="价格", xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)


def _plot_equity_curve(equity: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(xaxis_title="时间", yaxis_title="资金", height=300)
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="单品种 ICT + 谐波 实验室", layout="wide")
    st.title("单品种 ICT + 谐波 实验室")
    ts_codes, code_to_name = _load_stock_universe()
    if not ts_codes:
        st.error("stock_basic 为空")
        return
    asset_label = st.sidebar.selectbox("品种类型", options=list(ASSET_LABEL_TO_CODE.keys()))
    asset_type = ASSET_LABEL_TO_CODE[asset_label]
    freq_label = st.sidebar.selectbox("周期", options=SUPPORTED_FREQS)
    bar_count = st.sidebar.slider("K 线数量（最近 N 根）", min_value=100, max_value=1000, value=240, step=50)
    ts = st.sidebar.selectbox("选择股票", options=ts_codes, format_func=lambda x: code_to_name.get(x, x))
    show_ict = st.sidebar.checkbox("叠加 ICT 结构", value=True)
    show_harm = st.sidebar.checkbox("叠加谐波形态", value=False)
    run_demo_bt = st.sidebar.checkbox("运行示例回测", value=True)
    cfg = ICTConfig()
    with st.spinner("拉取K线..."):
        df = _load_bars(asset_type, ts, freq_label, bar_count)
    if show_ict:
        with st.spinner("计算ICT结构..."):
            df = compute_ict_structures(df, cfg)
    if show_harm:
        interval = "1D" if freq_label == "日线" else freq_label
        with st.spinner("检测谐波形态..."):
            pats = detect_harmonic_patterns(df, ts, interval)
            df.attrs["harmonic_patterns"] = pats
    _plot_main_chart(df, ts, show_ict, show_harm)
    if run_demo_bt:
        sig = _build_demo_signal(df)
        res = run_backtest_one_unit(df, sig)
        st.subheader("示例回测资金曲线")
        _plot_equity_curve(res.equity_curve)


if __name__ == "__main__":
    print("请使用: streamlit run ui/asset_lab_app.py")