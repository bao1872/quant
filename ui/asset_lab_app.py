from __future__ import annotations

import os
import sys
from typing import List, Dict, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data.repository import get_all_stock_basics
from data.source_factory import get_data_source
from factors.ict_smc import compute_ict_structures, ICTConfig
from factors.harmonic_patterns import detect_harmonic_patterns
from backtest.bar_backtest import run_backtest_one_unit
from strategy.ict_rr3_simple import generate_rr3_long_signals, backtest_fullsize_rr3
import analysis.ob_swing_tuner as ob_swing_tuner
from strategy.ict_mtf_lab import IctMtfConfig, run_ict_mtf_backtest, attach_entry_signals, compute_daily_trend_with_fallback, TrendState


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


def _plot_main_chart(df: pd.DataFrame, ts_code: str, show_ict: bool, show_harmonics: bool):
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
    if "trend_state" in df.columns:
        ts = df["trend_state"].tolist()
        idxs = df.index.tolist()
        start = 0
        for i in range(1, len(ts) + 1):
            if i == len(ts) or ts[i] != ts[start]:
                state = ts[start]
                x0 = x[start]
                x1 = x[i - 1]
                color_map = {
                    int(TrendState.STRONG_LONG): "rgba(255,0,0,0.08)",
                    int(TrendState.WEAK_LONG): "rgba(255,165,0,0.08)",
                    int(TrendState.WEAK_SHORT): "rgba(135,206,250,0.08)",
                    int(TrendState.STRONG_SHORT): "rgba(0,128,0,0.08)",
                    int(TrendState.FLAT): None,
                }
                color = color_map.get(int(state), None)
                if color is not None:
                    fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.25, layer="below", line_width=0, row=1, col=1)
                start = i
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
            ob_idx = list(ob_idx)[-5:]
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
    # 叠加买卖位置标记
    if "bt_trades" in df.attrs:
        rows = df.attrs["bt_trades"]
        entry_x: list = []
        entry_y: list = []
        exit_x: list = []
        exit_y: list = []
        for r in rows:
            ei = r.get("entry_idx")
            xi = r.get("exit_idx")
            if ei is not None and ei in df.index:
                entry_x.append(x[ei])
                entry_y.append(float(df.loc[ei, "close"]))
            else:
                ed = r.get("entry_dt")
                if ed is not None:
                    ed_str = ed.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ed, "strftime") else str(ed)
                    entry_x.append(ed_str)
                    if "datetime" in df.columns:
                        idx_match = df.index[df["datetime"].astype("datetime64[ns]") == ed]
                        if len(idx_match) > 0:
                            entry_y.append(float(df.loc[idx_match[0], "close"]))
                        else:
                            entry_y.append(float(df["close"].iloc[0]))
                    else:
                        entry_y.append(float(df["close"].iloc[0]))
            if xi is not None and xi in df.index:
                exit_x.append(x[xi])
                exit_y.append(float(df.loc[xi, "close"]))
            else:
                xd = r.get("exit_dt")
                if xd is not None:
                    xd_str = xd.strftime("%Y-%m-%d %H:%M:%S") if hasattr(xd, "strftime") else str(xd)
                    exit_x.append(xd_str)
                    if "datetime" in df.columns:
                        idx_match2 = df.index[df["datetime"].astype("datetime64[ns]") == xd]
                        if len(idx_match2) > 0:
                            exit_y.append(float(df.loc[idx_match2[0], "close"]))
                        else:
                            exit_y.append(float(df["close"].iloc[-1]))
                    else:
                        exit_y.append(float(df["close"].iloc[-1]))
        if entry_x and entry_y:
            fig.add_trace(
                go.Scatter(
                    x=entry_x,
                    y=entry_y,
                    mode="markers",
                    marker=dict(symbol="circle-open", size=10, line=dict(width=2, color="blue")),
                    name="入场",
                ),
                row=1,
                col=1,
            )
        if exit_x and exit_y:
            fig.add_trace(
                go.Scatter(
                    x=exit_x,
                    y=exit_y,
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="black"),
                    name="出场",
                ),
                row=1,
                col=1,
            )
    st.plotly_chart(fig, use_container_width=True)


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
    ts_codes, code_to_name = _load_stock_universe()
    if not ts_codes:
        st.error("stock_basic 为空")
        return
    ts = st.sidebar.selectbox("选择股票", options=ts_codes, format_func=lambda x: code_to_name.get(x, x))
    show_ict = st.sidebar.checkbox("叠加 ICT 结构", value=True)
    show_harm = st.sidebar.checkbox("叠加谐波形态", value=True)
    run_rr3_bt = st.sidebar.checkbox("运行 ICT R:R≥3 策略回测", value=True)
    with st.spinner("拉取K线..."):
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
            if freq_label == "日线":
                dt_df = compute_daily_trend_with_fallback(df, swing_len=20)
                df = df.merge(dt_df[["datetime", "trend_state"]], on="datetime", how="left")
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
    _plot_main_chart(df, ts, show_ict, show_harm)
    

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

    # 主图（含买卖标记）
    st.subheader("主图（含买卖标记）")
    _plot_main_chart(df, ts, show_ict, show_harm)


if __name__ == "__main__":
    main()