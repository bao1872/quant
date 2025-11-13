from __future__ import annotations

from datetime import date
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from db.connection import get_session
from data.repository import get_all_stock_basics
from factors import AbuPriceLevelProvider


def load_daily_bars(ts_code: str, start: date, end: date) -> pd.DataFrame:
    with get_session() as session:
        sql = text(
            """
            select trade_date, open, high, low, close, volume, amount
            from stock_daily
            where ts_code = :ts
              and trade_date between :start and :end
            order by trade_date
            """
        )
        df = pd.read_sql(sql, session.get_bind(), params={"ts": ts_code, "start": start, "end": end})
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_price_levels(ts_code: str, trade_date: date) -> pd.DataFrame:
    plp = AbuPriceLevelProvider()
    levels = plp.get_levels(ts_code, trade_date)
    if not levels:
        return pd.DataFrame(columns=["ts_code", "trade_date", "level_price", "level_type", "direction", "strength", "source_flags"])
    rows = []
    for lv in levels:
        rows.append({"ts_code": lv.ts_code, "trade_date": lv.trade_date, "level_price": lv.level_price, "level_type": lv.level_type, "direction": lv.direction, "strength": lv.strength, "source_flags": ",".join(lv.source_flags or [])})
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def main() -> None:
    st.set_page_config(page_title="A股阿布关键位策略可视化", layout="wide")
    st.title("A股阿布价格理论关键位可视化")
    basics = get_all_stock_basics()
    if not basics:
        st.error("stock_basic 为空，请先跑数据更新任务。")
        return
    ts_codes: List[str] = [b.ts_code for b in basics]
    names: List[str] = [getattr(b, "name", b.ts_code) for b in basics]
    code_to_name = dict(zip(ts_codes, names))
    ts = st.sidebar.selectbox("选择股票", options=ts_codes, format_func=lambda x: code_to_name.get(x, x))
    today = date.today()
    start = st.sidebar.date_input("开始日期", value=today.replace(year=today.year - 1))
    end = st.sidebar.date_input("结束日期", value=today)
    level_date = st.sidebar.date_input("关键位日期（通常选最近一个交易日）", value=end)
    st.sidebar.write("---")
    st.sidebar.write("提示：")
    st.sidebar.write("1. 先用数据更新任务拉取日线和关键位。")
    st.sidebar.write("2. 实盘/回测逻辑由后端模块完成，这里只负责可视化。")
    with st.spinner("加载日线数据..."):
        df_ohlc = load_daily_bars(ts, start, end)
    if df_ohlc.empty:
        st.warning("选定区间内无日线数据。")
        return
    with st.spinner("加载关键位数据..."):
        df_levels = load_price_levels(ts, level_date)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_ohlc["trade_date"], open=df_ohlc["open"], high=df_ohlc["high"], low=df_ohlc["low"], close=df_ohlc["close"], name="K线"))
    if not df_levels.empty:
        for direction, color in [("support", "green"), ("resistance", "red"), ("neutral", "blue")]:
            sub = df_levels[df_levels["direction"] == direction]
            for _, row in sub.iterrows():
                fig.add_hline(y=row["level_price"], line=dict(color=color, width=1, dash="dot"), annotation_text=f"{direction} | {row['level_type']} | {row['strength']}", annotation_position="right")
    fig.update_layout(xaxis_title="日期", yaxis_title="价格", xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader(f"{ts} {level_date} 关键价格列表")
    if df_levels.empty:
        st.info("该日尚未计算关键位。请先跑 AbuPriceLevelProvider.precompute(trade_date)。")
    else:
        st.dataframe(df_levels.sort_values("level_price"), use_container_width=True)


if __name__ == "__main__":
    print("请用命令运行 Streamlit 应用：\n\n    streamlit run ui/streamlit_app.py\n")

