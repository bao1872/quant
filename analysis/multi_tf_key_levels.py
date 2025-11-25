from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List, Sequence

import numpy as np
import pandas as pd


LevelKind = Literal["support", "resistance"]
LevelFreq = Literal["W", "D", "60m"]


@dataclass
class SingleTFLevel:
    price: float
    kind: LevelKind
    freq: LevelFreq
    strength_tf: float
    touch_count: int
    avg_reaction: float
    span_bars: int
    role_switch_count: int


@dataclass
class CompositeLevel:
    price: float
    kind: LevelKind
    total_strength: float
    strength_W: float
    strength_D: float
    strength_H1: float
    dominant_tf: LevelFreq
    members: List[SingleTFLevel]


def detect_single_tf_levels(bars: pd.DataFrame, freq: LevelFreq, swing_lookback: int = 3) -> List[SingleTFLevel]:
    if bars is None or bars.empty:
        return []
    df = bars.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    high = df.get("high", pd.Series(index=df.index, dtype=float)).astype(float)
    low = df.get("low", pd.Series(index=df.index, dtype=float)).astype(float)
    close = df.get("close", pd.Series(index=df.index, dtype=float)).astype(float)
    n = len(df)
    l = int(max(1, swing_lookback))
    prev_max = high.shift(1).rolling(l).max()
    next_max = high[::-1].shift(1).rolling(l).max()[::-1]
    is_high = (high > prev_max) & (high > next_max)
    prev_min = low.shift(1).rolling(l).min()
    next_min = low[::-1].shift(1).rolling(l).min()[::-1]
    is_low = (low < prev_min) & (low < next_min)
    idx_high = list(df.index[is_high.fillna(False)])
    idx_low = list(df.index[is_low.fillna(False)])
    touch_pct = 0.002
    react_w = 10
    fwd_max = close[::-1].rolling(react_w).max()[::-1]
    fwd_min = close[::-1].rolling(react_w).min()[::-1]
    rows: list[dict] = []
    for i in idx_low:
        p = float(low.at[i])
        lo = p * (1.0 - touch_pct)
        hi = p * (1.0 + touch_pct)
        m1 = (low <= hi) & (high >= lo)
        n_touch = int(m1.sum())
        fm = float(fwd_max.at[i]) if i in fwd_max.index else p
        avg_react = float(max(0.0, (fm - p) / max(p, 1e-9)))
        t_idx = df.index[m1]
        if len(t_idx) > 0:
            span = int(len(df.loc[t_idx.min():t_idx.max()]))
        else:
            span = 0
        above = close > p
        below = close < p
        role_flag = int(1 if (above.any() and below.any()) else 0)
        rows.append({
            "price": p,
            "kind": "support",
            "freq": freq,
            "touch_count": int(n_touch),
            "avg_reaction": float(avg_react),
            "span_bars": int(span),
            "role_switch_count": int(role_flag),
        })
    for i in idx_high:
        p = float(high.at[i])
        lo = p * (1.0 - touch_pct)
        hi = p * (1.0 + touch_pct)
        m1 = (low <= hi) & (high >= lo)
        n_touch = int(m1.sum())
        fm = float(fwd_min.at[i]) if i in fwd_min.index else p
        avg_react = float(max(0.0, (p - fm) / max(p, 1e-9)))
        t_idx = df.index[m1]
        if len(t_idx) > 0:
            span = int(len(df.loc[t_idx.min():t_idx.max()]))
        else:
            span = 0
        above = close > p
        below = close < p
        role_flag = int(1 if (above.any() and below.any()) else 0)
        rows.append({
            "price": p,
            "kind": "resistance",
            "freq": freq,
            "touch_count": int(n_touch),
            "avg_reaction": float(avg_react),
            "span_bars": int(span),
            "role_switch_count": int(role_flag),
        })
    if not rows:
        return []
    r = pd.DataFrame(rows)
    for col in ["touch_count", "avg_reaction", "span_bars"]:
        x = r[col].astype(float)
        mn = float(x.min())
        mx = float(x.max())
        if mx > mn:
            r[col + "_n"] = (x - mn) / (mx - mn)
        else:
            r[col + "_n"] = 0.0
    w_touch = 0.4
    w_react = 0.4
    w_span = 0.15
    w_role = 0.05
    r["strength_tf"] = (
        w_touch * r["touch_count_n"].astype(float) +
        w_react * r["avg_reaction_n"].astype(float) +
        w_span * r["span_bars_n"].astype(float) +
        w_role * r["role_switch_count"].astype(float)
    )
    out = [
        SingleTFLevel(
            price=float(row["price"]),
            kind=row["kind"],
            freq=row["freq"],
            strength_tf=float(row["strength_tf"]),
            touch_count=int(row["touch_count"]),
            avg_reaction=float(row["avg_reaction"]),
            span_bars=int(row["span_bars"]),
            role_switch_count=int(row["role_switch_count"]),
        )
        for _, row in r.sort_values("price").iterrows()
    ]
    return out


def compute_single_tf_levels(
    bars: pd.DataFrame,
    freq: LevelFreq,
    *,
    price_candidates: Sequence[float],
    touch_eps_pct: float = 0.003,
    reaction_window: int = 10,
) -> List[SingleTFLevel]:
    if bars is None or bars.empty or (price_candidates is None) or (len(price_candidates) == 0):
        return []
    df = bars.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    high = df.get("high", pd.Series(index=df.index, dtype=float)).astype(float)
    low = df.get("low", pd.Series(index=df.index, dtype=float)).astype(float)
    close = df.get("close", pd.Series(index=df.index, dtype=float)).astype(float)
    fwd_max = close[::-1].rolling(int(reaction_window)).max()[::-1]
    fwd_min = close[::-1].rolling(int(reaction_window)).min()[::-1]
    rows: list[dict] = []
    for p in price_candidates:
        p = float(p)
        lo = p * (1.0 - float(touch_eps_pct))
        hi = p * (1.0 + float(touch_eps_pct))
        m1 = (low <= hi) & (high >= lo)
        n_touch = int(m1.sum())
        if n_touch == 0:
            continue
        idx = df.index[m1]
        first_i = idx.min()
        last_i = idx.max()
        fm = float(fwd_max.at[first_i]) if first_i in fwd_max.index else p
        fn = float(fwd_min.at[first_i]) if first_i in fwd_min.index else p
        avg_react_up = float(max(0.0, (fm - p) / max(p, 1e-9)))
        avg_react_dn = float(max(0.0, (p - fn) / max(p, 1e-9)))
        avg_react = float(max(avg_react_up, avg_react_dn))
        span = int(len(df.loc[first_i:last_i]))
        above = close > p
        below = close < p
        role_sw = int(1 if (above.any() and below.any()) else 0)
        kind = "support" if avg_react_up >= avg_react_dn else "resistance"
        rows.append({
            "price": p,
            "kind": kind,
            "freq": freq,
            "touch_count": int(n_touch),
            "avg_reaction": float(avg_react),
            "span_bars": int(span),
            "role_switch_count": int(role_sw),
        })
    if not rows:
        return []
    r = pd.DataFrame(rows)
    for col in ["touch_count", "avg_reaction", "span_bars"]:
        x = r[col].astype(float)
        mn = float(x.min())
        mx = float(x.max())
        r[col + "_n"] = (x - mn) / (mx - mn) if mx > mn else 0.0
    w_touch = 0.4
    w_react = 0.4
    w_span = 0.15
    w_role = 0.05
    r["strength_tf"] = (
        w_touch * r["touch_count_n"].astype(float) +
        w_react * r["avg_reaction_n"].astype(float) +
        w_span * r["span_bars_n"].astype(float) +
        w_role * r["role_switch_count"].astype(float)
    )
    out = [
        SingleTFLevel(
            price=float(row["price"]),
            kind=row["kind"],
            freq=row["freq"],
            strength_tf=float(row["strength_tf"]),
            touch_count=int(row["touch_count"]),
            avg_reaction=float(row["avg_reaction"]),
            span_bars=int(row["span_bars"]),
            role_switch_count=int(row["role_switch_count"]),
        )
        for _, row in r.sort_values("price").iterrows()
    ]
    return out


def cluster_levels_with_hdbscan(
    levels: Sequence[SingleTFLevel],
    eps_pct: float = 0.005,
    min_cluster_size: int = 3,
    w_w: float = 4.0,
    w_d: float = 3.0,
    w_h: float = 2.0,
    allow_noise_as_singleton: bool = True,
) -> List[CompositeLevel]:
    if levels is None or len(levels) == 0:
        return []
    import hdbscan
    prices = np.array([float(x.price) for x in levels], dtype=float)
    p_mean = float(np.mean(prices)) if len(prices) > 0 else 1.0
    X = (prices / (p_mean if p_mean != 0 else 1.0)).reshape(-1, 1)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        metric="euclidean",
        cluster_selection_epsilon=float(eps_pct),
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    W_W = float(w_w)
    W_D = float(w_d)
    W_H = float(w_h)
    lab = np.array(labels, dtype=int)
    uniq = sorted(list(set(lab.tolist())))
    out: List[CompositeLevel] = []
    for lb in uniq:
        if lb < 0 and not allow_noise_as_singleton:
            continue
        idx = np.where(lab == lb)[0]
        if len(idx) == 0:
            continue
        members = [levels[i] for i in idx]
        st_W = float(np.sum([m.strength_tf for m in members if m.freq == "W"]))
        st_D = float(np.sum([m.strength_tf for m in members if m.freq == "D"]))
        st_H = float(np.sum([m.strength_tf for m in members if m.freq == "60m"]))
        wts_tf = np.array([m.strength_tf for m in members], dtype=float)
        if wts_tf.sum() == 0:
            wts_tf = np.ones_like(wts_tf)
        price_cluster = float(np.sum(np.array([m.price for m in members], dtype=float) * wts_tf) / np.sum(wts_tf))
        total_strength = float(W_W * st_W + W_D * st_D + W_H * st_H)
        dominant_tf = "W"
        if (st_D >= st_W) and (st_D >= st_H):
            dominant_tf = "D"
        elif (st_H >= st_W) and (st_H >= st_D):
            dominant_tf = "60m"
        sc_sup = float(np.sum([m.strength_tf for m in members if m.kind == "support"]))
        sc_res = float(np.sum([m.strength_tf for m in members if m.kind == "resistance"]))
        kind = "support" if sc_sup >= sc_res else "resistance"
        out.append(CompositeLevel(price=price_cluster, kind=kind, total_strength=total_strength, strength_W=st_W, strength_D=st_D, strength_H1=st_H, dominant_tf=dominant_tf, members=members))
    out = sorted(out, key=lambda x: x.price)
    return out


def build_composite_levels_for_symbol(
    weekly_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    h1_bars: pd.DataFrame,
    swing_lookback: int = 3,
    eps_pct: float = 0.005,
    min_cluster_size: int = 3,
    w_w: float = 4.0,
    w_d: float = 3.0,
    w_h: float = 2.0,
) -> List[CompositeLevel]:
    lw = detect_single_tf_levels(weekly_bars, "W", swing_lookback)
    ld = detect_single_tf_levels(daily_bars, "D", swing_lookback)
    lh = detect_single_tf_levels(h1_bars, "60m", swing_lookback)
    all_levels = lw + ld + lh
    comp = cluster_levels_with_hdbscan(
        all_levels,
        eps_pct=eps_pct,
        min_cluster_size=min_cluster_size,
        w_w=w_w,
        w_d=w_d,
        w_h=w_h,
    )
    return comp


if __name__ == "__main__":
    import importlib.util
    has_hdb = importlib.util.find_spec("hdbscan") is not None
    try_import_ui = importlib.util.find_spec("ui.asset_lab_app") is not None
    if try_import_ui:
        from ui.asset_lab_app import _load_bars as load_bars_for_ui, _to_weekly_bars_from_daily as to_weekly_bars_from_daily
        ts_code = "000001.SZ"
        daily = load_bars_for_ui("stock", ts_code, "日线", 500)
        h1 = load_bars_for_ui("stock", ts_code, "60分钟", 1000)
        weekly = to_weekly_bars_from_daily(daily)
        lw = detect_single_tf_levels(weekly, "W", 3)
        ld = detect_single_tf_levels(daily, "D", 3)
        lh = detect_single_tf_levels(h1, "60m", 3)
        print("single W/D/H:", len(lw), len(ld), len(lh))
        comps = build_composite_levels_for_symbol(weekly, daily, h1)
        print("composite count:", len(comps))
        for c in comps[:10]:
            print(f"price={c.price:.2f}, kind={c.kind}, S={c.total_strength:.2f}, W/D/H=({c.strength_W:.1f}, {c.strength_D:.1f}, {c.strength_H1:.1f}), dom={c.dominant_tf}, members={len(c.members)}")
        if not has_hdb:
            print("hdbscan not installed; clustering test ran in fallback with empty comps if unavailable")
    else:
        print("ui.asset_lab_app not found; skipping integrated self-test")