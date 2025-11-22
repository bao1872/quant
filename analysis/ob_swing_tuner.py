from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Tuple

import numpy as np
import pandas as pd

from factors.ict_smc import ICTConfig, compute_ict_structures


@dataclass
class SwingStats:
    swing_length: int
    n_ob: int
    min_width: float
    median_width: float
    mean_width: float


def _compute_ob_width_stats(
    bars: pd.DataFrame,
    swing_length: int,
    ob_flag_col: str = "ict_ob_flag",
    ob_top_col: str = "ict_ob_top",
    ob_bottom_col: str = "ict_ob_bottom",
) -> Optional[SwingStats]:
    cfg = ICTConfig(swing_length=swing_length)
    df_ict = compute_ict_structures(bars, cfg)
    if ob_flag_col not in df_ict.columns:
        return None
    mask_ob = df_ict[ob_flag_col].fillna(0) != 0
    if not mask_ob.any():
        return None
    ob_rows = df_ict.loc[mask_ob, [ob_top_col, ob_bottom_col]].dropna()
    if ob_rows.empty:
        return None
    width = (ob_rows[ob_top_col] - ob_rows[ob_bottom_col]).abs()
    mid_price = ((ob_rows[ob_top_col] + ob_rows[ob_bottom_col]) / 2.0).abs()
    mid_price = mid_price.replace(0, np.nan)
    rel_width = (width / mid_price).dropna()
    if rel_width.empty:
        return None
    n_ob = int(len(rel_width))
    min_w = float(rel_width.min())
    median_w = float(rel_width.median())
    mean_w = float(rel_width.mean())
    return SwingStats(
        swing_length=swing_length,
        n_ob=n_ob,
        min_width=min_w,
        median_width=median_w,
        mean_width=mean_w,
    )


def evaluate_swing_lengths(
    bars: pd.DataFrame,
    swing_lengths: Sequence[int],
) -> Dict[int, SwingStats]:
    stats_map: Dict[int, SwingStats] = {}
    for L in swing_lengths:
        stats = _compute_ob_width_stats(bars, int(L))
        if stats is not None:
            stats_map[int(L)] = stats
    return stats_map


def auto_choose_swing_length_min_based(
    stats_map: Dict[int, SwingStats],
    target_width: float = 0.01,
    default_L: Optional[int] = None,
) -> Optional[int]:
    if not stats_map:
        return default_L
    items: list[Tuple[int, SwingStats]] = sorted(stats_map.items(), key=lambda x: x[0])
    for L, s in items:
        if float(s.min_width) >= float(target_width):
            return int(L)
    best_L = None
    best_min = -1.0
    for L, s in items:
        if float(s.min_width) > float(best_min):
            best_min = float(s.min_width)
            best_L = int(L)
    return best_L


if __name__ == "__main__":
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    base = np.linspace(10, 12, n)
    df = pd.DataFrame({
        "datetime": idx,
        "open": base,
        "high": base + 0.5,
        "low": base - 0.5,
        "close": base + np.sin(np.linspace(0, 6.28, n)) * 0.2,
    })
    L_list = [3, 4, 5, 6, 8, 10, 12]
    stats = evaluate_swing_lengths(df, L_list)
    best = auto_choose_swing_length_min_based(stats, target_width=0.01, default_L=5)
    print(best, len(stats))