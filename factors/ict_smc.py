from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from smartmoneyconcepts import smc


@dataclass
class ICTConfig:
    swing_length: int = 20
    fvg_join_consecutive: bool = True
    liquidity_range_percent: float = 0.01
    bos_close_break: bool = True
    ob_close_mitigation: bool = False


def _prepare_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    req = ["open", "high", "low", "close"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"缺少字段:{miss}")
    ohlc = df.copy()
    if "volume" not in ohlc.columns:
        ohlc["volume"] = 0.0
    ohlc = ohlc[["open", "high", "low", "close", "volume"]].copy()
    ohlc.columns = [c.lower() for c in ohlc.columns]
    return ohlc


def compute_ict_structures(bars: pd.DataFrame, config: Optional[ICTConfig] = None) -> pd.DataFrame:
    cfg = config or ICTConfig()
    if "datetime" in bars.columns:
        df = bars.sort_values("datetime").reset_index(drop=True).copy()
    else:
        df = bars.sort_index().reset_index(drop=True).copy()
    ohlc = _prepare_ohlc(df)
    fvg = smc.fvg(ohlc, join_consecutive=cfg.fvg_join_consecutive)
    swings = smc.swing_highs_lows(ohlc, swing_length=cfg.swing_length)
    bos = smc.bos_choch(ohlc, swings, close_break=cfg.bos_close_break)
    ob = smc.ob(ohlc, swings, close_mitigation=cfg.ob_close_mitigation)
    liq = smc.liquidity(ohlc, swings, range_percent=cfg.liquidity_range_percent)
    out = df.copy()
    if fvg is not None and not fvg.empty:
        fvg = fvg.reset_index(drop=True)
        if "FVG" in fvg.columns:
            out["ict_fvg_flag"] = fvg["FVG"]
        if "Top" in fvg.columns:
            out["ict_fvg_top"] = fvg["Top"]
        if "Bottom" in fvg.columns:
            out["ict_fvg_bottom"] = fvg["Bottom"]
        if "MitigatedIndex" in fvg.columns:
            out["ict_fvg_mitigated_index"] = fvg["MitigatedIndex"]
    if swings is not None and not swings.empty:
        swings = swings.reset_index(drop=True)
        if "HighLow" in swings.columns:
            out["ict_sw_highlow"] = swings["HighLow"]
        if "Level" in swings.columns:
            out["ict_sw_level"] = swings["Level"]
    if bos is not None and not bos.empty:
        bos = bos.reset_index(drop=True)
        if "BOS" in bos.columns:
            out["ict_bos_flag"] = bos["BOS"]
        if "CHOCH" in bos.columns:
            out["ict_choch_flag"] = bos["CHOCH"]
        if "Level" in bos.columns:
            out["ict_bos_level"] = bos["Level"]
        if "BrokenIndex" in bos.columns:
            out["ict_bos_broken_index"] = bos["BrokenIndex"]
    if ob is not None and not ob.empty:
        ob = ob.reset_index(drop=True)
        if "OB" in ob.columns:
            out["ict_ob_flag"] = ob["OB"]
        if "Top" in ob.columns:
            out["ict_ob_top"] = ob["Top"]
        if "Bottom" in ob.columns:
            out["ict_ob_bottom"] = ob["Bottom"]
        if "OBVolume" in ob.columns:
            out["ict_ob_volume"] = ob["OBVolume"]
        if "Percentage" in ob.columns:
            out["ict_ob_strength"] = ob["Percentage"]
    if liq is not None and not liq.empty:
        liq = liq.reset_index(drop=True)
        if "Liquidity" in liq.columns:
            out["ict_liq_flag"] = liq["Liquidity"]
        if "Level" in liq.columns:
            out["ict_liq_level"] = liq["Level"]
        if "End" in liq.columns:
            out["ict_liq_end_index"] = liq["End"]
        if "Swept" in liq.columns:
            out["ict_liq_swept_index"] = liq["Swept"]
    return out


if __name__ == "__main__":
    d = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=50, freq="D"),
        "open": pd.Series(range(50)).astype(float) + 10,
        "high": pd.Series(range(50)).astype(float) + 11,
        "low": pd.Series(range(50)).astype(float) + 9,
        "close": pd.Series(range(50)).astype(float) + 10.5,
    })
    r = compute_ict_structures(d)
    print(r.shape)