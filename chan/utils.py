def ts_to_chan_code(ts_code: str) -> str:
    s = str(ts_code).strip().upper()
    if "." not in s:
        p = s.zfill(6)
        return ("SZ." + p) if p.startswith(("0", "3")) else ("SH." + p)
    code, exch = s.split(".")
    if exch in ("SZ", "SH"):
        return exch + "." + code.zfill(6)
    return s


def chan_to_ts_code(code: str) -> str:
    s = str(code).strip().upper()
    if "." not in s:
        return s
    exch, num = s.split(".")
    if exch in ("SZ", "SH"):
        return num.zfill(6) + "." + exch
    return s


def chan_lv_to_db_freq(lv: str) -> str:
    k = str(lv).lower()
    if k in ("day", "d"):
        return "1d"
    return k


def ui_label_to_db_freq(label: str) -> str:
    m = {"日线": "1d", "60分钟": "60m", "30分钟": "30m", "15分钟": "15m", "5分钟": "5m"}
    return m.get(label, label)