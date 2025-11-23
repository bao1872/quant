"""
TDX 服务器选择与探活模块。

核心思路：
- 维护候选 TDX 行情服务器 IP 列表；
- 首次遍历探活，选择连接成功且响应最快的作为“最佳服务器”；
- 将选出的 (ip, port) 缓存到本地 JSON，后续优先使用并做一次探活；
- 若缓存失效或不可用，重新扫描候选列表并更新缓存；
- 如果所有候选都无法连接，显式抛出错误。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

import json
from pytdx.hq import TdxHq_API


CACHE_PATH = Path(__file__).resolve().parent / "tdx_best_server.json"


def _load_cache() -> Optional[Tuple[str, int]]:
    if not CACHE_PATH.exists():
        return None
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    ip = data.get("ip")
    port = int(data.get("port")) if data.get("port") is not None else None
    if ip and port:
        return (str(ip), int(port))
    return None


def _save_cache(ip: str, port: int) -> None:
    CACHE_PATH.write_text(json.dumps({"ip": ip, "port": int(port)}, ensure_ascii=False), encoding="utf-8")


def candidate_servers() -> List[Tuple[str, int]]:
    return [
        ("119.147.164.60", 7709),
        ("180.153.18.171", 7709),
        ("114.80.149.19", 7709),
        ("115.238.90.165", 7709),
        ("123.125.108.23", 7709),
        ("218.108.98.244", 7709),
        # 可按需扩展更多官方推荐 IP
    ]


def probe_tdx_server(ip: str, port: int, ts_code: str = "000001.SZ") -> Optional[float]:
    api = TdxHq_API()
    ok = api.connect(ip, port)
    if not ok:
        api.disconnect()
        return None
    # 仅测试连接与一次请求耗时：拉取 1 条日线
    start = perf_counter()
    code, exch = ts_code.split(".")
    market = 0 if exch.upper() == "SZ" else 1
    data = api.get_security_bars(category=9, market=market, code=code, start=0, count=1)
    elapsed = perf_counter() - start
    api.disconnect()
    if not data:
        return None
    return float(elapsed)


def get_best_tdx_server(force_refresh: bool = False) -> Tuple[str, int]:
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            ip, port = cached
            delay = probe_tdx_server(ip, port)
            if delay is not None:
                return ip, port
    candidates = candidate_servers()
    best: Optional[Tuple[str, int, float]] = None
    for ip, port in candidates:
        delay = probe_tdx_server(ip, port)
        if delay is None:
            continue
        if best is None or delay < best[2]:
            best = (ip, port, delay)
    if best is None:
        raise RuntimeError("All TDX servers unavailable. Please check network/firewall (port 7709).")
    _save_cache(best[0], best[1])
    return best[0], best[1]


def select_new_best_server() -> Tuple[str, int]:
    return get_best_tdx_server(force_refresh=True)