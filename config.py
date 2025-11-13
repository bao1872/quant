# config.py
"""
项目全局配置。

说明：
- 这里放的是“运行环境相关”的配置：数据库连接、数据文件目录、市场编码等。
- 后续可以逐步改成优先读取环境变量，方便部署到不同机器。

当前包含：
- DATABASE_URL: Postgres 连接串
- TICK_BASE_DIR: tick 数据文件根目录
- MARKET_SZ / MARKET_SH: pytdx 市场编码
"""

import os
from pathlib import Path
import os

# =========================
# 数据库配置
# =========================

# 直接使用你提供的连接串
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://es:es123456@159.75.69.192:5432/bz",
)

# =========================
# Tick 数据目录
# =========================

# 默认放在项目下的 data/tick 目录，可以通过环境变量覆盖
DEFAULT_TICK_BASE_DIR = Path(__file__).resolve().parent / "data" / "tick"

TICK_BASE_DIR: str = os.getenv("TICK_BASE_DIR", str(DEFAULT_TICK_BASE_DIR))


# =========================
# 市场编码（pytdx 用）
# =========================

# pytdx 中：
#   0 = 深圳市场 (SZ)
#   1 = 上海市场 (SH)
MARKET_SZ: int = 0
MARKET_SH: int = 1

# =========================
# 股票池与采样规模
# =========================

def _parse_optional_int(env_name: str, default: str) -> int | None:
    v = os.getenv(env_name, default)
    v = v.strip() if isinstance(v, str) else v
    if v == "":
        return None
    return int(v)

STOCK_POOL_LIMIT: int | None = _parse_optional_int("STOCK_POOL_LIMIT", "20")
TICK_COUNT_LIMIT: int = int(os.getenv("TICK_COUNT_LIMIT", "1000"))


if __name__ == "__main__":
    print("[config] self test")
    print("DATABASE_URL:", DATABASE_URL)
    print("TICK_BASE_DIR:", TICK_BASE_DIR)
    print("MARKET_SZ:", MARKET_SZ)
    print("MARKET_SH:", MARKET_SH)
    print("STOCK_POOL_LIMIT:", STOCK_POOL_LIMIT)
    print("TICK_COUNT_LIMIT:", TICK_COUNT_LIMIT)

    # 检查 tick 目录是否存在，如不存在可以给个提示（不强制创建）
    tick_path = Path(TICK_BASE_DIR)
    print("Tick dir exists:", tick_path.exists())
