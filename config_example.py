from dataclasses import dataclass
from typing import Optional

DATABASE_URL = "postgresql+psycopg2://user:pass@localhost:5432/quant"
TICK_BASE_DIR = "data/tick"

MARKET_SZ = 0
MARKET_SH = 1

STOCK_POOL_LIMIT: Optional[int] = None
TICK_COUNT_LIMIT: int = 20000

@dataclass
class Settings:
    stock_pool_limit: Optional[int] = STOCK_POOL_LIMIT
    tick_count_limit: int = TICK_COUNT_LIMIT
    tick_base_dir: str = TICK_BASE_DIR

