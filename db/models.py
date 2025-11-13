class StockBasic:
    def __init__(self, ts_code: str):
        self.ts_code = ts_code


class StockDaily:
    ts_code = None
    trade_date = None
    def __init__(
        self,
        ts_code: str,
        trade_date,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        amount: float,
    ):
        self.ts_code = ts_code
        self.trade_date = trade_date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.amount = amount


class StockMinute:
    ts_code = None
    trade_date = None
    def __init__(
        self,
        ts_code: str,
        trade_date,
        minute: str,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        amount: float,
    ):
        self.ts_code = ts_code
        self.trade_date = trade_date
        self.minute = minute
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.amount = amount


class TickFileIndex:
    ts_code = None
    trade_date = None
    def __init__(
        self,
        ts_code: str,
        trade_date,
        market: str,
        file_path: str,
        record_cnt: int,
        time_start: str,
        time_end: str,
        checksum,
    ):
        self.ts_code = ts_code
        self.trade_date = trade_date
        self.market = market
        self.file_path = file_path
        self.record_cnt = record_cnt
        self.time_start = time_start
        self.time_end = time_end
        self.checksum = checksum
