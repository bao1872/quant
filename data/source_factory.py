from __future__ import annotations

from typing import Literal

from .base_source import DataSource
from .pytdx_source import PytdxDataSource


AssetType = Literal["stock", "index_future", "gov_bond"]


def get_data_source(asset_type: AssetType) -> DataSource:
    if asset_type == "stock":
        return PytdxDataSource()
    raise NotImplementedError(f"asset_type={asset_type} 未实现")


if __name__ == "__main__":
    ds = get_data_source("stock")
    print(type(ds).__name__)