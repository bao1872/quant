from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio.example_config import EXAMPLE_PORTFOLIO_CONFIG
from portfolio.runner import PortfolioBacktestRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run portfolio backtest.")
    parser.add_argument(
        "--output",
        type=str,
        default="portfolio_backtest_result.csv",
        help="回测结果导出路径（CSV）",
    )
    args = parser.parse_args()

    cfg = EXAMPLE_PORTFOLIO_CONFIG
    print(f"[cli] Start portfolio backtest: {cfg.name}")
    runner = PortfolioBacktestRunner(cfg)
    result = runner.run()
    df = result.to_equity_dataframe()
    print("[cli] Backtest summary:")
    print(df)
    out_path = Path(args.output)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cli] Result saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

