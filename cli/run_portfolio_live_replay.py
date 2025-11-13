from __future__ import annotations

import argparse
from pathlib import Path

from portfolio.example_config import EXAMPLE_PORTFOLIO_CONFIG
from portfolio.runner import PortfolioLiveReplayRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run portfolio live replay.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="portfolio_live_orders",
        help="每个实例订单日志导出目录（CSV）",
    )
    args = parser.parse_args()

    cfg = EXAMPLE_PORTFOLIO_CONFIG
    print(f"[cli] Start portfolio live replay (dry_run): {cfg.name}")
    runner = PortfolioLiveReplayRunner(cfg)
    all_orders = runner.run_replay()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in all_orders.items():
        fname = key.replace(":", "_") + ".csv"
        out_path = out_dir / fname
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[cli] Orders for {key} saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()

