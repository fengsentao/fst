#!/usr/bin/env python3
"""
示例: 多策略对比回测
对比不同策略在同一股票上的表现
"""

import sys
sys.path.insert(0, "..")

from fst.data import DataFetcher
from fst.strategies.strategies import (
    DualMAStrategy, MACDStrategy, RSIReversion,
    BOLLBreakout, KDJStrategy, TurtleStrategy,
)
from fst.backtest import BacktestEngine
from fst.config import BacktestConfig


def main():
    SYMBOL = "600036"  # 招商银行
    START = "20200101"
    END = "20251231"

    config = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        slippage=0.001,
    )

    # 获取数据
    fetcher = DataFetcher(source="akshare", cache_dir="./data_cache")
    data = fetcher.get_stock_daily(SYMBOL, START, END, adjust="qfq")

    if data.empty:
        print("❌ 数据获取失败")
        return

    # 策略列表
    strategies = [
        DualMAStrategy(short_period=5, long_period=20),
        DualMAStrategy(short_period=10, long_period=60),
        MACDStrategy(fast=12, slow=26, signal=9),
        RSIReversion(period=14, oversold=30, overbought=70),
        BOLLBreakout(period=20, std_dev=2.0),
        KDJStrategy(n=9),
        TurtleStrategy(entry_period=20, exit_period=10),
    ]

    engine = BacktestEngine(config)

    # 对比回测
    print(f"{'策略名称':<20} {'总收益':>10} {'年化收益':>10} {'最大回撤':>10} {'夏普比率':>10} {'交易次数':>10}")
    print("-" * 80)

    results = {}
    for strategy in strategies:
        result = engine.run_single(data, strategy, symbol=SYMBOL)
        results[strategy.name] = result

        print(f"{strategy.name:<20} "
              f"{result.total_return:>10.2%} "
              f"{result.annual_return:>10.2%} "
              f"{result.max_drawdown:>10.2%} "
              f"{result.sharpe_ratio:>10.3f} "
              f"{result.total_trades:>10}")

    # 找出最佳策略
    best_name = max(results, key=lambda k: results[k].sharpe_ratio)
    print(f"\n🏆 夏普比率最佳策略: {best_name} (Sharpe={results[best_name].sharpe_ratio:.3f})")


if __name__ == "__main__":
    main()
