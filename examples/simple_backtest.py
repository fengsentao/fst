#!/usr/bin/env python3
"""
示例: 使用FST-Quant回测双均线策略
"""

import sys
sys.path.insert(0, "..")

from fst.data import DataFetcher
from fst.strategies.strategies import DualMAStrategy
from fst.backtest import BacktestEngine
from fst.config import BacktestConfig
from fst.utils import plot_equity_curve


def main():
    # 1. 配置
    config = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        slippage=0.001,
    )

    # 2. 获取数据 (以招商银行为例)
    fetcher = DataFetcher(source="akshare", cache_dir="./data_cache")
    data = fetcher.get_stock_daily(
        symbol="600036",
        start_date="20200101",
        end_date="20251231",
        adjust="qfq",
    )

    if data.empty:
        print("❌ 数据获取失败，请检查网络连接")
        return

    print(f"📊 获取数据: {len(data)} 条")
    print(f"   时间范围: {data['date'].iloc[0]} ~ {data['date'].iloc[-1]}")

    # 3. 创建策略
    strategy = DualMAStrategy(short_period=5, long_period=20)

    # 4. 回测
    engine = BacktestEngine(config)
    result = engine.run_single(data, strategy, symbol="600036")

    # 5. 输出结果
    print(result.summary())

    # 6. 绘图
    save_path = plot_equity_curve(result, save_path="dual_ma_backtest.png")
    print(f"\n📈 图表已保存: {save_path}")


if __name__ == "__main__":
    main()
