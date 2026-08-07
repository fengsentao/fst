#!/usr/bin/env python3
"""
FST-Quant 命令行工具
用法:
  python -m fst.cli backtest --symbol 600036 --strategy dualma
  python -m fst.cli analyze --symbol 600036
  python -m fst.cli list
"""

import argparse
import sys

from fst.data import DataFetcher
from fst.config import load_config, BacktestConfig


def cmd_backtest(args):
    """运行回测"""
    from fst.strategies.strategies import (
        DualMAStrategy, MACDStrategy, RSIReversion,
        BOLLBreakout, KDJStrategy, TurtleStrategy,
    )
    from fst.backtest import BacktestEngine
    from fst.utils import plot_equity_curve

    strategy_map = {
        "dualma": lambda: DualMAStrategy(args.short_ma, args.long_ma),
        "macd": lambda: MACDStrategy(),
        "rsi": lambda: RSIReversion(),
        "boll": lambda: BOLLBreakout(),
        "kdj": lambda: KDJStrategy(),
        "turtle": lambda: TurtleStrategy(),
    }

    if args.strategy not in strategy_map:
        print(f"❌ 未知策略: {args.strategy}")
        print(f"   可用策略: {', '.join(strategy_map.keys())}")
        return

    config = BacktestConfig(
        initial_capital=args.capital,
        commission_rate=args.commission,
        stamp_tax_rate=args.stamp_tax,
        slippage=args.slippage,
    )

    print(f"📊 获取 {args.symbol} 数据...")
    fetcher = DataFetcher(source=args.data_source, cache_dir="./data_cache")
    data = fetcher.get_stock_daily(args.symbol, args.start, args.end, adjust="qfq")

    if data.empty:
        print("❌ 数据获取失败，请检查网络和代码")
        return

    strategy = strategy_map[args.strategy]()
    print(f"🚀 策略: {strategy.name}")
    print(f"📅 期间: {args.start} ~ {args.end}")
    print(f"💰 初始资金: ¥{args.capital:,.0f}")

    engine = BacktestEngine(config)
    result = engine.run_single(data, strategy, symbol=args.symbol)

    print(result.summary())

    if not args.no_plot:
        path = plot_equity_curve(result, save_path=f"{args.symbol}_{args.strategy}.png")
        print(f"\n📈 图表已保存: {path}")


def cmd_analyze(args):
    """技术分析"""
    from fst.indicators import compute_all_indicators
    from fst.utils import plot_indicators

    fetcher = DataFetcher(source=args.data_source, cache_dir="./data_cache")
    data = fetcher.get_stock_daily(args.symbol, args.start, args.end, adjust="qfq")

    if data.empty:
        print("❌ 数据获取失败")
        return

    df = compute_all_indicators(data)

    # 输出最新指标
    latest = df.iloc[-1]
    print(f"\n📊 {args.symbol} 最新指标:")
    print(f"  收盘价:  {latest['close']:.2f}")
    print(f"  MA5:     {latest.get('sma_5', 0):.2f}")
    print(f"  MA20:    {latest.get('sma_20', 0):.2f}")
    print(f"  MACD:    {latest.get('macd', 0):.4f}")
    print(f"  DIF:     {latest.get('dif', 0):.4f}")
    print(f"  DEA:     {latest.get('dea', 0):.4f}")
    print(f"  RSI:     {latest.get('rsi', 0):.2f}")
    print(f"  K:       {latest.get('k', 0):.2f}")
    print(f"  D:       {latest.get('d', 0):.2f}")
    print(f"  J:       {latest.get('j', 0):.2f}")
    print(f"  BOLL上轨: {latest.get('boll_upper', 0):.2f}")
    print(f"  BOLL下轨: {latest.get('boll_lower', 0):.2f}")

    if not args.no_plot:
        path = plot_indicators(df, save_path=f"{args.symbol}_indicators.png")
        print(f"\n📈 图表已保存: {path}")


def cmd_list(args):
    """列出可用策略"""
    print("\n📋 可用策略:")
    strategies = {
        "dualma": "双均线交叉策略 (短期/长期均线金叉死叉)",
        "macd": "MACD金叉死叉策略",
        "rsi": "RSI均值回归策略 (超买超卖)",
        "boll": "布林带突破策略",
        "kdj": "KDJ超买超卖策略",
        "turtle": "海龟交易策略 (突破N日高低点)",
    }
    for name, desc in strategies.items():
        print(f"  {name:<10} - {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="FST-Quant: A股量化交易框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-source", default="akshare",
                        choices=["akshare", "tushare"],
                        help="数据源 (默认: akshare)")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # backtest 命令
    bt_parser = subparsers.add_parser("backtest", help="运行策略回测")
    bt_parser.add_argument("--symbol", "-s", required=True, help="股票代码, 如 600036")
    bt_parser.add_argument("--strategy", "-st", default="dualma",
                           choices=["dualma", "macd", "rsi", "boll", "kdj", "turtle"])
    bt_parser.add_argument("--start", default="20200101", help="开始日期 YYYYMMDD")
    bt_parser.add_argument("--end", default="20251231", help="结束日期 YYYYMMDD")
    bt_parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    bt_parser.add_argument("--commission", type=float, default=0.0003, help="手续费率")
    bt_parser.add_argument("--stamp-tax", type=float, default=0.0005, help="印花税率")
    bt_parser.add_argument("--slippage", type=float, default=0.001, help="滑点")
    bt_parser.add_argument("--short-ma", type=int, default=5, help="短期均线周期")
    bt_parser.add_argument("--long-ma", type=int, default=20, help="长期均线周期")
    bt_parser.add_argument("--no-plot", action="store_true", help="不生成图表")

    # analyze 命令
    an_parser = subparsers.add_parser("analyze", help="技术分析")
    an_parser.add_argument("--symbol", "-s", required=True, help="股票代码")
    an_parser.add_argument("--start", default="20230101")
    an_parser.add_argument("--end", default="20251231")
    an_parser.add_argument("--no-plot", action="store_true")

    # list 命令
    subparsers.add_parser("list", help="列出可用策略")

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
