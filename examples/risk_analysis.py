#!/usr/bin/env python3
"""
示例: 风控测试
验证风控模块在极端行情下的表现
"""

import sys
sys.path.insert(0, "..")

from fst.data import DataFetcher
from fst.strategies.strategies import MACDStrategy
from fst.backtest import BacktestEngine
from fst.risk import RiskManager, PortfolioRiskAnalyzer
from fst.config import BacktestConfig, RiskConfig


def main():
    SYMBOL = "600036"
    config = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        slippage=0.001,
    )
    risk_config = RiskConfig(
        max_position_pct=0.20,
        max_total_position_pct=0.80,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.15,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
    )

    # 获取数据
    fetcher = DataFetcher(source="akshare", cache_dir="./data_cache")
    data = fetcher.get_stock_daily(SYMBOL, "20200101", "20251231", adjust="qfq")

    if data.empty:
        print("❌ 数据获取失败")
        return

    # 回测
    strategy = MACDStrategy()
    engine = BacktestEngine(config)
    result = engine.run_single(data, strategy, symbol=SYMBOL)

    # 风险报告
    risk_mgr = RiskManager(risk_config)
    analyzer = PortfolioRiskAnalyzer()
    report = analyzer.risk_report(result.equity_curve)

    print("=" * 50)
    print("         风险分析报告")
    print("=" * 50)
    print(f"  年化波动率:    {report['volatility_annual']:.2%}")
    print(f"  VaR(95%):      {report['var_95']:.2%}")
    print(f"  CVaR(95%):     {report['cvar_95']:.2%}")
    print(f"  最大回撤:      {report['max_drawdown']:.2%}")
    print(f"  偏度:          {report['skewness']:.3f}")
    print(f"  峰度:          {report['kurtosis']:.3f}")
    print("=" * 50)

    # 仓位建议
    if "atr" in data.columns:
        atr = data["atr"].iloc[-1]
    else:
        atr = (data["high"] - data["low"]).rolling(14).mean().iloc[-1]
    price = data["close"].iloc[-1]
    suggested = risk_mgr.get_position_sizing(
        price=price, atr=atr, risk_per_trade=0.01,
        equity=config.initial_capital
    )
    print(f"\n💡 海龟仓位建议: 当前价格 ¥{price:.2f}, ATR={atr:.2f}")
    print(f"   建议仓位: {suggested} 股 (投入 ¥{suggested * price:,.2f})")


if __name__ == "__main__":
    main()
