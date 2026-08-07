"""
回测引擎
支持单股票和多股票回测，提供完整的绩效分析
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger

from ..strategies import BaseStrategy, Signal, SignalType
from ..indicators import compute_all_indicators
from ..config import BacktestConfig


@dataclass
class Trade:
    """单笔交易记录"""
    date: pd.Timestamp
    symbol: str
    side: str              # "BUY" / "SELL"
    price: float
    shares: int
    amount: float
    commission: float
    tax: float
    slippage: float
    reason: str = ""


@dataclass
class Position:
    """持仓"""
    symbol: str
    shares: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0

    # 资金曲线
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 绩效指标
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    total_trades: int = 0
    avg_holding_days: float = 0.0

    # 交易记录
    trades: List[Trade] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "              回测报告 / Backtest Report",
            "=" * 60,
            f"  回测期间:      {self.start_date} ~ {self.end_date}",
            f"  初始资金:      ¥{self.initial_capital:,.2f}",
            f"  最终资金:      ¥{self.equity_curve['equity'].iloc[-1]:,.2f}" if not self.equity_curve.empty else "",
            "",
            "--- 收益指标 ---",
            f"  总收益率:      {self.total_return:.2%}",
            f"  年化收益率:    {self.annual_return:.2%}",
            f"  最大回撤:      {self.max_drawdown:.2%}",
            "",
            "--- 风险指标 ---",
            f"  夏普比率:      {self.sharpe_ratio:.3f}",
            f"  索提诺比率:    {self.sortino_ratio:.3f}",
            f"  卡玛比率:      {self.calmar_ratio:.3f}",
            "",
            "--- 交易统计 ---",
            f"  总交易次数:    {self.total_trades}",
            f"  胜率:          {self.win_rate:.2%}",
            f"  盈亏比:        {self.profit_loss_ratio:.2f}",
            f"  平均持仓天数:  {self.avg_holding_days:.1f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    回测引擎

    支持:
    - 单股票回测
    - 多股票回测
    - 佣金/印花税/滑点
    - 绩效分析 (收益率、夏普比率、最大回撤等)
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run_single(self, data: pd.DataFrame, strategy: BaseStrategy,
                   symbol: str = "unknown") -> BacktestResult:
        """
        单股票回测

        参数:
            data: 日线数据 (date, open, high, low, close, volume)
            strategy: 策略实例
            symbol: 股票代码
        """
        logger.info(f"开始回测: {strategy.name} | {symbol}")

        # 计算指标
        df = compute_all_indicators(data)
        df = df.reset_index(drop=True)

        # 生成信号
        strategy.reset()
        signals = strategy.generate_signals(df, symbol=symbol)

        # 模拟交易
        result = self._simulate(df, signals, symbol)
        result.start_date = str(df["date"].iloc[0].date())
        result.end_date = str(df["date"].iloc[-1].date())
        result.initial_capital = self.config.initial_capital

        # 计算绩效指标
        self._compute_metrics(result)

        logger.info(f"回测完成: 总收益 {result.total_return:.2%}, 最大回撤 {result.max_drawdown:.2%}")
        return result

    def run_multi(self, data_dict: Dict[str, pd.DataFrame],
                  strategy: BaseStrategy) -> BacktestResult:
        """
        多股票回测

        参数:
            data_dict: {symbol: DataFrame} 字典
            strategy: 策略实例
        """
        logger.info(f"开始多股票回测: {strategy.name} | {len(data_dict)}只股票")

        all_signals = []
        for symbol, df in data_dict.items():
            ind_df = compute_all_indicators(df).reset_index(drop=True)
            sigs = strategy.generate_signals(ind_df, symbol=symbol)
            all_signals.extend(sigs)

        # 按日期排序
        all_signals.sort(key=lambda s: s.date)

        # 合并所有日期
        all_dates = set()
        for df in data_dict.values():
            all_dates.update(df["date"].tolist())
        all_dates = sorted(all_dates)

        combined = pd.DataFrame({"date": all_dates})
        for symbol, df in data_dict.items():
            merged = combined.merge(df[["date", "close"]], on="date", how="left")
            combined[symbol] = merged["close"].ffill()

        result = self._simulate_multi(combined, all_signals, data_dict)
        result.initial_capital = self.config.initial_capital
        self._compute_metrics(result)
        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _simulate(self, df: pd.DataFrame, signals: List[Signal],
                  symbol: str) -> BacktestResult:
        """模拟交易"""
        capital = self.config.initial_capital
        shares = 0
        trades = []
        equity_records = []

        signal_map = {}
        for s in signals:
            signal_map.setdefault(s.date, []).append(s)

        for i, row in df.iterrows():
            date = row["date"]
            price = row["close"]

            # 检查止损/止盈
            if shares > 0:
                avg_cost = capital / (shares + capital / price) if shares > 0 else 0
                # 简化: 使用买入均价跟踪
                pass

            # 执行信号
            if date in signal_map:
                for sig in signal_map[date]:
                    if sig.signal_type == SignalType.BUY and shares == 0:
                        # 买入 (考虑滑点)
                        exec_price = price * (1 + self.config.slippage)
                        max_shares = int(capital * 0.95 / exec_price / 100) * 100  # A股100股整数
                        if max_shares >= 100:
                            amount = max_shares * exec_price
                            commission = max(amount * self.config.commission_rate, 5.0)  # 最低5元
                            slippage_cost = max_shares * price * self.config.slippage

                            if amount + commission <= capital:
                                shares = max_shares
                                capital -= (amount + commission)
                                trades.append(Trade(
                                    date=date, symbol=symbol, side="BUY",
                                    price=exec_price, shares=max_shares,
                                    amount=amount, commission=commission,
                                    tax=0, slippage=slippage_cost,
                                    reason=sig.reason,
                                ))

                    elif sig.signal_type == SignalType.SELL and shares > 0:
                        # 卖出
                        exec_price = price * (1 - self.config.slippage)
                        amount = shares * exec_price
                        commission = max(amount * self.config.commission_rate, 5.0)
                        tax = amount * self.config.stamp_tax_rate
                        slippage_cost = shares * price * self.config.slippage

                        capital += (amount - commission - tax)
                        trades.append(Trade(
                            date=date, symbol=symbol, side="SELL",
                            price=exec_price, shares=shares,
                            amount=amount, commission=commission,
                            tax=tax, slippage=slippage_cost,
                            reason=sig.reason,
                        ))
                        shares = 0

            # 记录权益
            market_value = shares * price
            equity = capital + market_value
            equity_records.append({
                "date": date,
                "equity": equity,
                "capital": capital,
                "market_value": market_value,
                "shares": shares,
            })

        result = BacktestResult(
            equity_curve=pd.DataFrame(equity_records),
            trades=trades,
        )
        return result

    def _simulate_multi(self, dates_df: pd.DataFrame,
                        signals: List[Signal],
                        data_dict: Dict[str, pd.DataFrame]) -> BacktestResult:
        """多股票模拟"""
        capital = self.config.initial_capital
        positions: Dict[str, Position] = {}
        trades = []
        equity_records = []

        signal_map = {}
        for s in signals:
            signal_map.setdefault(s.date, []).append(s)

        for i, row in dates_df.iterrows():
            date = row["date"]

            # 更新持仓市值
            for sym, pos in positions.items():
                if sym in dates_df.columns:
                    price = row[sym]
                    if not pd.isna(price) and price > 0:
                        pos.market_value = pos.shares * price
                        pos.unrealized_pnl = pos.market_value - pos.shares * pos.avg_cost

            # 执行信号
            if date in signal_map:
                for sig in signal_map[date]:
                    sym = sig.symbol
                    price = dates_df.loc[i, sym] if sym in dates_df.columns else None
                    if price is None or pd.isna(price) or price <= 0:
                        continue

                    if sig.signal_type == SignalType.BUY and sym not in positions:
                        exec_price = price * (1 + self.config.slippage)
                        alloc = capital * 0.25  # 最多用25%资金
                        max_shares = int(alloc / exec_price / 100) * 100
                        if max_shares >= 100:
                            amount = max_shares * exec_price
                            commission = max(amount * self.config.commission_rate, 5.0)
                            if amount + commission <= capital:
                                capital -= (amount + commission)
                                positions[sym] = Position(
                                    symbol=sym, shares=max_shares,
                                    avg_cost=exec_price, market_value=max_shares * price,
                                )
                                trades.append(Trade(
                                    date=date, symbol=sym, side="BUY",
                                    price=exec_price, shares=max_shares,
                                    amount=amount, commission=commission,
                                    tax=0, slippage=max_shares * price * self.config.slippage,
                                    reason=sig.reason,
                                ))

                    elif sig.signal_type == SignalType.SELL and sym in positions:
                        pos = positions[sym]
                        exec_price = price * (1 - self.config.slippage)
                        amount = pos.shares * exec_price
                        commission = max(amount * self.config.commission_rate, 5.0)
                        tax = amount * self.config.stamp_tax_rate
                        capital += (amount - commission - tax)
                        trades.append(Trade(
                            date=date, symbol=sym, side="SELL",
                            price=exec_price, shares=pos.shares,
                            amount=amount, commission=commission,
                            tax=tax, slippage=pos.shares * price * self.config.slippage,
                            reason=sig.reason,
                        ))
                        del positions[sym]

            market_value = sum(p.market_value for p in positions.values())
            equity = capital + market_value
            equity_records.append({
                "date": date, "equity": equity,
                "capital": capital, "market_value": market_value,
            })

        return BacktestResult(
            equity_curve=pd.DataFrame(equity_records),
            trades=trades,
        )

    def _compute_metrics(self, result: BacktestResult):
        """计算绩效指标"""
        if result.equity_curve.empty:
            return

        ec = result.equity_curve
        returns = ec["equity"].pct_change().dropna()
        initial = self.config.initial_capital

        # 总收益
        final_equity = ec["equity"].iloc[-1]
        result.total_return = (final_equity - initial) / initial

        # 年化收益
        days = (ec["date"].iloc[-1] - ec["date"].iloc[0]).days
        if days > 0:
            result.annual_return = (1 + result.total_return) ** (365 / days) - 1

        # 最大回撤
        cummax = ec["equity"].cummax()
        drawdown = (ec["equity"] - cummax) / cummax
        result.max_drawdown = abs(drawdown.min())

        # 夏普比率 (无风险利率假设3%)
        if len(returns) > 1 and returns.std() > 0:
            rf_daily = 0.03 / 252
            result.sharpe_ratio = (returns.mean() - rf_daily) / returns.std() * np.sqrt(252)

        # 索提诺比率
        downside = returns[returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            rf_daily = 0.03 / 252
            result.sortino_ratio = (returns.mean() - rf_daily) / downside.std() * np.sqrt(252)

        # 卡玛比率
        if result.max_drawdown > 0:
            result.calmar_ratio = result.annual_return / result.max_drawdown

        # 交易统计
        result.total_trades = len(result.trades)
        if result.total_trades > 0:
            buy_sell_pairs = []
            open_trade = None
            for t in result.trades:
                if t.side == "BUY":
                    open_trade = t
                elif t.side == "SELL" and open_trade:
                    profit = (t.price - open_trade.price) * t.shares - t.commission - t.tax - open_trade.commission
                    days_held = (t.date - open_trade.date).days
                    buy_sell_pairs.append({
                        "profit": profit, "days": days_held,
                        "return": (t.price - open_trade.price) / open_trade.price,
                    })
                    open_trade = None

            if buy_sell_pairs:
                wins = [p for p in buy_sell_pairs if p["profit"] > 0]
                losses = [p for p in buy_sell_pairs if p["profit"] <= 0]
                result.win_rate = len(wins) / len(buy_sell_pairs)
                avg_win = np.mean([p["profit"] for p in wins]) if wins else 0
                avg_loss = abs(np.mean([p["profit"] for p in losses])) if losses else 1
                result.profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                result.avg_holding_days = np.mean([p["days"] for p in buy_sell_pairs])
