"""
风控模块
包括仓位管理、止损止盈、风险监控
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from loguru import logger

from ..strategies import Signal, SignalType
from ..config import RiskConfig


@dataclass
class RiskCheck:
    """风控检查结果"""
    passed: bool
    reason: str = ""
    adjusted_signal: Optional[Signal] = None


class RiskManager:
    """风控管理器"""

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self._daily_pnl = 0.0
        self._peak_equity = 0.0
        self._current_equity = 0.0

    def check_signal(self, signal: Signal, positions: Dict, equity: float,
                     cash: float) -> RiskCheck:
        """
        对交易信号进行风控检查

        参数:
            signal: 待执行信号
            positions: 当前持仓 {symbol: Position}
            equity: 总权益
            cash: 现金
        """
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        if signal.signal_type == SignalType.BUY:
            return self._check_buy(signal, positions, equity, cash)
        elif signal.signal_type == SignalType.SELL:
            return self._check_sell(signal, positions, equity, cash)

        return RiskCheck(passed=True)

    def _check_buy(self, signal: Signal, positions: Dict,
                    equity: float, cash: float) -> RiskCheck:
        """检查买入信号"""
        # 1. 检查总仓位
        total_position_value = sum(p.market_value for p in positions.values())
        total_position_pct = total_position_value / equity if equity > 0 else 0

        if total_position_pct >= self.config.max_total_position_pct:
            return RiskCheck(
                passed=False,
                reason=f"总仓位已达{total_position_pct:.1%}, 超过上限{self.config.max_total_position_pct:.1%}",
            )

        # 2. 检查单票仓位
        existing = positions.get(signal.symbol)
        if existing:
            new_position_pct = (existing.market_value + signal.price * signal.shares) / equity
        else:
            new_position_pct = signal.target_pct

        if new_position_pct > self.config.max_position_pct:
            # 调整信号
            adjusted_shares = int(equity * self.config.max_position_pct / signal.price / 100) * 100
            if adjusted_shares >= 100:
                adjusted = Signal(
                    date=signal.date, symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    strength=signal.strength,
                    price=signal.price,
                    target_pct=self.config.max_position_pct,
                    reason=f"风控调整: {signal.reason}",
                )
                return RiskCheck(
                    passed=True,
                    reason=f"单票仓位超限, 调整为{adjusted_shares}股",
                    adjusted_signal=adjusted,
                )
            return RiskCheck(
                passed=False,
                reason=f"单票仓位{new_position_pct:.1%}超过上限{self.config.max_position_pct:.1%}",
            )

        # 3. 检查现金是否充足
        if cash < signal.price * 100 * 1.001:  # 至少买100股
            return RiskCheck(
                passed=False,
                reason=f"现金不足: ¥{cash:,.2f} < 买入所需最低金额",
            )

        # 4. 检查每日亏损
        daily_loss_pct = self._daily_pnl / equity if equity > 0 else 0
        if daily_loss_pct < -self.config.max_daily_loss_pct:
            return RiskCheck(
                passed=False,
                reason=f"今日亏损已达{daily_loss_pct:.2%}, 超过限额{self.config.max_daily_loss_pct:.2%}, 暂停交易",
            )

        # 5. 检查最大回撤
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.config.max_drawdown_pct:
                return RiskCheck(
                    passed=False,
                    reason=f"回撤已达{drawdown:.2%}, 超过限额{self.config.max_drawdown_pct:.2%}",
                )

        return RiskCheck(passed=True)

    def _check_sell(self, signal: Signal, positions: Dict,
                     equity: float, cash: float) -> RiskCheck:
        """检查卖出信号 (卖出一般不需要严格风控)"""
        if signal.symbol not in positions:
            return RiskCheck(
                passed=False,
                reason=f"未持有{signal.symbol}, 无法卖出",
            )
        return RiskCheck(passed=True)

    def check_stop_loss(self, symbol: str, current_price: float,
                         avg_cost: float) -> Optional[Signal]:
        """检查是否触发止损"""
        if avg_cost <= 0:
            return None
        loss_pct = (current_price - avg_cost) / avg_cost
        if loss_pct <= -self.config.stop_loss_pct:
            logger.warning(f"⚠️ {symbol} 触发止损! 亏损 {loss_pct:.2%}")
            return Signal(
                date=pd.Timestamp.now(),
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=1.0,
                price=current_price,
                target_pct=0.0,
                reason=f"止损触发 ({loss_pct:.2%})",
            )
        return None

    def check_take_profit(self, symbol: str, current_price: float,
                           avg_cost: float) -> Optional[Signal]:
        """检查是否触发止盈"""
        if avg_cost <= 0:
            return None
        profit_pct = (current_price - avg_cost) / avg_cost
        if profit_pct >= self.config.take_profit_pct:
            logger.info(f"🎉 {symbol} 触发止盈! 盈利 {profit_pct:.2%}")
            return Signal(
                date=pd.Timestamp.now(),
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=0.8,
                price=current_price,
                target_pct=0.0,
                reason=f"止盈触发 ({profit_pct:.2%})",
            )
        return None

    def update_daily_pnl(self, pnl: float):
        """更新当日盈亏"""
        self._daily_pnl += pnl

    def reset_daily(self):
        """每日重置"""
        self._daily_pnl = 0.0

    def get_position_sizing(self, price: float, atr: float,
                             risk_per_trade: float = 0.01,
                             equity: float = 1_000_000) -> int:
        """
        基于ATR的仓位计算 (海龟交易法)

        参数:
            price: 当前价格
            atr: ATR值
            risk_per_trade: 每笔交易风险比例
            equity: 总权益
        返回:
            建议购买股数 (100的整数倍)
        """
        if atr <= 0 or price <= 0:
            return 0

        risk_amount = equity * risk_per_trade
        unit = risk_amount / atr  # 一个ATR对应的资金
        shares = int(unit / price / 100) * 100
        return max(shares, 0)


class PortfolioRiskAnalyzer:
    """组合风险分析"""

    @staticmethod
    def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """计算VaR (在险价值)"""
        if returns.empty:
            return 0.0
        return abs(np.percentile(returns, (1 - confidence) * 100))

    @staticmethod
    def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """计算CVaR (条件VaR)"""
        var = PortfolioRiskAnalyzer.compute_var(returns, confidence)
        tail = returns[returns <= -var]
        return abs(tail.mean()) if not tail.empty else var

    @staticmethod
    def compute_correlation_matrix(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算股票间相关性矩阵"""
        returns = {}
        for sym, df in data_dict.items():
            if "close" in df.columns:
                returns[sym] = df.set_index("date")["close"].pct_change()
        return pd.DataFrame(returns).corr()

    @staticmethod
    def risk_report(equity_curve: pd.DataFrame) -> dict:
        """生成风险报告"""
        returns = equity_curve["equity"].pct_change().dropna()

        return {
            "volatility_annual": returns.std() * np.sqrt(252),
            "var_95": PortfolioRiskAnalyzer.compute_var(returns, 0.95),
            "cvar_95": PortfolioRiskAnalyzer.compute_cvar(returns, 0.95),
            "max_drawdown": (equity_curve["equity"] / equity_curve["equity"].cummax() - 1).min(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }
