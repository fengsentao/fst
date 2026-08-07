"""
内置量化策略集合

策略列表:
  1. DualMAStrategy   - 双均线交叉策略
  2. MACDStrategy     - MACD金叉死叉策略
  3. RSIReversion     - RSI均值回归策略
  4. BOLLBreakout     - 布林带突破策略
  5. KDJStrategy      - KDJ超买超卖策略
  6. GridStrategy     - 网格交易策略
  7. TurtleStrategy   - 海龟交易策略
  8. MultiFactor      - 多因子选股策略
"""

import pandas as pd
import numpy as np
from typing import List
from . import BaseStrategy, Signal, SignalType


class DualMAStrategy(BaseStrategy):
    """
    双均线交叉策略
    短期均线上穿长期均线 -> 买入
    短期均线下穿长期均线 -> 卖出
    """

    def __init__(self, short_period: int = 5, long_period: int = 20):
        super().__init__(
            name="DualMA",
            short_period=short_period,
            long_period=long_period,
        )
        self.short_period = short_period
        self.long_period = long_period

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        signals = []
        symbol = kwargs.get("symbol", "unknown")
        df = data.copy()

        df["sma_short"] = df["close"].rolling(self.short_period, min_periods=1).mean()
        df["sma_long"] = df["close"].rolling(self.long_period, min_periods=1).mean()

        prev_short = df["sma_short"].shift(1)
        prev_long = df["sma_long"].shift(1)

        for i in range(1, len(df)):
            # 金叉: 短均线从下往上穿越长均线
            if prev_short.iloc[i] <= prev_long.iloc[i] and df["sma_short"].iloc[i] > df["sma_long"].iloc[i]:
                signals.append(Signal(
                    date=df["date"].iloc[i],
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=min(abs(df["sma_short"].iloc[i] - df["sma_long"].iloc[i]) / df["close"].iloc[i] * 10, 1.0),
                    price=df["close"].iloc[i],
                    target_pct=0.5,
                    reason=f"MA{self.short_period}上穿MA{self.long_period}",
                ))
            # 死叉
            elif prev_short.iloc[i] >= prev_long.iloc[i] and df["sma_short"].iloc[i] < df["sma_long"].iloc[i]:
                signals.append(Signal(
                    date=df["date"].iloc[i],
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=0.8,
                    price=df["close"].iloc[i],
                    target_pct=0.0,
                    reason=f"MA{self.short_period}下穿MA{self.long_period}",
                ))

        return signals


class MACDStrategy(BaseStrategy):
    """
    MACD策略
    DIF上穿DEA -> 买入 (金叉)
    DIF下穿DEA -> 卖出 (死叉)
    柱状图由负转正 / 由正转负 作为辅助信号
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(name="MACD", fast=fast, slow=slow, signal=signal)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        signals = []
        symbol = kwargs.get("symbol", "unknown")
        df = data.copy()

        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal_period, adjust=False).mean()

        prev_dif = dif.shift(1)
        prev_dea = dea.shift(1)

        for i in range(1, len(df)):
            # 金叉
            if prev_dif.iloc[i] <= prev_dea.iloc[i] and dif.iloc[i] > dea.iloc[i]:
                strength = min(abs(dif.iloc[i] - dea.iloc[i]) / df["close"].iloc[i] * 20, 1.0)
                signals.append(Signal(
                    date=df["date"].iloc[i],
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=df["close"].iloc[i],
                    target_pct=0.5,
                    reason="MACD金叉",
                ))
            # 死叉
            elif prev_dif.iloc[i] >= prev_dea.iloc[i] and dif.iloc[i] < dea.iloc[i]:
                signals.append(Signal(
                    date=df["date"].iloc[i],
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=0.8,
                    price=df["close"].iloc[i],
                    target_pct=0.0,
                    reason="MACD死叉",
                ))

        return signals


class RSIReversion(BaseStrategy):
    """
    RSI均值回归策略
    RSI < 超卖线 -> 买入
    RSI > 超买线 -> 卖出
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(name="RSI", period=period, oversold=oversold, overbought=overbought)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        signals = []
        symbol = kwargs.get("symbol", "unknown")
        df = data.copy()

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.period, min_periods=self.period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df["rsi"] = rsi

        prev_rsi = df["rsi"].shift(1)

        for i in range(1, len(df)):
            rsi_val = df["rsi"].iloc[i]
            prev_val = prev_rsi.iloc[i]

            if pd.isna(rsi_val) or pd.isna(prev_val):
                continue

            if prev_val <= self.oversold and rsi_val > self.oversold:
                strength = (self.oversold - prev_val) / self.oversold
                signals.append(Signal(
                    date=df["date"].iloc[i],
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=min(max(strength, 0.3), 1.0),
                    price=df["close"].iloc[i],
                    target_pct=0.5,
                    reason=f"RSI超卖反弹({prev_val:.1f}->{rsi_val:.1f})",
                ))
            elif prev_val >= self.overbought and rsi_val < self.overbought:
                signals.append(Signal(
                    date=df["date"].iloc[i],
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=0.7,
                    price=df["close"].iloc[i],
                    target_pct=0.0,
                    reason=f"RSI超买回落({prev_val:.1f}->{rsi_val:.1f})",
                ))

        return signals


class BOLLBreakout(BaseStrategy):
    """
    布林带突破策略
    价格突破下轨 -> 买入 (超跌反弹)
    价格突破上轨 -> 卖出 (过热)
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(name="BOLL", period=period, std_dev=std_dev)
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        signals = []
        symbol = kwargs.get("symbol", "unknown")
        df = data.copy()

        mid = df["close"].rolling(self.period, min_periods=1).mean()
        std = df["close"].rolling(self.period, min_periods=1).std()
        upper = mid + self.std_dev * std
        lower = mid - self.std_dev * std

        prev_close = df["close"].shift(1)

        for i in range(1, len(df)):
            if pd.isna(lower.iloc[i]):
                continue

            # 突破下轨
            if prev_close.iloc[i] >= lower.iloc[i] and df["close"].iloc[i] < lower.iloc[i]:
                signals.append(Signal(
                    date=df["date"].iloc[i],
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    price=df["close"].iloc[i],
                    target_pct=0.4,
                    reason="突破布林下轨",
                ))
            # 突破上轨
            elif prev_close.iloc[i] <= upper.iloc[i] and df["close"].iloc[i] > upper.iloc[i]:
                signals.append(Signal(
                    date=df["date"].iloc[i],
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=0.7,
                    price=df["close"].iloc[i],
                    target_pct=0.0,
                    reason="突破布林上轨",
                ))

        return signals


class KDJStrategy(BaseStrategy):
    """
    KDJ策略
    J值超卖区(K<20且J<0) -> 买入
    J值超买区(K>80且J>100) -> 卖出
    """

    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        super().__init__(name="KDJ", n=n, m1=m1, m2=m2)
        self.n = n
        self.m1 = m1
        self.m2 = m2

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        signals = []
        symbol = kwargs.get("symbol", "unknown")
        df = data.copy()

        lowest = df["low"].rolling(self.n, min_periods=1).min()
        highest = df["high"].rolling(self.n, min_periods=1).max()
        rsv = (df["close"] - lowest) / (highest - lowest).replace(0, np.nan) * 100
        rsv = rsv.fillna(50)

        k = rsv.ewm(alpha=1/self.m1, adjust=False).mean()
        d = k.ewm(alpha=1/self.m2, adjust=False).mean()
        j = 3 * k - 2 * d

        prev_k, prev_j = k.shift(1), j.shift(1)

        for i in range(1, len(df)):
            k_val, j_val = k.iloc[i], j.iloc[i]
            pk, pj = prev_k.iloc[i], prev_j.iloc[i]
            if pd.isna(k_val) or pd.isna(pk):
                continue

            if pk < 20 and pj < 0 and k_val > 20 and j_val > 0:
                signals.append(Signal(
                    date=df["date"].iloc[i], symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=min(abs(j_val) / 50, 1.0),
                    price=df["close"].iloc[i],
                    target_pct=0.5,
                    reason=f"KDJ超卖反转(K={k_val:.1f},J={j_val:.1f})",
                ))
            elif pk > 80 and pj > 100 and k_val < 80 and j_val < 100:
                signals.append(Signal(
                    date=df["date"].iloc[i], symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=0.7,
                    price=df["close"].iloc[i],
                    target_pct=0.0,
                    reason=f"KDJ超买反转(K={k_val:.1f},J={j_val:.1f})",
                ))

        return signals


class GridStrategy(BaseStrategy):
    """
    网格交易策略
    在设定的价格区间内，每隔一定间距挂单
    适合震荡市
    """

    def __init__(self, grid_size: float = 0.03, num_grids: int = 10,
                 upper_price: float = 0, lower_price: float = 0):
        super().__init__(name="Grid", grid_size=grid_size, num_grids=num_grids)
        self.grid_size = grid_size
        self.num_grids = num_grids
        self.upper_price = upper_price
        self.lower_price = lower_price

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        signals = []
        symbol = kwargs.get("symbol", "unknown")
        df = data.copy()

        if self.upper_price == 0:
            self.upper_price = df["close"].max() * 1.05
        if self.lower_price == 0:
            self.lower_price = df["close"].min() * 0.95

        grid_prices = np.linspace(self.lower_price, self.upper_price, self.num_grids)
        prev_close = df["close"].shift(1)

        for i in range(1, len(df)):
            price = df["close"].iloc[i]
            prev_p = prev_close.iloc[i]
            if pd.isna(prev_p):
                continue

            for gp in grid_prices:
                if prev_p < gp <= price:
                    signals.append(Signal(
                        date=df["date"].iloc[i], symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=0.5,
                        price=price,
                        target_pct=1.0 / self.num_grids,
                        reason=f"网格买入@{gp:.2f}",
                    ))
                elif prev_p > gp >= price:
                    signals.append(Signal(
                        date=df["date"].iloc[i], symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=0.5,
                        price=price,
                        target_pct=1.0 / self.num_grids,
                        reason=f"网格卖出@{gp:.2f}",
                    ))

        return signals


class TurtleStrategy(BaseStrategy):
    """
    海龟交易策略
    突破N日最高价 -> 买入
    跌破N日最低价 -> 卖出
    使用ATR计算仓位大小
    """

    def __init__(self, entry_period: int = 20, exit_period: int = 10,
                 atr_period: int = 20, risk_pct: float = 0.01):
        super().__init__(name="Turtle",
                         entry_period=entry_period, exit_period=exit_period,
                         atr_period=atr_period, risk_pct=risk_pct)
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_pct = risk_pct

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        signals = []
        symbol = kwargs.get("symbol", "unknown")
        df = data.copy()

        highest = df["high"].rolling(self.entry_period, min_periods=1).max()
        lowest = df["low"].rolling(self.exit_period, min_periods=1).min()

        prev_h = highest.shift(1)
        prev_l = lowest.shift(1)

        for i in range(1, len(df)):
            price = df["close"].iloc[i]
            ph, pl = prev_h.iloc[i], prev_l.iloc[i]
            if pd.isna(ph) or pd.isna(pl):
                continue

            if price > ph:
                signals.append(Signal(
                    date=df["date"].iloc[i], symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    price=price,
                    target_pct=0.4,
                    reason=f"海龟突破{self.entry_period}日高点({ph:.2f})",
                ))
            elif price < pl:
                signals.append(Signal(
                    date=df["date"].iloc[i], symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=0.8,
                    price=price,
                    target_pct=0.0,
                    reason=f"海龟跌破{self.exit_period}日低点({pl:.2f})",
                ))

        return signals


class MultiFactorStrategy(BaseStrategy):
    """
    多因子选股策略
    根据多个因子综合评分，选取得分最高的股票
    因子: 动量、波动率、成交量、RSI
    """

    def __init__(self, lookback: int = 60, top_n: int = 5):
        super().__init__(name="MultiFactor", lookback=lookback, top_n=top_n)
        self.lookback = lookback
        self.top_n = top_n

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        """多因子策略需要多只股票的数据, 此处对单只股票生成综合评分信号"""
        signals = []
        symbol = kwargs.get("symbol", "unknown")
        df = data.copy()

        if len(df) < self.lookback:
            return signals

        # 计算因子
        momentum = df["close"].pct_change(20)      # 20日动量
        volatility = df["close"].pct_change().rolling(20).std()  # 波动率(低好)
        vol_ratio = df["volume"] / df["volume"].rolling(20).mean()  # 成交量比

        # RSI因子
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # 综合评分 (标准化)
        def normalize(s):
            return (s - s.rolling(60, min_periods=10).mean()) / s.rolling(60, min_periods=10).std().replace(0, np.nan)

        score = (
            normalize(momentum) * 0.3 +
            normalize(-volatility) * 0.3 +  # 低波动得高分
            normalize(vol_ratio) * 0.2 +
            normalize(50 - (rsi - 50).abs()) * 0.2  # RSI中性得高分
        )

        score = score.fillna(0)

        for i in range(self.lookback, len(df)):
            if score.iloc[i] > 1.5:  # 高于1.5个标准差
                signals.append(Signal(
                    date=df["date"].iloc[i], symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=min(score.iloc[i] / 3, 1.0),
                    price=df["close"].iloc[i],
                    target_pct=0.2,
                    reason=f"多因子高评分({score.iloc[i]:.2f})",
                ))
            elif score.iloc[i] < -1.5:
                signals.append(Signal(
                    date=df["date"].iloc[i], symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=min(abs(score.iloc[i]) / 3, 1.0),
                    price=df["close"].iloc[i],
                    target_pct=0.0,
                    reason=f"多因子低评分({score.iloc[i]:.2f})",
                ))

        return signals
