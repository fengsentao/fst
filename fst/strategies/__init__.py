"""
策略基类和信号定义
"""

import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """交易信号"""
    date: pd.Timestamp
    symbol: str
    signal_type: SignalType
    strength: float = 1.0          # 信号强度 0~1
    price: float = 0.0            # 建议价格
    target_pct: float = 0.0       # 目标仓位比例
    reason: str = ""              # 信号原因

    def __repr__(self):
        return (f"Signal({self.date.date()} {self.symbol} "
                f"{self.signal_type.value} strength={self.strength:.2f} "
                f"reason={self.reason})")


class BaseStrategy(ABC):
    """
    策略基类
    所有策略都继承此类并实现 generate_signals 方法
    """

    def __init__(self, name: str = "BaseStrategy", **kwargs):
        self.name = name
        self.params = kwargs
        self._signals: List[Signal] = []

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        """
        生成交易信号

        参数:
            data: 包含行情数据和指标的 DataFrame
            **kwargs: 额外参数
        返回:
            信号列表
        """
        pass

    def on_bar(self, date: pd.Timestamp, row: pd.Series, portfolio: dict) -> Optional[Signal]:
        """
        逐bar回调 (用于实盘/事件驱动)
        默认调用 generate_signals, 子类可覆写实现逐bar逻辑
        """
        return None

    def reset(self):
        """重置策略状态"""
        self._signals.clear()

    def __repr__(self):
        return f"{self.name}(params={self.params})"
