"""
配置管理模块
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BrokerConfig:
    """券商配置"""
    name: str = "simulated"           # simulated / eastmoney / ths
    api_key: str = ""
    api_secret: str = ""
    account_id: str = ""


@dataclass
class DataConfig:
    """数据源配置"""
    source: str = "akshare"           # akshare / tushare / csv
    tushare_token: str = ""
    cache_dir: str = "./data_cache"
    start_date: str = "20200101"
    end_date: str = ""


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1_000_000.0    # 初始资金
    commission_rate: float = 0.0003          # 手续费率 (万三)
    stamp_tax_rate: float = 0.0005           # 印花税率 (卖出万五)
    slippage: float = 0.001                  # 滑点
    benchmark: str = "000300"                # 基准指数 (沪深300)
    start_date: str = "20200101"
    end_date: str = ""
    frequency: str = "daily"                 # daily / weekly / monthly


@dataclass
class RiskConfig:
    """风控配置"""
    max_position_pct: float = 0.25           # 单票最大仓位
    max_total_position_pct: float = 0.80     # 最大总仓位
    max_daily_loss_pct: float = 0.03         # 日最大亏损
    max_drawdown_pct: float = 0.15           # 最大回撤
    stop_loss_pct: float = 0.08              # 止损比例
    take_profit_pct: float = 0.20            # 止盈比例
    max_open_orders: int = 5                 # 最大挂单数


@dataclass
class SystemConfig:
    """系统总配置"""
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    log_level: str = "INFO"


def load_config(path: str) -> SystemConfig:
    """从YAML文件加载配置"""
    if not os.path.exists(path):
        return SystemConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = SystemConfig()
    if "broker" in raw:
        for k, v in raw["broker"].items():
            if hasattr(config.broker, k):
                setattr(config.broker, k, v)
    if "data" in raw:
        for k, v in raw["data"].items():
            if hasattr(config.data, k):
                setattr(config.data, k, v)
    if "backtest" in raw:
        for k, v in raw["backtest"].items():
            if hasattr(config.backtest, k):
                setattr(config.backtest, k, v)
    if "risk" in raw:
        for k, v in raw["risk"].items():
            if hasattr(config.risk, k):
                setattr(config.risk, k, v)
    if "log_level" in raw:
        config.log_level = raw["log_level"]
    return config


def save_config(config: SystemConfig, path: str):
    """保存配置到YAML文件"""
    data = {
        "broker": {
            "name": config.broker.name,
            "api_key": config.broker.api_key,
            "api_secret": config.broker.api_secret,
            "account_id": config.broker.account_id,
        },
        "data": {
            "source": config.data.source,
            "tushare_token": config.data.tushare_token,
            "cache_dir": config.data.cache_dir,
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
        },
        "backtest": {
            "initial_capital": config.backtest.initial_capital,
            "commission_rate": config.backtest.commission_rate,
            "stamp_tax_rate": config.backtest.stamp_tax_rate,
            "slippage": config.backtest.slippage,
            "benchmark": config.backtest.benchmark,
            "start_date": config.backtest.start_date,
            "end_date": config.backtest.end_date,
            "frequency": config.backtest.frequency,
        },
        "risk": {
            "max_position_pct": config.risk.max_position_pct,
            "max_total_position_pct": config.risk.max_total_position_pct,
            "max_daily_loss_pct": config.risk.max_daily_loss_pct,
            "max_drawdown_pct": config.risk.max_drawdown_pct,
            "stop_loss_pct": config.risk.stop_loss_pct,
            "take_profit_pct": config.risk.take_profit_pct,
            "max_open_orders": config.risk.max_open_orders,
        },
        "log_level": config.log_level,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
