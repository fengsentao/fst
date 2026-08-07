# FST-Quant: A股量化交易框架

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个功能完整的A股量化交易框架，支持策略开发、回测分析、风控管理和数据可视化。内置 Web 管理界面。

## ✨ 特性

- 🌐 **Web管理界面**: FastAPI + Vue3 前后端分离，支持图表分析
- 🎯 **8种内置策略**: 双均线、MACD、RSI、布林带、KDJ、网格、海龟、多因子
- 📊 **12+技术指标**: MA、EMA、MACD、RSI、KDJ、BOLL、ATR、CCI、OBV 等
- 🔄 **完整回测引擎**: 支持佣金、印花税、滑点、100股整数约束
- 🛡️ **风控模块**: 仓位管理、止损止盈、VaR/CVaR、最大回撤控制
- 📈 **数据可视化**: 权益曲线、回撤图、月度收益热力图、指标图表
- 🗃️ **多数据源**: 支持 akshare (免费) 和 tushare
- 💻 **CLI工具**: 命令行一键运行回测和分析

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/fengsentao/fst.git
cd fst

# 安装依赖
pip install -r requirements.txt

# 或以开发模式安装
pip install -e .
```

## 🚀 快速开始

### Web界面 (推荐)

```bash
# 安装Web依赖
pip install -r web/requirements.txt

# 启动Web服务
bash web/start.sh

# 或直接启动
python -m uvicorn web.backend.app:app --host 0.0.0.0 --port 8000

# 访问: http://localhost:8000
# API文档: http://localhost:8000/docs
```

Web界面功能:
- 📊 仪表盘 - 系统总览和快捷操作
- 🔬 策略回测 - 选择股票和策略，运行回测，查看收益曲线和交易记录
- 🎯 策略管理 - 查看所有内置策略说明和参数
- 📈 行情数据 - 查询个股行情和技术指标
- 📋 回测记录 - 查看历史回测任务

### 命令行使用

```bash
# 查看可用策略
python -m fst.cli list

# 运行双均线策略回测 (招商银行)
python -m fst.cli backtest --symbol 600036 --strategy dualma --start 20200101 --end 20251231

# 运行MACD策略回测
python -m fst.cli backtest --symbol 600519 --strategy macd --capital 500000

# 技术分析
python -m fst.cli analyze --symbol 600036

# 指定使用 tushare 数据源
python -m fst.cli backtest --symbol 600036 --strategy rsi --data-source tushare
```

### Python代码

```python
from fst.data import DataFetcher
from fst.strategies.strategies import MACDStrategy
from fst.backtest import BacktestEngine
from fst.config import BacktestConfig
from fst.utils import plot_equity_curve

# 1. 获取数据
fetcher = DataFetcher(source="akshare")
data = fetcher.get_stock_daily("600036", "20200101", "20251231", adjust="qfq")

# 2. 选择策略
strategy = MACDStrategy(fast=12, slow=26, signal=9)

# 3. 配置回测
config = BacktestConfig(
    initial_capital=1_000_000,
    commission_rate=0.0003,   # 万三手续费
    stamp_tax_rate=0.0005,    # 万五印花税 (仅卖出)
    slippage=0.001,           # 0.1% 滑点
)

# 4. 运行回测
engine = BacktestEngine(config)
result = engine.run_single(data, strategy, symbol="600036")

# 5. 查看结果
print(result.summary())

# 6. 绘制图表
plot_equity_curve(result, save_path="result.png")
```

## 📋 内置策略

| 策略 | 类名 | 说明 | 适用场景 |
|------|------|------|----------|
| 双均线 | `DualMAStrategy` | 短期/长期均线金叉死叉 | 趋势市 |
| MACD | `MACDStrategy` | DIF与DEA交叉 | 趋势市 |
| RSI均值回归 | `RSIReversion` | RSI超买超卖反转 | 震荡市 |
| 布林带 | `BOLLBreakout` | 突破上下轨 | 震荡→趋势 |
| KDJ | `KDJStrategy` | J值超买超卖 | 震荡市 |
| 网格 | `GridStrategy` | 固定间距网格交易 | 震荡市 |
| 海龟 | `TurtleStrategy` | N日高低点突破 | 强趋势市 |
| 多因子 | `MultiFactorStrategy` | 多因子综合评分 | 选股 |

## 📊 技术指标

| 类别 | 指标 |
|------|------|
| 趋势 | SMA, EMA, MACD, 布林带 |
| 动量 | RSI, KDJ, Williams %R, CCI, BIAS |
| 波动 | ATR |
| 成交量 | OBV, VWAP, 成交量均线 |

## 🛡️ 风控功能

```python
from fst.risk import RiskManager
from fst.config import RiskConfig

risk_config = RiskConfig(
    max_position_pct=0.25,       # 单票最大仓位25%
    max_total_position_pct=0.80, # 总仓位不超过80%
    max_daily_loss_pct=0.03,     # 日最大亏损3%
    max_drawdown_pct=0.15,       # 最大回撤15%
    stop_loss_pct=0.08,          # 止损线8%
    take_profit_pct=0.20,        # 止盈线20%
)

risk_mgr = RiskManager(risk_config)
```

## 📁 项目结构

```
fst/
├── fst/
│   ├── __init__.py         # 包入口
│   ├── cli.py              # 命令行工具
│   ├── config.py           # 配置管理
│   ├── data/               # 数据获取模块
│   │   └── __init__.py     # akshare/tushare 数据接口
│   ├── indicators/         # 技术指标
│   │   └── __init__.py     # 12+ 指标计算
│   ├── strategies/         # 策略模块
│   │   ├── __init__.py     # 策略基类
│   │   └── strategies.py   # 8种内置策略
│   ├── backtest/           # 回测引擎
│   │   └── __init__.py     # 事件驱动回测
│   ├── risk/               # 风控模块
│   │   └── __init__.py     # 仓位/止损/VaR
│   └── utils/              # 可视化工具
│       └── __init__.py     # 图表绘制
├── examples/               # 使用示例
│   ├── simple_backtest.py
│   ├── strategy_comparison.py
│   └── risk_analysis.py
├── config.yaml             # 配置模板
├── setup.py
├── requirements.txt
└── README.md
```

## ⚙️ 配置

复制 `config.yaml` 并根据需要修改:

```yaml
data:
  source: akshare
  cache_dir: ./data_cache
backtest:
  initial_capital: 1000000
  commission_rate: 0.0003
  stamp_tax_rate: 0.0005
  slippage: 0.001
risk:
  max_position_pct: 0.25
  stop_loss_pct: 0.08
```

## 🔧 自定义策略

继承 `BaseStrategy` 并实现 `generate_signals`:

```python
from fst.strategies import BaseStrategy, Signal, SignalType

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="MyStrategy")

    def generate_signals(self, data, **kwargs):
        signals = []
        symbol = kwargs.get("symbol", "unknown")

        for i in range(1, len(data)):
            # 你的交易逻辑
            if some_condition:
                signals.append(Signal(
                    date=data["date"].iloc[i],
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    price=data["close"].iloc[i],
                    target_pct=0.3,
                    reason="我的买入信号",
                ))
        return signals
```

## 📝 注意事项

1. **数据源**: akshare 是免费开源数据源，无需注册即可使用
2. **A股限制**: 回测引擎已考虑100股整数、T+1等A股交易规则
3. **仅为研究**: 本框架仅供学习研究使用，不构成投资建议
4. **回测不等于实盘**: 历史回测结果不代表未来收益

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
