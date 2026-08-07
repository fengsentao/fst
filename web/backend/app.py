"""
FST-Quant Web 后端 - FastAPI 应用
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# 确保可以导入 fst 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fst.data import DataFetcher
from fst.indicators import compute_all_indicators
from fst.strategies import Signal, SignalType
from fst.strategies.strategies import (
    DualMAStrategy, MACDStrategy, RSIReversion,
    BOLLBreakout, KDJStrategy, GridStrategy,
    TurtleStrategy, MultiFactorStrategy,
)
from fst.backtest import BacktestEngine, BacktestResult
from fst.config import BacktestConfig, RiskConfig
from fst.risk import RiskManager, PortfolioRiskAnalyzer


# ------------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------------

class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="股票代码，如 600036")
    strategy: str = Field("dualma", description="策略名称")
    start_date: str = Field("20220101", description="开始日期 YYYYMMDD")
    end_date: str = Field("20251231", description="结束日期 YYYYMMDD")
    initial_capital: float = Field(1000000, description="初始资金")
    commission_rate: float = Field(0.0003, description="手续费率")
    stamp_tax_rate: float = Field(0.0005, description="印花税率")
    slippage: float = Field(0.001, description="滑点")
    strategy_params: dict = Field(default_factory=dict, description="策略参数")


class StockSearchRequest(BaseModel):
    keyword: str = Field("", description="搜索关键词")


class StrategyInfo(BaseModel):
    name: str
    display_name: str
    description: str
    params: list


# ------------------------------------------------------------------
# 策略注册表
# ------------------------------------------------------------------

STRATEGIES = {
    "dualma": {
        "class": DualMAStrategy,
        "display_name": "双均线交叉",
        "description": "短期均线上穿长期均线买入，下穿卖出。适合趋势行情。",
        "params": [
            {"name": "short_period", "label": "短期均线", "type": "int", "default": 5, "min": 2, "max": 60},
            {"name": "long_period", "label": "长期均线", "type": "int", "default": 20, "min": 5, "max": 250},
        ],
    },
    "macd": {
        "class": MACDStrategy,
        "display_name": "MACD金叉死叉",
        "description": "DIF上穿DEA买入，下穿卖出。经典趋势跟踪策略。",
        "params": [
            {"name": "fast", "label": "快线周期", "type": "int", "default": 12},
            {"name": "slow", "label": "慢线周期", "type": "int", "default": 26},
            {"name": "signal", "label": "信号线", "type": "int", "default": 9},
        ],
    },
    "rsi": {
        "class": RSIReversion,
        "display_name": "RSI均值回归",
        "description": "RSI低于超卖线买入，高于超买线卖出。适合震荡行情。",
        "params": [
            {"name": "period", "label": "RSI周期", "type": "int", "default": 14},
            {"name": "oversold", "label": "超卖线", "type": "float", "default": 30},
            {"name": "overbought", "label": "超买线", "type": "float", "default": 70},
        ],
    },
    "boll": {
        "class": BOLLBreakout,
        "display_name": "布林带突破",
        "description": "价格突破下轨买入，突破上轨卖出。",
        "params": [
            {"name": "period", "label": "周期", "type": "int", "default": 20},
            {"name": "std_dev", "label": "标准差倍数", "type": "float", "default": 2.0},
        ],
    },
    "kdj": {
        "class": KDJStrategy,
        "display_name": "KDJ超买超卖",
        "description": "J值超卖区买入，超买区卖出。",
        "params": [
            {"name": "n", "label": "KDJ周期", "type": "int", "default": 9},
            {"name": "m1", "label": "K平滑", "type": "int", "default": 3},
            {"name": "m2", "label": "D平滑", "type": "int", "default": 3},
        ],
    },
    "grid": {
        "class": GridStrategy,
        "display_name": "网格交易",
        "description": "在价格区间内按固定间距网格交易，适合震荡市。",
        "params": [
            {"name": "grid_size", "label": "网格间距", "type": "float", "default": 0.03},
            {"name": "num_grids", "label": "网格数量", "type": "int", "default": 10},
        ],
    },
    "turtle": {
        "class": TurtleStrategy,
        "display_name": "海龟交易",
        "description": "突破N日最高价买入，跌破M日最低价卖出。经典趋势突破策略。",
        "params": [
            {"name": "entry_period", "label": "入场周期", "type": "int", "default": 20},
            {"name": "exit_period", "label": "出场周期", "type": "int", "default": 10},
            {"name": "atr_period", "label": "ATR周期", "type": "int", "default": 20},
        ],
    },
    "multifactor": {
        "class": MultiFactorStrategy,
        "display_name": "多因子选股",
        "description": "综合动量、波动率、成交量、RSI多因子评分。",
        "params": [
            {"name": "lookback", "label": "回看周期", "type": "int", "default": 60},
            {"name": "top_n", "label": "选股数量", "type": "int", "default": 5},
        ],
    },
}

# 任务存储
backtest_tasks = {}
data_cache = {}


# ------------------------------------------------------------------
# FastAPI 应用
# ------------------------------------------------------------------

app = FastAPI(
    title="FST-Quant API",
    description="A股量化交易系统 Web API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# API 路由
# ------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "FST-Quant API 服务运行中", "version": "1.0.0"}


@app.get("/api/strategies")
async def list_strategies():
    """获取所有可用策略"""
    result = []
    for key, info in STRATEGIES.items():
        result.append({
            "id": key,
            "name": info["display_name"],
            "description": info["description"],
            "params": info["params"],
        })
    return {"strategies": result}


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
    """提交回测任务"""
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"未知策略: {req.strategy}")

    task_id = str(uuid.uuid4())[:8]
    backtest_tasks[task_id] = {"status": "running", "progress": 0}

    background_tasks.add_task(_run_backtest_task, task_id, req)
    return {"task_id": task_id, "status": "running"}


@app.get("/api/backtest/{task_id}")
async def get_backtest_result(task_id: str):
    """获取回测结果"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = backtest_tasks[task_id]
    if task["status"] == "running":
        return {"status": "running", "progress": task.get("progress", 0)}
    return task


@app.get("/api/backtest/{task_id}/equity")
async def get_equity_curve(task_id: str):
    """获取权益曲线数据"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = backtest_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="回测未完成")
    return {"equity_curve": task.get("equity_curve", [])}


@app.get("/api/backtest/{task_id}/trades")
async def get_trades(task_id: str):
    """获取交易记录"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = backtest_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="回测未完成")
    return {"trades": task.get("trades", [])}


@app.get("/api/backtest/{task_id}/signals")
async def get_signals(task_id: str):
    """获取交易信号"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = backtest_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="回测未完成")
    return {"signals": task.get("signals", [])}


@app.post("/api/data/stock")
async def fetch_stock_data(symbol: str, start_date: str = "20220101",
                           end_date: str = "20251231", adjust: str = "qfq"):
    """获取股票行情数据"""
    try:
        fetcher = DataFetcher(source="akshare", cache_dir="./data_cache")
        df = fetcher.get_stock_daily(symbol, start_date, end_date, adjust)
        if df.empty:
            raise HTTPException(status_code=404, detail="未获取到数据")
        return {
            "symbol": symbol,
            "count": len(df),
            "data": json.loads(df[["date", "open", "high", "low", "close", "volume"]].to_json(orient="records")),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/indicators")
async def compute_indicators(symbol: str, start_date: str = "20220101",
                              end_date: str = "20251231"):
    """获取带指标的数据"""
    try:
        fetcher = DataFetcher(source="akshare", cache_dir="./data_cache")
        df = fetcher.get_stock_daily(symbol, start_date, end_date, adjust="qfq")
        if df.empty:
            raise HTTPException(status_code=404, detail="未获取到数据")
        df = compute_all_indicators(df)
        cols = ["date", "open", "high", "low", "close", "volume",
                "sma_5", "sma_10", "sma_20", "dif", "dea", "macd",
                "rsi", "k", "d", "j", "boll_upper", "boll_lower", "atr"]
        available = [c for c in cols if c in df.columns]
        return {
            "symbol": symbol,
            "count": len(df),
            "data": json.loads(df[available].to_json(orient="records")),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/list")
async def list_backtest_tasks():
    """列出所有回测任务"""
    tasks = []
    for tid, task in backtest_tasks.items():
        tasks.append({
            "task_id": tid,
            "status": task.get("status", "unknown"),
            "symbol": task.get("symbol", ""),
            "strategy": task.get("strategy", ""),
            "created_at": task.get("created_at", ""),
        })
    return {"tasks": tasks}


# ------------------------------------------------------------------
# 后台回测任务
# ------------------------------------------------------------------

def _run_backtest_task(task_id: str, req: BacktestRequest):
    """后台执行回测"""
    try:
        backtest_tasks[task_id]["progress"] = 10
        backtest_tasks[task_id]["symbol"] = req.symbol
        backtest_tasks[task_id]["strategy"] = req.strategy
        backtest_tasks[task_id]["created_at"] = datetime.now().isoformat()

        # 获取数据
        fetcher = DataFetcher(source="akshare", cache_dir="./data_cache")
        data = fetcher.get_stock_daily(
            req.symbol, req.start_date, req.end_date, adjust="qfq"
        )
        if data.empty:
            backtest_tasks[task_id]["status"] = "failed"
            backtest_tasks[task_id]["error"] = "数据获取失败"
            return

        backtest_tasks[task_id]["progress"] = 40

        # 创建策略
        strategy_info = STRATEGIES[req.strategy]
        strategy = strategy_info["class"](**req.strategy_params)

        # 配置回测
        config = BacktestConfig(
            initial_capital=req.initial_capital,
            commission_rate=req.commission_rate,
            stamp_tax_rate=req.stamp_tax_rate,
            slippage=req.slippage,
        )

        backtest_tasks[task_id]["progress"] = 60

        # 运行回测
        engine = BacktestEngine(config)
        result = engine.run_single(data, strategy, symbol=req.symbol)

        backtest_tasks[task_id]["progress"] = 80

        # 序列化结果
        ec = result.equity_curve
        equity_curve = []
        if not ec.empty:
            for _, row in ec.iterrows():
                equity_curve.append({
                    "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                    "equity": round(float(row["equity"]), 2),
                    "capital": round(float(row.get("capital", 0)), 2),
                    "market_value": round(float(row.get("market_value", 0)), 2),
                })

        trades = []
        for t in result.trades:
            trades.append({
                "date": str(t.date.date()) if hasattr(t.date, "date") else str(t.date),
                "symbol": t.symbol,
                "side": t.side,
                "price": round(float(t.price), 2),
                "shares": int(t.shares),
                "amount": round(float(t.amount), 2),
                "commission": round(float(t.commission), 2),
                "tax": round(float(t.tax), 2),
                "reason": t.reason,
            })

        backtest_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "equity_curve": equity_curve,
            "trades": trades,
            "summary": {
                "total_return": round(result.total_return * 100, 2),
                "annual_return": round(result.annual_return * 100, 2),
                "max_drawdown": round(result.max_drawdown * 100, 2),
                "sharpe_ratio": round(result.sharpe_ratio, 3),
                "sortino_ratio": round(result.sortino_ratio, 3),
                "calmar_ratio": round(result.calmar_ratio, 3),
                "win_rate": round(result.win_rate * 100, 2),
                "profit_loss_ratio": round(result.profit_loss_ratio, 2),
                "total_trades": result.total_trades,
                "avg_holding_days": round(result.avg_holding_days, 1),
                "start_date": result.start_date,
                "end_date": result.end_date,
                "initial_capital": req.initial_capital,
                "final_equity": round(float(ec["equity"].iloc[-1]), 2) if not ec.empty else req.initial_capital,
            },
        })

    except Exception as e:
        backtest_tasks[task_id]["status"] = "failed"
        backtest_tasks[task_id]["error"] = str(e)


# ------------------------------------------------------------------
# 静态文件服务 (Vue前端)
# ------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Vue SPA 路由兜底"""
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
