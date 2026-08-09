"""
FST-Quant Web 后端 - FastAPI 应用 (含JWT认证)
"""

import os
import sys
import json
import uuid
import hashlib
import secrets
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

import jwt

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
# JWT 配置
# ------------------------------------------------------------------

JWT_SECRET = os.environ.get("FST_JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

# 默认管理员账户 (username: admin, password: fst@2024)
# 密码使用 sha256 哈希存储
USERS_DB = {
    "admin": {
        "username": "admin",
        "password_hash": hashlib.sha256("fst@2024".encode()).hexdigest(),
        "role": "admin",
        "created_at": "2024-01-01",
    }
}

security = HTTPBearer(auto_error=False)


# ------------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

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


# ------------------------------------------------------------------
# 认证工具
# ------------------------------------------------------------------

def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的认证令牌")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """依赖注入：获取当前登录用户"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="未登录，请先登录")
    payload = verify_token(credentials.credentials)
    username = payload.get("sub")
    if username not in USERS_DB:
        raise HTTPException(status_code=401, detail="用户不存在")
    return {"username": username, "role": USERS_DB[username]["role"]}


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
    description="A股量化交易系统 Web API (需要登录认证)",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# 认证 API
# ------------------------------------------------------------------

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """用户登录"""
    user = USERS_DB.get(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    password_hash = hashlib.sha256(req.password.encode()).hexdigest()
    if password_hash != user["password_hash"]:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_token(user["username"], user["role"])
    return {
        "token": token,
        "username": user["username"],
        "role": user["role"],
        "expires_in": JWT_EXPIRE_HOURS * 3600,
    }


@app.get("/api/auth/me")
async def get_me(user=Depends(get_current_user)):
    """获取当前用户信息"""
    return {"username": user["username"], "role": user["role"]}


@app.post("/api/auth/change-password")
async def change_password(req: ChangePasswordRequest, user=Depends(get_current_user)):
    """修改密码"""
    old_hash = hashlib.sha256(req.old_password.encode()).hexdigest()
    if old_hash != USERS_DB[user["username"]]["password_hash"]:
        raise HTTPException(status_code=400, detail="原密码错误")
    new_hash = hashlib.sha256(req.new_password.encode()).hexdigest()
    USERS_DB[user["username"]]["password_hash"] = new_hash
    return {"message": "密码修改成功"}


# ------------------------------------------------------------------
# 业务 API (需要认证)
# ------------------------------------------------------------------

@app.get("/api/strategies")
async def list_strategies(user=Depends(get_current_user)):
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


# ------------------------------------------------------------------
# 股票列表（支持名称搜索）
# ------------------------------------------------------------------

STOCK_LIST = [
    {"code": "600000", "name": "浦发银行"},
    {"code": "600009", "name": "上海机场"},
    {"code": "600016", "name": "民生银行"},
    {"code": "600028", "name": "中国石化"},
    {"code": "600030", "name": "中信证券"},
    {"code": "600031", "name": "三一重工"},
    {"code": "600036", "name": "招商银行"},
    {"code": "600048", "name": "保利发展"},
    {"code": "600050", "name": "中国联通"},
    {"code": "600085", "name": "同仁堂"},
    {"code": "600104", "name": "上汽集团"},
    {"code": "600111", "name": "北方稀土"},
    {"code": "600115", "name": "东方航空"},
    {"code": "600132", "name": "重庆啤酒"},
    {"code": "600150", "name": "中国船舶"},
    {"code": "600176", "name": "中国巨石"},
    {"code": "600183", "name": "生益科技"},
    {"code": "600196", "name": "复星医药"},
    {"code": "600276", "name": "恒瑞医药"},
    {"code": "600309", "name": "万华化学"},
    {"code": "600346", "name": "恒力石化"},
    {"code": "600352", "name": "浙江龙盛"},
    {"code": "600406", "name": "国电南瑞"},
    {"code": "600436", "name": "片仔癀"},
    {"code": "600438", "name": "通威股份"},
    {"code": "600519", "name": "贵州茅台"},
    {"code": "600547", "name": "山东黄金"},
    {"code": "600570", "name": "恒生电子"},
    {"code": "600585", "name": "海螺水泥"},
    {"code": "600588", "name": "用友网络"},
    {"code": "600600", "name": "青岛啤酒"},
    {"code": "600660", "name": "福耀玻璃"},
    {"code": "600690", "name": "海尔智家"},
    {"code": "600703", "name": "三安光电"},
    {"code": "600741", "name": "华域汽车"},
    {"code": "600809", "name": "山西汾酒"},
    {"code": "600837", "name": "海通证券"},
    {"code": "600887", "name": "伊利股份"},
    {"code": "600893", "name": "航发动力"},
    {"code": "600900", "name": "长江电力"},
    {"code": "600919", "name": "江苏银行"},
    {"code": "600926", "name": "杭州银行"},
    {"code": "600941", "name": "中国移动"},
    {"code": "601012", "name": "隆基绿能"},
    {"code": "601066", "name": "中信建投"},
    {"code": "601088", "name": "中国神华"},
    {"code": "601111", "name": "中国国航"},
    {"code": "601138", "name": "工业富联"},
    {"code": "601166", "name": "兴业银行"},
    {"code": "601186", "name": "中国铁建"},
    {"code": "601211", "name": "国泰君安"},
    {"code": "601225", "name": "陕西煤业"},
    {"code": "601288", "name": "农业银行"},
    {"code": "601318", "name": "中国平安"},
    {"code": "601328", "name": "交通银行"},
    {"code": "601336", "name": "新华保险"},
    {"code": "601390", "name": "中国中铁"},
    {"code": "601398", "name": "工商银行"},
    {"code": "601601", "name": "中国太保"},
    {"code": "601628", "name": "中国人寿"},
    {"code": "601633", "name": "长城汽车"},
    {"code": "601668", "name": "中国建筑"},
    {"code": "601688", "name": "华泰证券"},
    {"code": "601766", "name": "中国中车"},
    {"code": "601788", "name": "光大证券"},
    {"code": "601818", "name": "光大银行"},
    {"code": "601857", "name": "中国石油"},
    {"code": "601881", "name": "中国银河"},
    {"code": "601888", "name": "中国中免"},
    {"code": "601899", "name": "紫金矿业"},
    {"code": "601919", "name": "中远海控"},
    {"code": "601985", "name": "中国核电"},
    {"code": "601988", "name": "中国银行"},
    {"code": "603019", "name": "中科曙光"},
    {"code": "603259", "name": "药明康德"},
    {"code": "603288", "name": "海天味业"},
    {"code": "603501", "name": "韦尔股份"},
    {"code": "000001", "name": "平安银行"},
    {"code": "000002", "name": "万科A"},
    {"code": "000063", "name": "中兴通讯"},
    {"code": "000100", "name": "TCL科技"},
    {"code": "000157", "name": "中联重科"},
    {"code": "000333", "name": "美的集团"},
    {"code": "000338", "name": "潍柴动力"},
    {"code": "000425", "name": "徐工机械"},
    {"code": "000538", "name": "云南白药"},
    {"code": "000568", "name": "泸州老窖"},
    {"code": "000596", "name": "古井贡酒"},
    {"code": "000625", "name": "长安汽车"},
    {"code": "000651", "name": "格力电器"},
    {"code": "000661", "name": "长春高新"},
    {"code": "000725", "name": "京东方A"},
    {"code": "000776", "name": "广发证券"},
    {"code": "000858", "name": "五粮液"},
    {"code": "000895", "name": "双汇发展"},
    {"code": "000938", "name": "紫光股份"},
    {"code": "000977", "name": "浪潮信息"},
    {"code": "002001", "name": "新和成"},
    {"code": "002027", "name": "分众传媒"},
    {"code": "002049", "name": "紫光国微"},
    {"code": "002142", "name": "宁波银行"},
    {"code": "002230", "name": "科大讯飞"},
    {"code": "002304", "name": "洋河股份"},
    {"code": "002352", "name": "顺丰控股"},
    {"code": "002371", "name": "北方华创"},
    {"code": "002415", "name": "海康威视"},
    {"code": "002460", "name": "赣锋锂业"},
    {"code": "002475", "name": "立讯精密"},
    {"code": "002594", "name": "比亚迪"},
    {"code": "002714", "name": "牧原股份"},
    {"code": "300015", "name": "爱尔眼科"},
    {"code": "300059", "name": "东方财富"},
    {"code": "300122", "name": "智飞生物"},
    {"code": "300124", "name": "汇川技术"},
    {"code": "300274", "name": "阳光电源"},
    {"code": "300750", "name": "宁德时代"},
    {"code": "300760", "name": "迈瑞医疗"},
]

_stock_list_cache = None

def _load_stock_list():
    """动态加载A股股票列表"""
    global _stock_list_cache
    if _stock_list_cache is not None:
        return _stock_list_cache
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        _stock_list_cache = [{"code": row["code"], "name": row["name"]} for _, row in df.iterrows()]
        print(f"已加载 {len(_stock_list_cache)} 只A股股票")
    except Exception as e:
        print(f"加载股票列表失败: {e}，使用内置列表")
        _stock_list_cache = STOCK_LIST
    return _stock_list_cache


@app.get("/api/stocks")
async def search_stocks(q: str = "", user=Depends(get_current_user)):
    """搜索股票（支持代码和名称模糊匹配）"""
    stocks = _load_stock_list()
    if not q:
        return {"stocks": stocks[:20]}
    q_lower = q.lower()
    results = []
    for stock in stocks:
        if q_lower in stock["code"].lower() or q_lower in stock["name"].lower():
            results.append(stock)
        if len(results) >= 20:
            break
    return {"stocks": results}


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest, background_tasks: BackgroundTasks,
                       user=Depends(get_current_user)):
    """提交回测任务"""
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"未知策略: {req.strategy}")

    task_id = str(uuid.uuid4())[:8]
    backtest_tasks[task_id] = {"status": "running", "progress": 0, "user": user["username"]}

    background_tasks.add_task(_run_backtest_task, task_id, req)
    return {"task_id": task_id, "status": "running"}


@app.get("/api/backtest/list")
async def list_backtest_tasks(user=Depends(get_current_user)):
    """列出当前用户的所有回测任务"""
    tasks = []
    for tid, task in backtest_tasks.items():
        if task.get("user") == user["username"] or user["role"] == "admin":
            tasks.append({
                "task_id": tid,
                "status": task.get("status", "unknown"),
                "symbol": task.get("symbol", ""),
                "strategy": task.get("strategy", ""),
                "created_at": task.get("created_at", ""),
            })
    return {"tasks": tasks}


@app.get("/api/backtest/{task_id}")
async def get_backtest_result(task_id: str, user=Depends(get_current_user)):
    """获取回测结果"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = backtest_tasks[task_id]
    if task.get("user") != user["username"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="无权访问此任务")
    if task["status"] == "running":
        return {"status": "running", "progress": task.get("progress", 0)}
    return task


@app.get("/api/backtest/{task_id}/equity")
async def get_equity_curve(task_id: str, user=Depends(get_current_user)):
    """获取权益曲线数据"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = backtest_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="回测未完成")
    return {"equity_curve": task.get("equity_curve", [])}


@app.get("/api/backtest/{task_id}/trades")
async def get_trades(task_id: str, user=Depends(get_current_user)):
    """获取交易记录"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = backtest_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="回测未完成")
    return {"trades": task.get("trades", [])}


@app.post("/api/data/stock")
async def fetch_stock_data(symbol: str, start_date: str = "20220101",
                           end_date: str = "20251231", adjust: str = "qfq",
                           user=Depends(get_current_user)):
    """获取股票行情数据"""
    try:
        df = pd.DataFrame()
        # 尝试 akshare
        try:
            fetcher = DataFetcher(source="akshare", cache_dir="./data_cache")
            df = fetcher.get_stock_daily(symbol, start_date, end_date, adjust)
        except Exception as e:
            print(f"akshare获取数据失败: {e}")
        # 回退到模拟数据
        if df.empty:
            df = _generate_mock_data(symbol, start_date, end_date)
        if df.empty:
            raise HTTPException(status_code=404, detail="未获取到数据")
        # 日期转字符串
        df["date"] = df["date"].astype(str)
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
                              end_date: str = "20251231",
                              user=Depends(get_current_user)):
    """获取带指标的数据"""
    try:
        df = pd.DataFrame()
        # 尝试 akshare
        try:
            fetcher = DataFetcher(source="akshare", cache_dir="./data_cache")
            df = fetcher.get_stock_daily(symbol, start_date, end_date, adjust="qfq")
        except Exception as e:
            print(f"akshare获取数据失败: {e}")
        # 回退到模拟数据
        if df.empty:
            df = _generate_mock_data(symbol, start_date, end_date)
        if df.empty:
            raise HTTPException(status_code=404, detail="未获取到数据")
        # 日期转字符串
        df["date"] = df["date"].astype(str)
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
        data = pd.DataFrame()

        # 方法1: akshare (带超时，使用线程)
        try:
            import concurrent.futures
            def fetch_data():
                fetcher = DataFetcher(source="akshare", cache_dir="./data_cache")
                return fetcher.get_stock_daily(
                    req.symbol, req.start_date, req.end_date, adjust="qfq"
                )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_data)
                try:
                    data = future.result(timeout=10)
                except concurrent.futures.TimeoutError:
                    print("akshare获取超时(10s)，使用模拟数据")
                except Exception as e:
                    print(f"akshare获取数据失败: {e}")
        except Exception as e:
            print(f"akshare线程执行失败: {e}")

        # 方法2: 模拟数据
        if data.empty:
            print("使用模拟数据进行回测")
            data = _generate_mock_data(req.symbol, req.start_date, req.end_date)

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
        import traceback
        traceback.print_exc()
        backtest_tasks[task_id]["status"] = "failed"
        backtest_tasks[task_id]["error"] = str(e)


def _generate_mock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """生成模拟数据"""
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        dates = pd.date_range(start=start, end=end, freq='B')
        if len(dates) == 0:
            return pd.DataFrame()
        np.random.seed(hash(symbol) % 2**32)
        initial_price = 50 + np.random.randn() * 20
        returns = np.random.randn(len(dates)) * 0.02
        prices = initial_price * (1 + returns).cumsum()
        prices = np.maximum(prices, 10)
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.randn(len(dates)) * 0.01),
            'high': prices * (1 + abs(np.random.randn(len(dates)) * 0.02)),
            'low': prices * (1 - abs(np.random.randn(len(dates)) * 0.02)),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates)).astype(float),
        })
        data = compute_all_indicators(data)
        return data
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------------
# 静态文件服务 (Vue前端)
# ------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

if os.path.exists(FRONTEND_DIR):
    @app.get("/")
    async def root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

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
