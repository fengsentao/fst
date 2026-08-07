"""
技术指标计算模块
支持常用的技术分析指标，纯 pandas/numpy 实现，无需 TA-Lib
"""

import pandas as pd
import numpy as np


# ======================================================================
# 趋势类指标
# ======================================================================

def SMA(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=period, min_periods=1).mean()


def EMA(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=period, adjust=False).mean()


def MACD(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD指标
    返回: (DIF, DEA, MACD柱)
    """
    ema_fast = EMA(close, fast)
    ema_slow = EMA(close, slow)
    dif = ema_fast - ema_slow
    dea = EMA(dif, signal)
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar


def BOLL(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    """
    布林带
    返回: (上轨, 中轨, 下轨)
    """
    mid = SMA(close, period)
    std = close.rolling(window=period, min_periods=1).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


# ======================================================================
# 动量类指标
# ======================================================================

def RSI(close: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标 RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def KDJ(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 9, m1: int = 3, m2: int = 3):
    """
    KDJ随机指标
    返回: (K, D, J)
    """
    lowest_low = low.rolling(window=n, min_periods=1).min()
    highest_high = high.rolling(window=n, min_periods=1).max()

    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)

    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def WR(high: pd.Series, low: pd.Series, close: pd.Series,
       period: int = 14) -> pd.Series:
    """威廉指标 Williams %R"""
    highest = high.rolling(window=period, min_periods=1).max()
    lowest = low.rolling(window=period, min_periods=1).min()
    wr = -100 * (highest - close) / (highest - lowest).replace(0, np.nan)
    return wr


def CCI(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """顺势指标 CCI"""
    tp = (high + low + close) / 3
    sma_tp = SMA(tp, period)
    mad = tp.rolling(window=period, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)
    return cci


def ATR(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """平均真实波幅 ATR"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def BIAS(close: pd.Series, period: int = 12) -> pd.Series:
    """乖离率"""
    ma = SMA(close, period)
    return (close - ma) / ma * 100


# ======================================================================
# 成交量类指标
# ======================================================================

def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """能量潮 OBV"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (volume * direction).cumsum()
    return obv


def VWAP(close: pd.Series, volume: pd.Series) -> pd.Series:
    """成交量加权平均价"""
    return (close * volume).cumsum() / volume.cumsum()


def Volume_MA(volume: pd.Series, period: int = 20) -> pd.Series:
    """成交量移动平均"""
    return SMA(volume, period)


# ======================================================================
# 综合计算
# ======================================================================

def compute_all_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    一次性计算常用指标并添加到DataFrame

    参数:
        df: 必须包含 date, open, high, low, close, volume 列
        params: 参数覆盖, 如 {"macd_fast": 10}
    返回:
        添加了指标列的 DataFrame
    """
    p = {
        "sma_periods": [5, 10, 20, 60],
        "ema_periods": [12, 26],
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "boll_period": 20,
        "rsi_period": 14,
        "kdj_n": 9,
        "atr_period": 14,
    }
    if params:
        p.update(params)

    df = df.copy()

    # 均线
    for n in p["sma_periods"]:
        df[f"sma_{n}"] = SMA(df["close"], n)
    for n in p["ema_periods"]:
        df[f"ema_{n}"] = EMA(df["close"], n)

    # MACD
    df["dif"], df["dea"], df["macd"] = MACD(df["close"], p["macd_fast"], p["macd_slow"], p["macd_signal"])

    # 布林带
    df["boll_upper"], df["boll_mid"], df["boll_lower"] = BOLL(df["close"], p["boll_period"])

    # RSI
    df["rsi"] = RSI(df["close"], p["rsi_period"])

    # KDJ
    df["k"], df["d"], df["j"] = KDJ(df["high"], df["low"], df["close"], p["kdj_n"])

    # ATR
    df["atr"] = ATR(df["high"], df["low"], df["close"], p["atr_period"])

    # 常用形态
    df["bias_12"] = BIAS(df["close"], 12)
    df["obv"] = OBV(df["close"], df["volume"])

    # 价格变动
    df["pct_change"] = df["close"].pct_change()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=5, min_periods=1).mean()

    return df
