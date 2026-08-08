"""
数据获取模块
支持从 akshare、tushare、CSV文件 获取A股行情数据
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger


class DataFetcher:
    """A股行情数据获取器"""

    def __init__(self, source: str = "akshare", cache_dir: str = "./data_cache", tushare_token: str = ""):
        self.source = source
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        if source == "tushare":
            import tushare as ts
            ts.set_token(tushare_token)
            self._ts_pro = ts.pro_api()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def get_stock_daily(self, symbol: str, start_date: str, end_date: str,
                        adjust: str = "qfq") -> pd.DataFrame:
        """
        获取个股日线行情

        参数:
            symbol: 股票代码, 如 "000001", "600036"
            start_date: 开始日期 "YYYYMMDD"
            end_date: 结束日期 "YYYYMMDD"
            adjust: 复权方式 "qfq"前复权 / "hfq"后复权 / ""不复权
        返回:
            DataFrame with columns: date, open, high, low, close, volume, amount, turnover
        """
        cache_key = f"daily_{symbol}_{start_date}_{end_date}_{adjust}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        if self.source == "akshare":
            df = self._akshare_stock_daily(symbol, start_date, end_date, adjust)
        elif self.source == "tushare":
            df = self._tushare_stock_daily(symbol, start_date, end_date, adjust)
        else:
            raise ValueError(f"不支持的数据源: {self.source}")

        if df is not None and not df.empty:
            self._save_cache(cache_key, df)
        return df

    def get_index_daily(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数日线行情, 如 "000300"(沪深300) / "000001"(上证指数)"""
        cache_key = f"index_{index_code}_{start_date}_{end_date}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        if self.source == "akshare":
            df = self._akshare_index_daily(index_code, start_date, end_date)
        elif self.source == "tushare":
            df = self._tushare_index_daily(index_code, start_date, end_date)
        else:
            raise ValueError(f"不支持的数据源: {self.source}")

        if df is not None and not df.empty:
            self._save_cache(cache_key, df)
        return df

    def get_stock_list(self) -> pd.DataFrame:
        """获取A股全部股票列表"""
        cache_key = "stock_list"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        if self.source == "akshare":
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            df = df[["代码", "名称"]].rename(columns={"代码": "code", "名称": "name"})
        elif self.source == "tushare":
            df = self._ts_pro.stock_basic(exchange="", list_status="L",
                                           fields="ts_code,symbol,name")
            df = df.rename(columns={"ts_code": "code", "symbol": "symbol"})
        else:
            raise ValueError(f"不支持的数据源: {self.source}")

        self._save_cache(cache_key, df)
        return df

    def get_financial_data(self, symbol: str) -> pd.DataFrame:
        """获取财务数据"""
        cache_key = f"finance_{symbol}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        if self.source == "akshare":
            import akshare as ak
            df = ak.stock_financial_analysis_indicator(symbol=symbol)
            if df is not None and not df.empty:
                self._save_cache(cache_key, df)
            return df
        elif self.source == "tushare":
            ts_code = self._symbol_to_ts(symbol)
            df = self._ts_pro.fina_indicator(ts_code=ts_code)
            if df is not None and not df.empty:
                self._save_cache(cache_key, df)
            return df
        return pd.DataFrame()

    def get_realtime_quote(self, symbols: list) -> pd.DataFrame:
        """获取实时行情"""
        if self.source == "akshare":
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            df = df[df["代码"].isin(symbols)]
            return df
        elif self.source == "tushare":
            ts_codes = [self._symbol_to_ts(s) for s in symbols]
            df = self._ts_pro.daily(ts_code=",".join(ts_codes))
            return df
        return pd.DataFrame()

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """从CSV文件加载数据"""
        df = pd.read_csv(filepath, parse_dates=["date"])
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"CSV缺少必要列: {col}")
        return df

    # ------------------------------------------------------------------
    # akshare 实现
    # ------------------------------------------------------------------

    def _akshare_stock_daily(self, symbol: str, start_date: str, end_date: str,
                              adjust: str) -> pd.DataFrame:
        import akshare as ak

        adjust_map = {"qfq": "qfq", "hfq": "hfq", "": ""}
        ak_adjust = adjust_map.get(adjust, "qfq")

        # 格式化日期为 YYYY-MM-DD
        sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

        # 方法1: stock_zh_a_hist (东方财富，可能被墙)
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=ak_adjust,
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "换手率": "turnover", "涨跌幅": "pct_change",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                return df
        except Exception as e:
            logger.warning(f"stock_zh_a_hist 失败: {e}")

        # 方法2: stock_zh_a_daily (网易，备用)
        try:
            # 需要加市场前缀
            if symbol.startswith("6") or symbol.startswith("9"):
                full_symbol = f"sh{symbol}"
            else:
                full_symbol = f"sz{symbol}"

            df = ak.stock_zh_a_daily(
                symbol=full_symbol,
                start_date=sd,
                end_date=ed,
                adjust=ak_adjust,
            )
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                # 确保列名一致
                for col in ["open", "high", "low", "close", "volume"]:
                    if col not in df.columns:
                        df[col] = 0
                return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning(f"stock_zh_a_daily 也失败: {e}")

        return pd.DataFrame()

    def _akshare_index_daily(self, index_code: str, start_date: str,
                              end_date: str) -> pd.DataFrame:
        import akshare as ak

        # akshare 的指数代码格式
        symbol_map = {
            "000300": "000300",   # 沪深300
            "000001": "000001",   # 上证指数
            "399001": "399001",   # 深证成指
            "399006": "399006",   # 创业板指
        }
        symbol = symbol_map.get(index_code, index_code)

        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{symbol}" if symbol.startswith("000") else f"sz{symbol}")
        except Exception:
            try:
                df = ak.index_zh_a_hist(symbol=symbol, period="daily",
                                         start_date=start_date, end_date=end_date)
            except Exception as e:
                logger.error(f"akshare 获取指数 {index_code} 数据失败: {e}")
                return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # 标准化列名
        rename_map = {
            "日期": "date", "date": "date",
            "开盘": "open", "open": "open",
            "收盘": "close", "close": "close",
            "最高": "high", "high": "high",
            "最低": "low", "low": "low",
            "成交量": "volume", "volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # tushare 实现
    # ------------------------------------------------------------------

    def _tushare_stock_daily(self, symbol: str, start_date: str,
                              end_date: str, adjust: str) -> pd.DataFrame:
        ts_code = self._symbol_to_ts(symbol)

        if adjust in ("qfq", "hfq"):
            df = self._ts_pro.daily(ts_code=ts_code,
                                     start_date=start_date, end_date=end_date)
            if adjust == "qfq":
                adj_df = self._ts_pro.adj_factor(ts_code=ts_code,
                                                  start_date=start_date, end_date=end_date)
                if adj_df is not None and adj_df is not None:
                    df = df.merge(adj_df[["trade_date", "adj_factor"]], on="trade_date")
                    latest_factor = df["adj_factor"].iloc[-1]
                    for col in ["open", "high", "low", "close"]:
                        df[col] = df[col] * df["adj_factor"] / latest_factor
        else:
            df = self._ts_pro.daily(ts_code=ts_code,
                                     start_date=start_date, end_date=end_date)

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "trade_date": "date",
            "vol": "volume",
            "pct_chg": "pct_change",
        })
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _tushare_index_daily(self, index_code: str, start_date: str,
                              end_date: str) -> pd.DataFrame:
        ts_code = self._index_to_ts(index_code)
        df = self._ts_pro.index_daily(ts_code=ts_code,
                                       start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "trade_date": "date",
            "vol": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _symbol_to_ts(symbol: str) -> str:
        """普通代码转 tushare 格式: 000001 -> 000001.SZ"""
        if symbol.startswith("6") or symbol.startswith("9"):
            return f"{symbol}.SH"
        return f"{symbol}.SZ"

    @staticmethod
    def _index_to_ts(index_code: str) -> str:
        if index_code.startswith("0"):
            return f"{index_code}.SH"
        return f"{index_code}.SZ"

    def _load_cache(self, key: str) -> pd.DataFrame:
        import hashlib
        path = os.path.join(self.cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.pkl")
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            age_hours = (datetime.now().timestamp() - mtime) / 3600
            if age_hours < 24:  # 缓存24小时有效
                return pd.read_pickle(path)
        return None

    def _save_cache(self, key: str, df: pd.DataFrame):
        import hashlib
        path = os.path.join(self.cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.pkl")
        df.to_pickle(path)
