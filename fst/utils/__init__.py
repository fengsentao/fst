"""
可视化工具模块
提供回测结果的图表绘制
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional


# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_equity_curve(result, benchmark_curve: Optional[pd.DataFrame] = None,
                      save_path: str = "backtest_report.png"):
    """
    绘制权益曲线

    参数:
        result: BacktestResult 对象
        benchmark_curve: 基准曲线 DataFrame (date, equity)
        save_path: 保存路径
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    ec = result.equity_curve
    dates = pd.to_datetime(ec["date"])

    # 1. 权益曲线
    ax1 = axes[0]
    ax1.plot(dates, ec["equity"], label="Strategy", linewidth=1.5, color="#2196F3")
    if benchmark_curve is not None and not benchmark_curve.empty:
        bm = benchmark_curve.copy()
        bm_dates = pd.to_datetime(bm["date"])
        # 归一化到初始资金
        bm["equity_norm"] = bm["equity"] / bm["equity"].iloc[0] * ec["equity"].iloc[0]
        ax1.plot(bm_dates, bm["equity_norm"], label="Benchmark",
                 linewidth=1, color="#9E9E9E", linestyle="--")
    ax1.set_title("Equity Curve / 权益曲线", fontsize=14)
    ax1.set_ylabel("Equity (¥)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"¥{x:,.0f}"))

    # 2. 回撤
    ax2 = axes[1]
    cummax = ec["equity"].cummax()
    drawdown = (ec["equity"] - cummax) / cummax * 100
    ax2.fill_between(dates, drawdown, 0, color="#F44336", alpha=0.3)
    ax2.plot(dates, drawdown, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown / 回撤", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. 持仓市值
    ax3 = axes[2]
    if "market_value" in ec.columns:
        ax3.fill_between(dates, ec["market_value"], 0, color="#4CAF50", alpha=0.3)
        ax3.plot(dates, ec["market_value"], color="#4CAF50", linewidth=0.8)
    ax3.set_ylabel("Position (¥)")
    ax3.set_title("Position Value / 持仓市值", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. 日收益
    ax4 = axes[3]
    daily_return = ec["equity"].pct_change().dropna() * 100
    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in daily_return]
    ax4.bar(dates.iloc[1:], daily_return, color=colors, alpha=0.6, width=1)
    ax4.set_ylabel("Daily Return (%)")
    ax4.set_title("Daily Returns / 日收益", fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_indicators(df: pd.DataFrame, indicators: list = None,
                    save_path: str = "indicators.png"):
    """
    绘制技术指标

    参数:
        df: 包含指标的DataFrame
        indicators: 要绘制的指标列表
        save_path: 保存路径
    """
    if indicators is None:
        indicators = ["sma_5", "sma_20", "boll_upper", "boll_lower", "macd", "rsi"]

    available = [ind for ind in indicators if ind in df.columns]
    n_plots = 1 + len([i for i in available if i in ("rsi", "k", "d", "j", "macd", "dif", "dea", "obv")])

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots),
                              sharex=True, gridspec_kw={"height_ratios": [3] + [1] * (n_plots - 1)})
    if n_plots == 1:
        axes = [axes]

    dates = pd.to_datetime(df["date"])
    ax_idx = 0

    # 价格 + 均线
    ax = axes[ax_idx]
    ax.plot(dates, df["close"], label="Close", linewidth=1.2, color="black")
    for ind in available:
        if ind.startswith("sma_") or ind.startswith("ema_"):
            ax.plot(dates, df[ind], label=ind, linewidth=0.8, alpha=0.8)
    if "boll_upper" in df.columns:
        ax.fill_between(dates, df.get("boll_upper", 0), df.get("boll_lower", 0),
                         alpha=0.1, color="blue")
    ax.set_title("Price & Indicators")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax_idx += 1

    # MACD
    if "macd" in available and ax_idx < n_plots:
        ax = axes[ax_idx]
        if "dif" in df.columns and "dea" in df.columns:
            ax.plot(dates, df["dif"], label="DIF", linewidth=1)
            ax.plot(dates, df["dea"], label="DEA", linewidth=1)
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in df["macd"]]
        ax.bar(dates, df["macd"], color=colors, alpha=0.6, width=1)
        ax.set_title("MACD")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    # RSI
    if "rsi" in available and ax_idx < n_plots:
        ax = axes[ax_idx]
        ax.plot(dates, df["rsi"], label="RSI", color="#9C27B0", linewidth=1)
        ax.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax.axhline(30, color="green", linestyle="--", alpha=0.5)
        ax.set_title("RSI")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_monthly_returns(result, save_path: str = "monthly_returns.png"):
    """绘制月度收益热力图"""
    ec = result.equity_curve
    ec = ec.copy()
    ec["date"] = pd.to_datetime(ec["date"])
    ec.set_index("date", inplace=True)
    ec["return"] = ec["equity"].pct_change()

    monthly = ec["return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = [f"{m}月" for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.6)))
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto",
                    vmin=-10, vmax=10)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=9, color="black")
    plt.colorbar(im, label="Return (%)")
    ax.set_title("Monthly Returns / 月度收益")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
