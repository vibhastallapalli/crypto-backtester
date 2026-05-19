import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    trades: list
    equity_curve: pd.Series
    signals: pd.DataFrame


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _simulate(df: pd.DataFrame, signal_col: str, asset: str, strategy_name: str) -> BacktestResult:
    close = df["Close"].values.astype(float)
    signals = df[signal_col].values
    dates = df.index
    n = len(df)

    equity = np.ones(n)
    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    entry_equity = 1.0
    current_equity = 1.0
    trades = []

    for i in range(1, n):
        sig = signals[i]
        price = close[i]

        if not in_trade and sig == "buy":
            in_trade = True
            entry_price = price
            entry_idx = i
            entry_equity = current_equity

        if in_trade:
            current_equity = entry_equity * (price / entry_price)
            equity[i] = current_equity
            if sig == "sell" and i > entry_idx:
                pnl_pct = (price - entry_price) / entry_price
                trades.append({
                    "asset": asset,
                    "entry_date": str(dates[entry_idx].date()),
                    "exit_date": str(dates[i].date()),
                    "entry_price": round(float(entry_price), 6),
                    "exit_price": round(float(price), 6),
                    "signal_type": strategy_name,
                    "pnl": round(float(pnl_pct * entry_price), 6),
                    "pnl_pct": round(float(pnl_pct * 100), 4),
                })
                in_trade = False
        else:
            equity[i] = current_equity

    return BacktestResult(
        trades=trades,
        equity_curve=pd.Series(equity, index=dates),
        signals=df,
    )


def ma_crossover(df: pd.DataFrame, asset: str, fast: int = 10, slow: int = 30) -> BacktestResult:
    df = df.copy()
    df["fast_ma"] = df["Close"].rolling(fast).mean()
    df["slow_ma"] = df["Close"].rolling(slow).mean()
    cross_up = (df["fast_ma"] > df["slow_ma"]) & (df["fast_ma"].shift(1) <= df["slow_ma"].shift(1))
    cross_dn = (df["fast_ma"] < df["slow_ma"]) & (df["fast_ma"].shift(1) >= df["slow_ma"].shift(1))
    df["signal"] = "hold"
    df.loc[cross_up, "signal"] = "buy"
    df.loc[cross_dn, "signal"] = "sell"
    df = df.dropna(subset=["fast_ma", "slow_ma"]).copy()
    return _simulate(df, "signal", asset, "MA Crossover")


def rsi_mean_reversion(
    df: pd.DataFrame,
    asset: str,
    period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
) -> BacktestResult:
    df = df.copy()
    df["rsi"] = _calc_rsi(df["Close"], period)
    cross_os = (df["rsi"] < oversold) & (df["rsi"].shift(1) >= oversold)
    cross_ob = (df["rsi"] > overbought) & (df["rsi"].shift(1) <= overbought)
    df["signal"] = "hold"
    df.loc[cross_os, "signal"] = "buy"
    df.loc[cross_ob, "signal"] = "sell"
    df = df.dropna(subset=["rsi"]).copy()
    return _simulate(df, "signal", asset, "RSI Mean Reversion")


def volume_breakout(
    df: pd.DataFrame,
    asset: str,
    multiplier: float = 2.0,
    lookback: int = 20,
) -> BacktestResult:
    df = df.copy()
    df["avg_vol"] = df["Volume"].rolling(lookback).mean()
    df["high_lookback"] = df["Close"].rolling(lookback).max().shift(1)
    df = df.dropna(subset=["avg_vol", "high_lookback"]).copy()

    buy_mask = (df["Volume"] > multiplier * df["avg_vol"]) & (df["Close"] > df["high_lookback"])

    signals = ["hold"] * len(df)
    in_trade = False
    exit_bar = -1

    for i in range(len(df)):
        if not in_trade and buy_mask.iloc[i]:
            signals[i] = "buy"
            in_trade = True
            exit_bar = i + 5
        elif in_trade and i >= exit_bar:
            signals[i] = "sell"
            in_trade = False

    df["signal"] = signals
    return _simulate(df, "signal", asset, "Volume Breakout")


def compute_metrics(result: BacktestResult) -> dict:
    equity = result.equity_curve
    trades = result.trades

    empty = {
        "total_return": 0.0, "win_rate": 0.0, "sharpe": 0.0,
        "max_drawdown": 0.0, "num_trades": 0, "avg_pnl_pct": 0.0,
        "profit_factor": 0.0,
    }
    if not trades:
        return empty

    total_return = (float(equity.iloc[-1]) - 1.0) * 100.0

    pnl_pcts = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p < 0]
    win_rate = len(wins) / len(trades) * 100

    daily_rets = equity.pct_change().dropna()
    active = daily_rets[daily_rets != 0]
    if len(active) > 1 and active.std() > 0:
        sharpe = float((active.mean() * 252) / (active.std() * np.sqrt(252)))
    else:
        sharpe = 0.0

    peak = equity.cummax()
    max_drawdown = float(((equity - peak) / peak).min() * 100)

    gross_profit = sum(p for p in pnl_pcts if p > 0)
    gross_loss = abs(sum(losses))
    profit_factor: float | str = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")
    if profit_factor == float("inf"):
        profit_factor = "∞"

    return {
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown, 2),
        "num_trades": len(trades),
        "avg_pnl_pct": round(float(np.mean(pnl_pcts)), 2),
        "profit_factor": profit_factor,
    }
