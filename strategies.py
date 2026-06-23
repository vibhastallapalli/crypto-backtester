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


def _simulate(
    df: pd.DataFrame,
    signal_col: str,
    asset: str,
    strategy_name: str,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
) -> BacktestResult:
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
            pnl_pct = (price - entry_price) / entry_price
            sl_hit = stop_loss > 0 and pnl_pct <= -stop_loss / 100
            tp_hit = take_profit > 0 and pnl_pct >= take_profit / 100
            exit_signal = (sig == "sell" and i > entry_idx) or sl_hit or tp_hit
            if exit_signal:
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


def ma_crossover(df: pd.DataFrame, asset: str, fast: int = 10, slow: int = 30, stop_loss: float = 0.0, take_profit: float = 0.0) -> BacktestResult:
    df = df.copy()
    df["fast_ma"] = df["Close"].rolling(fast).mean()
    df["slow_ma"] = df["Close"].rolling(slow).mean()
    cross_up = (df["fast_ma"] > df["slow_ma"]) & (df["fast_ma"].shift(1) <= df["slow_ma"].shift(1))
    cross_dn = (df["fast_ma"] < df["slow_ma"]) & (df["fast_ma"].shift(1) >= df["slow_ma"].shift(1))
    df["signal"] = "hold"
    df.loc[cross_up, "signal"] = "buy"
    df.loc[cross_dn, "signal"] = "sell"
    df = df.dropna(subset=["fast_ma", "slow_ma"]).copy()
    return _simulate(df, "signal", asset, "MA Crossover", stop_loss, take_profit)


def rsi_mean_reversion(
    df: pd.DataFrame,
    asset: str,
    period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
) -> BacktestResult:
    df = df.copy()
    df["rsi"] = _calc_rsi(df["Close"], period)
    cross_os = (df["rsi"] < oversold) & (df["rsi"].shift(1) >= oversold)
    cross_ob = (df["rsi"] > overbought) & (df["rsi"].shift(1) <= overbought)
    df["signal"] = "hold"
    df.loc[cross_os, "signal"] = "buy"
    df.loc[cross_ob, "signal"] = "sell"
    df = df.dropna(subset=["rsi"]).copy()
    return _simulate(df, "signal", asset, "RSI Mean Reversion", stop_loss, take_profit)


def volume_breakout(
    df: pd.DataFrame,
    asset: str,
    multiplier: float = 2.0,
    lookback: int = 20,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
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
    return _simulate(df, "signal", asset, "Volume Breakout", stop_loss, take_profit)


def bollinger_bands(
    df: pd.DataFrame,
    asset: str,
    period: int = 20,
    std_dev: float = 2.0,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
) -> BacktestResult:
    df = df.copy()
    df["bb_mid"] = df["Close"].rolling(period).mean()
    rolling_std = df["Close"].rolling(period).std()
    df["bb_upper"] = df["bb_mid"] + std_dev * rolling_std
    df["bb_lower"] = df["bb_mid"] - std_dev * rolling_std
    df = df.dropna(subset=["bb_mid", "bb_upper", "bb_lower"]).copy()
    touch_lower = (df["Close"] <= df["bb_lower"]) & (df["Close"].shift(1) > df["bb_lower"].shift(1))
    touch_upper = (df["Close"] >= df["bb_upper"]) & (df["Close"].shift(1) < df["bb_upper"].shift(1))
    df["signal"] = "hold"
    df.loc[touch_lower, "signal"] = "buy"
    df.loc[touch_upper, "signal"] = "sell"
    return _simulate(df, "signal", asset, "Bollinger Bands", stop_loss, take_profit)


def ema_crossover(df: pd.DataFrame, asset: str, fast: int = 9, slow: int = 21, stop_loss: float = 0.0, take_profit: float = 0.0) -> BacktestResult:
    df = df.copy()
    df["fast_ema"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["slow_ema"] = df["Close"].ewm(span=slow, adjust=False).mean()
    cross_up = (df["fast_ema"] > df["slow_ema"]) & (df["fast_ema"].shift(1) <= df["slow_ema"].shift(1))
    cross_dn = (df["fast_ema"] < df["slow_ema"]) & (df["fast_ema"].shift(1) >= df["slow_ema"].shift(1))
    df["signal"] = "hold"
    df.loc[cross_up, "signal"] = "buy"
    df.loc[cross_dn, "signal"] = "sell"
    return _simulate(df, "signal", asset, "EMA Crossover", stop_loss, take_profit)


def macd_crossover(
    df: pd.DataFrame,
    asset: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
) -> BacktestResult:
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    cross_up = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    cross_dn = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    df["signal"] = "hold"
    df.loc[cross_up, "signal"] = "buy"
    df.loc[cross_dn, "signal"] = "sell"
    df = df.dropna(subset=["macd", "macd_signal"]).copy()
    return _simulate(df, "signal", asset, "MACD Crossover", stop_loss, take_profit)


def compute_metrics(result: BacktestResult) -> dict:
    equity = result.equity_curve
    trades = result.trades

    empty = {
        "total_return": 0.0, "win_rate": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
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

    downside = daily_rets[daily_rets < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = float((active.mean() * 252) / (downside.std() * np.sqrt(252)))
    else:
        sortino = 0.0

    ann_return = float(active.mean() * 252) * 100
    calmar = round(ann_return / abs(max_drawdown), 2) if max_drawdown < 0 else 0.0

    return {
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": calmar,
        "max_drawdown": round(max_drawdown, 2),
        "num_trades": len(trades),
        "avg_pnl_pct": round(float(np.mean(pnl_pcts)), 2),
        "profit_factor": profit_factor,
    }
