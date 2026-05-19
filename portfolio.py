import numpy as np
import pandas as pd
from data_fetcher import fetch_prices


def get_returns_matrix(assets: list, start: str, end: str) -> pd.DataFrame:
    returns = {}
    for asset in assets:
        df = fetch_prices(asset, start, end)
        if not df.empty:
            returns[asset] = df["Close"].pct_change().dropna()
    if not returns:
        return pd.DataFrame()
    return pd.DataFrame(returns).dropna()


def correlation_matrix(assets: list, start: str, end: str) -> pd.DataFrame:
    returns = get_returns_matrix(assets, start, end)
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()


def risk_return_stats(assets: list, start: str, end: str) -> pd.DataFrame:
    returns = get_returns_matrix(assets, start, end)
    if returns.empty:
        return pd.DataFrame()
    rows = []
    for asset in returns.columns:
        r = returns[asset]
        ann_ret = float(r.mean() * 252)
        ann_vol = float(r.std() * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        rows.append({
            "Asset": asset,
            "Annual Return (%)": round(ann_ret * 100, 2),
            "Annual Volatility (%)": round(ann_vol * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
        })
    return pd.DataFrame(rows).set_index("Asset")
