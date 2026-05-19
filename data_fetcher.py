import yfinance as yf
import pandas as pd
from database import get_prices, upsert_prices, init_db

TICKER_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "AVAX": "AVAX-USD",
    "DOGE": "DOGE-USD",
}


def fetch_prices(asset: str, start: str, end: str) -> pd.DataFrame:
    init_db()
    cached = get_prices(asset, start, end)
    if not cached.empty:
        cached.columns = [c.capitalize() for c in cached.columns]
        return cached

    ticker = TICKER_MAP.get(asset, asset + "-USD")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    upsert_prices(df, asset)
    return df
