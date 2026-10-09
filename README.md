# CryptoBacktest

A Streamlit app for comparing crypto trading strategies, reviewing saved trades, and exploring backtest results.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Price data is fetched through Yahoo Finance when it is not already cached. The app stores cached prices and trades in `cryptobacktest.db` beside `app.py`; this local database is ignored by Git. An internet connection is needed to fetch uncached prices.
