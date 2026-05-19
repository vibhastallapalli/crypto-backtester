import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "cryptobacktest.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            UNIQUE(asset, date)
        );

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            exit_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            signal_type TEXT NOT NULL,
            pnl REAL NOT NULL,
            pnl_pct REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            asset TEXT NOT NULL,
            weight REAL,
            daily_return REAL,
            UNIQUE(date, asset)
        );

        CREATE INDEX IF NOT EXISTS idx_prices_asset_date ON prices(asset, date);
        CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset);
        CREATE INDEX IF NOT EXISTS idx_portfolio_date ON portfolio(date);
    """)
    conn.commit()
    conn.close()


def upsert_prices(df: pd.DataFrame, asset: str):
    conn = get_connection()
    rows = [
        (asset, str(row.Index.date()), row.Open, row.High, row.Low, row.Close, row.Volume)
        for row in df.itertuples()
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO prices (asset, date, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()


def get_prices(asset: str, start: str, end: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices WHERE asset=? AND date>=? AND date<=? ORDER BY date",
        conn,
        params=(asset, start, end),
        parse_dates=["date"],
        index_col="date",
    )
    conn.close()
    return df


def save_trades(trades: list[dict]):
    if not trades:
        return
    conn = get_connection()
    conn.executemany(
        """INSERT INTO trades (asset, entry_date, exit_date, entry_price, exit_price, signal_type, pnl, pnl_pct)
           VALUES (:asset, :entry_date, :exit_date, :entry_price, :exit_price, :signal_type, :pnl, :pnl_pct)""",
        trades,
    )
    conn.commit()
    conn.close()


def get_trades(asset: str = None, start: str = None, end: str = None, win_loss: str = None) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM trades WHERE 1=1"
    params = []
    if asset:
        query += " AND asset=?"
        params.append(asset)
    if start:
        query += " AND entry_date>=?"
        params.append(start)
    if end:
        query += " AND exit_date<=?"
        params.append(end)
    if win_loss == "Win":
        query += " AND pnl>0"
    elif win_loss == "Loss":
        query += " AND pnl<=0"
    query += " ORDER BY entry_date DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def run_raw_sql(query: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


def upsert_portfolio(records: list[dict]):
    if not records:
        return
    conn = get_connection()
    conn.executemany(
        """INSERT OR REPLACE INTO portfolio (date, asset, weight, daily_return)
           VALUES (:date, :asset, :weight, :daily_return)""",
        records,
    )
    conn.commit()
    conn.close()
