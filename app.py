import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize as scipy_minimize

from database import get_trades, init_db, run_raw_sql, save_trades
from data_fetcher import fetch_prices
from portfolio import correlation_matrix, risk_return_stats
from strategies import (
    BacktestResult,
    bollinger_bands,
    compute_metrics,
    ema_crossover,
    keltner_channel,
    ma_crossover,
    macd_crossover,
    rsi_mean_reversion,
    stochastic_oscillator,
    volume_breakout,
    vwap_crossover,
    zscore_mean_reversion,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoBacktest",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()

# ── Theme constants ────────────────────────────────────────────────────────────
BG = "#0a0e1a"
CARD = "#0d1526"
ACCENT = "#00d4aa"
BORDER = "#1a2744"
TEXT = "#e0e0e0"
MUTED = "#8892a0"
RED = "#ff4757"
YELLOW = "#ffd32a"

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=CARD,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="Space Mono, monospace", size=12),
    xaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False),
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

/* Global */
.stApp {{ background-color: {BG}; color: {TEXT}; }}
section[data-testid="stMain"] > div {{ padding-top: 1rem; }}

/* Headers */
h1, h2, h3,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    font-family: 'Space Mono', monospace !important;
    color: {ACCENT} !important;
    letter-spacing: -0.02em;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: {CARD};
    border-radius: 10px;
    padding: 4px 6px;
    border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {MUTED};
    border-radius: 7px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    padding: 6px 16px;
    border: none;
}}
.stTabs [aria-selected="true"] {{
    background: {ACCENT}22 !important;
    color: {ACCENT} !important;
}}

/* Buttons */
.stButton > button {{
    background: {ACCENT};
    color: {BG};
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 7px;
    padding: 8px 28px;
    font-size: 0.85rem;
    transition: background 0.15s;
}}
.stButton > button:hover {{ background: #00b899 !important; color: {BG} !important; }}

/* Metric cards */
.mc-wrap {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 10px 12px;
    text-align: center;
    height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}
.mc-label {{
    color: {MUTED};
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 5px;
    font-family: 'Space Mono', monospace;
}}
.mc-value {{
    font-family: 'Space Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1;
}}
.mc-pos {{ color: {ACCENT}; }}
.mc-neg {{ color: {RED}; }}
.mc-neutral {{ color: {TEXT}; }}

/* Inputs */
.stSelectbox label, .stMultiSelect label,
.stSlider label, .stTextArea label,
.stDateInput label {{ color: {MUTED} !important; font-size: 0.8rem; }}
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {CARD} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
}}
textarea {{ background: {CARD} !important; color: {TEXT} !important; }}

/* Divider */
hr {{ border-color: {BORDER}; }}

/* Dataframe */
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    f'<h1 style="margin-bottom:0;font-family:Space Mono,monospace;color:{ACCENT};">'
    "⬡ CryptoBacktest</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:{MUTED};margin-top:2px;margin-bottom:1rem;">'
    "Algorithmic crypto strategy backtester — powered by yfinance + SQLite</p>",
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE", "LTC", "DOT", "LINK", "MATIC", "UNI"]
STRATEGIES = ["MA Crossover", "EMA Crossover", "RSI Mean Reversion", "Stochastic Oscillator", "Volume Breakout", "MACD Crossover", "Bollinger Bands", "Z-Score MR", "Keltner Channel", "VWAP Crossover"]
D_START = datetime.date(2022, 1, 1)
D_END = datetime.date(2024, 12, 31)

EXAMPLE_QUERIES = {
    "── Select an example ──": "",
    "Recent trades (20)": "SELECT * FROM trades ORDER BY entry_date DESC LIMIT 20",
    "Best trades by PnL%": (
        "SELECT asset, signal_type, entry_date, exit_date, pnl_pct "
        "FROM trades WHERE pnl > 0 ORDER BY pnl_pct DESC LIMIT 10"
    ),
    "Worst trades by PnL%": (
        "SELECT asset, signal_type, entry_date, exit_date, pnl_pct "
        "FROM trades WHERE pnl < 0 ORDER BY pnl_pct ASC LIMIT 10"
    ),
    "Win rate by asset": (
        "SELECT asset, COUNT(*) AS total, "
        "SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) AS wins, "
        "ROUND(100.0*SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)/COUNT(*),2) AS win_rate_pct "
        "FROM trades GROUP BY asset ORDER BY win_rate_pct DESC"
    ),
    "Avg PnL by strategy": (
        "SELECT signal_type, COUNT(*) AS trades, "
        "ROUND(AVG(pnl_pct),2) AS avg_pnl_pct, "
        "ROUND(SUM(CASE WHEN pnl>0 THEN pnl_pct ELSE 0 END),2) AS gross_profit_pct "
        "FROM trades GROUP BY signal_type"
    ),
    "Latest cached prices": (
        "SELECT asset, MAX(date) AS last_date, "
        "ROUND(close,2) AS last_close FROM prices GROUP BY asset ORDER BY asset"
    ),
    "Price data (last 30 rows)": (
        "SELECT asset, date, ROUND(open,2) AS open, ROUND(high,2) AS high, "
        "ROUND(low,2) AS low, ROUND(close,2) AS close, "
        "CAST(volume AS INTEGER) AS volume "
        "FROM prices ORDER BY date DESC LIMIT 30"
    ),
    "Total PnL by asset": (
        "SELECT asset, ROUND(SUM(pnl_pct),2) AS total_pnl_pct, "
        "COUNT(*) AS trades FROM trades GROUP BY asset ORDER BY total_pnl_pct DESC"
    ),
    "Monthly trade count": (
        "SELECT SUBSTR(entry_date,1,7) AS month, COUNT(*) AS trades "
        "FROM trades GROUP BY month ORDER BY month"
    ),
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def metric_card(col, label: str, value, pct: bool = False, color: str = "neutral") -> None:
    val_str = f"{value}{'%' if pct else ''}"
    col.markdown(
        f'<div class="mc-wrap">'
        f'<div class="mc-label">{label}</div>'
        f'<div class="mc-value mc-{color}">{val_str}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _color_for(value, positive_good: bool = True) -> str:
    try:
        v = float(str(value).replace("%", "").replace("∞", "999"))
    except Exception:
        return "neutral"
    if v > 0:
        return "pos" if positive_good else "neg"
    if v < 0:
        return "neg" if positive_good else "pos"
    return "neutral"


def apply_plotly_layout(fig: go.Figure, **kwargs) -> go.Figure:
    layout = {**PLOTLY_BASE, **kwargs}
    fig.update_layout(**layout)
    return fig


def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = len(df)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if up > dn and up > 0 else 0.0
        minus_dm[i] = dn if dn > up and dn > 0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    tr[0] = high[0] - low[0]

    def _wilder(arr, p):
        out = np.zeros(n)
        if p - 1 < n:
            out[p - 1] = arr[:p].sum()
        for i in range(p, n):
            out[i] = out[i - 1] - out[i - 1] / p + arr[i]
        return out

    atr_s = _wilder(tr, period)
    plus_s = _wilder(plus_dm, period)
    minus_s = _wilder(minus_dm, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(atr_s > 0, 100.0 * plus_s / atr_s, 0.0)
        minus_di = np.where(atr_s > 0, 100.0 * minus_s / atr_s, 0.0)
        dx = np.where(plus_di + minus_di > 0,
                      100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
    adx = np.zeros(n)
    start = period * 2 - 2
    if start < n:
        adx[start] = dx[period - 1: period * 2 - 1].mean()
    for i in range(start + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return pd.Series(adx, index=df.index)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈  Strategy Backtester", "🗂  Portfolio Analyzer", "📋  Trade Log", "🔍  SQL Explorer", "⚡  Batch Backtest", "📡  Signal Scanner"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Strategy Backtester
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        asset = st.selectbox("Asset", ASSETS)
    with c2:
        date_range = st.date_input(
            "Date Range",
            value=(D_START, D_END),
            min_value=datetime.date(2018, 1, 1),
            max_value=datetime.date.today(),
            key="bt_dates",
        )
    with c3:
        strategy = st.selectbox("Strategy", STRATEGIES)

    # Dynamic params
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.8rem;margin:6px 0 2px;">Parameters</p>',
        unsafe_allow_html=True,
    )
    if strategy == "MA Crossover":
        p1, p2, p3 = st.columns([1, 1, 2])
        fast_ma = p1.slider("Fast MA", 2, 50, 10)
        slow_ma = p2.slider("Slow MA", 10, 200, 30)
        params = {"fast": fast_ma, "slow": slow_ma}
    elif strategy == "EMA Crossover":
        p1, p2, p3 = st.columns([1, 1, 2])
        fast_ema = p1.slider("Fast EMA", 2, 50, 9)
        slow_ema = p2.slider("Slow EMA", 5, 200, 21)
        params = {"fast": fast_ema, "slow": slow_ema}
    elif strategy == "RSI Mean Reversion":
        p1, p2, p3 = st.columns(3)
        rsi_period = p1.slider("RSI Period", 5, 50, 14)
        oversold = p2.slider("Oversold", 10, 45, 30)
        overbought = p3.slider("Overbought", 55, 90, 70)
        params = {"period": rsi_period, "oversold": oversold, "overbought": overbought}
    elif strategy == "Stochastic Oscillator":
        p1, p2, p3, p4 = st.columns(4)
        k_period = p1.slider("%K Period", 5, 50, 14)
        d_period = p2.slider("%D Period", 1, 10, 3)
        stoch_os = p3.slider("Oversold", 5, 40, 20)
        stoch_ob = p4.slider("Overbought", 60, 95, 80)
        params = {"k_period": k_period, "d_period": d_period, "oversold": stoch_os, "overbought": stoch_ob}
    elif strategy == "Volume Breakout":
        p1, p2, p3 = st.columns([1, 1, 2])
        vol_mult = p1.slider("Volume Multiplier", 1.0, 5.0, 2.0, 0.1)
        lookback = p2.slider("Lookback", 5, 60, 20)
        params = {"multiplier": vol_mult, "lookback": lookback}
    elif strategy == "MACD Crossover":
        p1, p2, p3 = st.columns(3)
        macd_fast = p1.slider("Fast EMA", 5, 50, 12)
        macd_slow = p2.slider("Slow EMA", 10, 100, 26)
        macd_sig = p3.slider("Signal EMA", 3, 30, 9)
        params = {"fast": macd_fast, "slow": macd_slow, "signal": macd_sig}
    elif strategy == "Bollinger Bands":
        p1, p2, p3 = st.columns([1, 1, 2])
        bb_period = p1.slider("Period", 5, 50, 20)
        bb_std = p2.slider("Std Dev", 1.0, 4.0, 2.0, 0.1)
        params = {"period": bb_period, "std_dev": bb_std}
    elif strategy == "Z-Score MR":
        p1, p2, p3 = st.columns([1, 1, 2])
        zs_window = p1.slider("Window", 5, 100, 20)
        zs_threshold = p2.slider("Threshold", 0.5, 4.0, 2.0, 0.1,
                                  help="Buy when Z-score < −threshold; sell when Z-score crosses 0.")
        params = {"window": zs_window, "threshold": zs_threshold}
    elif strategy == "Keltner Channel":
        p1, p2, p3 = st.columns([1, 1, 2])
        kc_ema_period = p1.slider("EMA Period", 5, 50, 20)
        kc_atr_mult = p2.slider("ATR Mult", 0.5, 4.0, 1.5, 0.1)
        params = {"ema_period": kc_ema_period, "atr_mult": kc_atr_mult}
    else:  # VWAP Crossover
        p1, p2, p3 = st.columns([1, 1, 2])
        vwap_window = p1.slider("VWAP Window", 5, 60, 20,
                                help="Rolling window (days) for volume-weighted average price calculation.")
        params = {"window": vwap_window}

    st.markdown(
        f'<p style="color:{MUTED};font-size:0.8rem;margin:6px 0 2px;">Risk Controls</p>',
        unsafe_allow_html=True,
    )
    rc1, rc2, rc3, rc4 = st.columns([1, 1, 1, 1])
    stop_loss_pct = rc1.slider("Stop Loss %", 0.0, 30.0, 0.0, 0.5,
                                help="0 = disabled. Exit trade if loss exceeds this %.")
    take_profit_pct = rc2.slider("Take Profit %", 0.0, 100.0, 0.0, 1.0,
                                  help="0 = disabled. Exit trade if gain exceeds this %.")
    atr_trail_mult = rc3.slider("ATR Trail Mult", 0.0, 5.0, 0.0, 0.5,
                                 help="0 = disabled. Trailing stop = price − mult × ATR. Overrides fixed stop when triggered.")
    atr_period = rc4.slider("ATR Period", 5, 30, 14,
                             help="Lookback period for ATR calculation.")

    st.markdown("")
    run_btn = st.button("▶  Run Backtest", key="run_bt")

    if run_btn:
        if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
            st.error("Select a start AND end date.")
            st.stop()

        start_s, end_s = str(date_range[0]), str(date_range[1])

        with st.spinner(f"Fetching {asset} price data…"):
            df = fetch_prices(asset, start_s, end_s)

        if df.empty:
            st.error(f"No data returned for {asset}. Try a different date range.")
            st.stop()

        with st.spinner("Running backtest…"):
            sl_tp = {"stop_loss": stop_loss_pct, "take_profit": take_profit_pct,
                     "atr_trail_mult": atr_trail_mult, "atr_period": atr_period}
            if strategy == "MA Crossover":
                result: BacktestResult = ma_crossover(df, asset, **params, **sl_tp)
            elif strategy == "EMA Crossover":
                result = ema_crossover(df, asset, **params, **sl_tp)
            elif strategy == "RSI Mean Reversion":
                result = rsi_mean_reversion(df, asset, **params, **sl_tp)
            elif strategy == "Stochastic Oscillator":
                result = stochastic_oscillator(df, asset, **params, **sl_tp)
            elif strategy == "Volume Breakout":
                result = volume_breakout(df, asset, **params, **sl_tp)
            elif strategy == "MACD Crossover":
                result = macd_crossover(df, asset, **params, **sl_tp)
            elif strategy == "Bollinger Bands":
                result = bollinger_bands(df, asset, **params, **sl_tp)
            elif strategy == "Z-Score MR":
                result = zscore_mean_reversion(df, asset, **params, **sl_tp)
            elif strategy == "Keltner Channel":
                result = keltner_channel(df, asset, **params, **sl_tp)
            else:
                result = vwap_crossover(df, asset, **params, **sl_tp)

        if result.trades:
            save_trades(result.trades)

        st.session_state.update(
            bt_result=result,
            bt_metrics=compute_metrics(result),
            bt_df=df,
            bt_asset=asset,
            bt_strategy=strategy,
            bt_params=params,
        )
        st.session_state["bt_risk"] = {
            "stop_loss": stop_loss_pct,
            "take_profit": take_profit_pct,
            "atr_trail_mult": atr_trail_mult,
            "atr_period": atr_period,
        }
        for _k in ("mc_curves", "mc_pnl", "wf_rows", "hm_data", "adv_beta"):
            st.session_state.pop(_k, None)

    # ── Results ────────────────────────────────────────────────────────────────
    if "bt_result" in st.session_state:
        result: BacktestResult = st.session_state["bt_result"]
        metrics: dict = st.session_state["bt_metrics"]
        df: pd.DataFrame = st.session_state["bt_df"]
        cur_asset: str = st.session_state["bt_asset"]
        cur_strat: str = st.session_state["bt_strategy"]
        cur_params: dict = st.session_state["bt_params"]

        st.divider()

        # Metric cards — row 1
        cols = st.columns(5)
        metric_card(cols[0], "Total Return", metrics["total_return"], pct=True,
                    color=_color_for(metrics["total_return"]))
        metric_card(cols[1], "Win Rate", metrics["win_rate"], pct=True,
                    color=_color_for(metrics["win_rate"]))
        metric_card(cols[2], "Sharpe Ratio", metrics["sharpe"],
                    color=_color_for(metrics["sharpe"]))
        metric_card(cols[3], "Sortino Ratio", metrics["sortino"],
                    color=_color_for(metrics["sortino"]))
        metric_card(cols[4], "Calmar Ratio", metrics["calmar"],
                    color=_color_for(metrics["calmar"]))
        st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
        # Metric cards — row 2
        cols2 = st.columns(4)
        metric_card(cols2[0], "Max Drawdown", metrics["max_drawdown"], pct=True,
                    color="neg" if metrics["max_drawdown"] < 0 else "neutral")
        metric_card(cols2[1], "# Trades", metrics["num_trades"], color="neutral")
        metric_card(cols2[2], "Avg PnL%", metrics["avg_pnl_pct"], pct=True,
                    color=_color_for(metrics["avg_pnl_pct"]))
        metric_card(cols2[3], "Profit Factor", metrics["profit_factor"],
                    color=_color_for(metrics["profit_factor"]))

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        # ── Price chart with signals ───────────────────────────────────────────
        sig_df = result.signals
        buys = sig_df[sig_df["signal"] == "buy"]
        sells = sig_df[sig_df["signal"] == "sell"]

        vol_colors = [
            ACCENT if df["Close"].iloc[i] >= df["Open"].iloc[i] else RED
            for i in range(len(df))
        ]

        fig_p = go.Figure()
        fig_p.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price",
            increasing=dict(line=dict(color=ACCENT, width=1), fillcolor="rgba(0,212,170,0.55)"),
            decreasing=dict(line=dict(color=RED, width=1), fillcolor="rgba(255,71,87,0.55)"),
        ))

        if cur_strat == "Stochastic Oscillator" and "stoch_k" in sig_df.columns:
            scale = sig_df["Close"].max() / 100
            mid = sig_df["Close"].min()
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=mid + sig_df["stoch_k"] * scale * 0.3,
                mode="lines", name="%K (scaled)",
                line=dict(color=YELLOW, width=1, dash="dot"), opacity=0.7,
            ))
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=mid + sig_df["stoch_d"] * scale * 0.3,
                mode="lines", name="%D (scaled)",
                line=dict(color="#a29bfe", width=1, dash="dot"), opacity=0.7,
            ))
        elif cur_strat == "EMA Crossover" and "fast_ema" in sig_df.columns:
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["fast_ema"],
                mode="lines", name=f"Fast EMA ({cur_params['fast']})",
                line=dict(color=YELLOW, width=1.2, dash="dot"),
            ))
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["slow_ema"],
                mode="lines", name=f"Slow EMA ({cur_params['slow']})",
                line=dict(color="#ff6b81", width=1.2, dash="dot"),
            ))
        elif cur_strat == "MA Crossover" and "fast_ma" in sig_df.columns:
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["fast_ma"],
                mode="lines", name=f"Fast MA ({cur_params['fast']})",
                line=dict(color=YELLOW, width=1.2, dash="dot"),
            ))
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["slow_ma"],
                mode="lines", name=f"Slow MA ({cur_params['slow']})",
                line=dict(color="#ff6b81", width=1.2, dash="dot"),
            ))
        elif cur_strat == "RSI Mean Reversion" and "rsi" in sig_df.columns:
            fig_p.add_trace(go.Scatter(
                x=sig_df.index,
                y=sig_df["rsi"] / 100 * sig_df["Close"].max(),
                mode="lines", name="RSI (scaled)",
                line=dict(color="#a29bfe", width=1, dash="dot"),
                opacity=0.6,
            ))
        elif cur_strat == "Bollinger Bands" and "bb_upper" in sig_df.columns:
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["bb_upper"],
                mode="lines", name="Upper Band",
                line=dict(color=YELLOW, width=1, dash="dot"), opacity=0.7,
            ))
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["bb_mid"],
                mode="lines", name="Mid Band",
                line=dict(color=MUTED, width=1, dash="dot"), opacity=0.5,
            ))
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["bb_lower"],
                mode="lines", name="Lower Band",
                line=dict(color="#ff6b81", width=1, dash="dot"), opacity=0.7,
                fill="tonexty", fillcolor="rgba(255,107,129,0.04)",
            ))
        elif cur_strat == "MACD Crossover" and "macd" in sig_df.columns:
            scale = sig_df["Close"].max() / (sig_df["macd"].abs().max() or 1)
            mid = sig_df["Close"].mean()
            fig_p.add_trace(go.Scatter(
                x=sig_df.index,
                y=mid + sig_df["macd"] * scale * 0.15,
                mode="lines", name="MACD (scaled)",
                line=dict(color="#fd79a8", width=1, dash="dot"),
                opacity=0.6,
            ))
            fig_p.add_trace(go.Scatter(
                x=sig_df.index,
                y=mid + sig_df["macd_signal"] * scale * 0.15,
                mode="lines", name="Signal (scaled)",
                line=dict(color="#fdcb6e", width=1, dash="dot"),
                opacity=0.6,
            ))
        elif cur_strat == "Z-Score MR" and "zscore" in sig_df.columns:
            price_mid = sig_df["Close"].mean()
            price_range = sig_df["Close"].max() - sig_df["Close"].min()
            zs_scale = price_range / (sig_df["zscore"].abs().max() * 4 or 1)
            fig_p.add_trace(go.Scatter(
                x=sig_df.index,
                y=price_mid + sig_df["zscore"] * zs_scale,
                mode="lines", name="Z-Score (scaled)",
                line=dict(color="#a29bfe", width=1, dash="dot"),
                opacity=0.65,
            ))
        elif cur_strat == "Keltner Channel" and "kc_upper" in sig_df.columns:
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["kc_upper"],
                mode="lines", name="KC Upper",
                line=dict(color=YELLOW, width=1, dash="dot"), opacity=0.7,
            ))
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["kc_mid"],
                mode="lines", name="KC Mid (EMA)",
                line=dict(color=MUTED, width=1, dash="dot"), opacity=0.5,
            ))
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["kc_lower"],
                mode="lines", name="KC Lower",
                line=dict(color="#ff6b81", width=1, dash="dot"), opacity=0.7,
                fill="tonexty", fillcolor="rgba(255,107,129,0.04)",
            ))
        elif cur_strat == "VWAP Crossover" and "vwap" in sig_df.columns:
            fig_p.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df["vwap"],
                mode="lines", name=f"VWAP ({cur_params['window']}d)",
                line=dict(color="#a29bfe", width=1.5, dash="dot"),
                opacity=0.85,
            ))

        if not buys.empty:
            fig_p.add_trace(go.Scatter(
                x=buys.index, y=buys["Low"] * 0.985,
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=11, color=ACCENT,
                            line=dict(color="white", width=1)),
            ))
        if not sells.empty:
            fig_p.add_trace(go.Scatter(
                x=sells.index, y=sells["High"] * 1.015,
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=11, color=RED,
                            line=dict(color="white", width=1)),
            ))

        fig_p.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            marker_line_width=0,
            opacity=0.45,
            yaxis="y2",
        ))

        fig_p.update_layout(
            **{k: v for k, v in PLOTLY_BASE.items() if k not in ("xaxis", "yaxis", "margin")},
            title=f"{cur_asset} — {cur_strat}",
            xaxis=dict(
                gridcolor=BORDER, showgrid=True, zeroline=False,
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                gridcolor=BORDER, showgrid=True, zeroline=False,
                title="Price (USD)", domain=[0.28, 1.0],
            ),
            yaxis2=dict(
                gridcolor=BORDER, showgrid=False, zeroline=False,
                title="Volume", domain=[0.0, 0.24],
                tickfont=dict(color=MUTED, size=10),
                titlefont=dict(color=MUTED, size=11),
            ),
            height=540,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # ── Equity curve vs Buy & Hold ─────────────────────────────────────────
        equity = result.equity_curve
        common_start = equity.index[0]
        bh_prices = df["Close"].loc[common_start:]
        bh = (bh_prices / float(bh_prices.iloc[0])).reindex(equity.index, method="ffill")

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity.index, y=equity,
            mode="lines", name="Strategy",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
        ))
        fig_eq.add_trace(go.Scatter(
            x=bh.index, y=bh,
            mode="lines", name="Buy & Hold",
            line=dict(color=MUTED, width=1.5, dash="dash"),
        ))
        fig_eq.add_hline(y=1.0, line_dash="dot", line_color=BORDER, opacity=0.6)
        apply_plotly_layout(
            fig_eq,
            title="Equity Curve vs Buy & Hold",
            yaxis_title="Portfolio Value (normalized to 1)",
            height=340,
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Drawdown chart ─────────────────────────────────────────────────────
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown,
            mode="lines", name="Drawdown",
            line=dict(color=RED, width=1.5),
            fill="tozeroy", fillcolor="rgba(255,71,87,0.12)",
        ))
        fig_dd.add_hline(
            y=float(drawdown.min()), line_dash="dot", line_color=YELLOW,
            annotation_text=f"Max DD: {drawdown.min():.2f}%",
            annotation_font_color=YELLOW, annotation_position="bottom right",
        )
        apply_plotly_layout(
            fig_dd,
            title="Drawdown (%)",
            yaxis_title="Drawdown (%)",
            height=220,
            margin=dict(l=50, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Rolling Sharpe ratio ───────────────────────────────────────────────
        window = 30
        daily_rets = equity.pct_change().dropna()
        rolling_mean = daily_rets.rolling(window).mean()
        rolling_std = daily_rets.rolling(window).std()
        rolling_sharpe = (rolling_mean * 252) / (rolling_std * np.sqrt(252))
        rolling_sharpe = rolling_sharpe.dropna()

        if len(rolling_sharpe) > 1:
            rs_colors = [ACCENT if v >= 0 else RED for v in rolling_sharpe]
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Scatter(
                x=rolling_sharpe.index, y=rolling_sharpe,
                mode="lines", name=f"{window}d Rolling Sharpe",
                line=dict(color=ACCENT, width=1.5),
                fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
            ))
            fig_rs.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.5)
            fig_rs.add_hline(y=1, line_dash="dash", line_color=YELLOW, opacity=0.6,
                             annotation_text="Sharpe = 1", annotation_font_color=YELLOW,
                             annotation_position="bottom right")
            apply_plotly_layout(
                fig_rs,
                title=f"Rolling {window}-Day Sharpe Ratio",
                yaxis_title="Sharpe Ratio",
                height=240,
                margin=dict(l=50, r=20, t=45, b=35),
            )
            st.plotly_chart(fig_rs, use_container_width=True)

        # ── Advanced Risk Analytics ────────────────────────────────────────────
        with st.expander("📊  Advanced Risk Analytics"):
            adv_rets = equity.pct_change().dropna()
            var_95 = float(np.percentile(adv_rets, 5))
            cvar_95 = float(adv_rets[adv_rets <= var_95].mean())
            pos_sum = adv_rets[adv_rets > 0].sum()
            neg_sum = adv_rets[adv_rets < 0].abs().sum()
            omega = float(pos_sum / neg_sum) if neg_sum > 0 else float("inf")

            adv_c1, adv_c2, adv_c3 = st.columns(3)
            metric_card(adv_c1, "Daily VaR 95%", f"{var_95 * 100:.2f}%",
                        color="neg" if var_95 < 0 else "neutral")
            metric_card(adv_c2, "Daily CVaR 95%", f"{cvar_95 * 100:.2f}%",
                        color="neg" if cvar_95 < 0 else "neutral")
            metric_card(adv_c3, "Omega Ratio", f"{omega:.2f}",
                        color=_color_for(omega - 1))

            if cur_asset != "BTC":
                beta_btn = st.button("▶  Compute Rolling Beta vs BTC (60d)", key="beta_btn")
                if beta_btn:
                    btc_df_beta = fetch_prices("BTC", str(df.index[0].date()), str(df.index[-1].date()))
                    if not btc_df_beta.empty:
                        btc_rets = btc_df_beta["Close"].pct_change().dropna()
                        common_idx = adv_rets.index.intersection(btc_rets.index)
                        if len(common_idx) > 60:
                            strat_al = adv_rets.reindex(common_idx)
                            btc_al = btc_rets.reindex(common_idx)
                            r_beta = (strat_al.rolling(60).cov(btc_al) / btc_al.rolling(60).var()).dropna()
                            st.session_state["adv_beta"] = r_beta

                if "adv_beta" in st.session_state:
                    r_beta = st.session_state["adv_beta"]
                    fig_beta = go.Figure()
                    fig_beta.add_trace(go.Scatter(
                        x=r_beta.index, y=r_beta,
                        mode="lines", name="60d Rolling Beta vs BTC",
                        line=dict(color=YELLOW, width=1.5),
                    ))
                    fig_beta.add_hline(y=1.0, line_dash="dash", line_color=ACCENT, opacity=0.6,
                                       annotation_text="Beta = 1", annotation_font_color=ACCENT,
                                       annotation_position="top right")
                    fig_beta.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.5)
                    apply_plotly_layout(
                        fig_beta,
                        title=f"Rolling 60-Day Beta — {cur_asset} Strategy vs BTC",
                        yaxis_title="Beta",
                        height=260,
                        margin=dict(l=50, r=20, t=45, b=35),
                    )
                    st.plotly_chart(fig_beta, use_container_width=True)
            else:
                st.info("Asset is BTC — beta vs BTC is 1 by definition.")

        # ── Market Regime Analysis ─────────────────────────────────────────────
        with st.expander("🔭  Market Regime Analysis"):
            ma200 = df["Close"].rolling(200, min_periods=50).mean()
            adx_s = _calc_adx(df)

            regime = pd.Series("Sideways", index=df.index)
            regime[(df["Close"] > ma200) & (adx_s > 25)] = "Bull"
            regime[(df["Close"] < ma200) & (adx_s > 25)] = "Bear"

            _reg_fill = {"Bull": "rgba(0,212,170,0.13)", "Bear": "rgba(255,71,87,0.13)",
                         "Sideways": "rgba(136,146,160,0.07)"}

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=df.index, y=df["Close"],
                mode="lines", name="Price",
                line=dict(color=TEXT, width=1.5),
            ))
            fig_reg.add_trace(go.Scatter(
                x=df.index, y=ma200,
                mode="lines", name="200-day MA",
                line=dict(color=YELLOW, width=1.2, dash="dot"),
            ))

            prev_reg = None
            seg_start = None
            for dt, reg in regime.items():
                if reg != prev_reg:
                    if prev_reg is not None:
                        fig_reg.add_vrect(x0=seg_start, x1=dt,
                                          fillcolor=_reg_fill[prev_reg], line_width=0, layer="below")
                    seg_start = dt
                    prev_reg = reg
            if prev_reg is not None:
                fig_reg.add_vrect(x0=seg_start, x1=df.index[-1],
                                  fillcolor=_reg_fill[prev_reg], line_width=0, layer="below")

            apply_plotly_layout(
                fig_reg,
                title=f"{cur_asset} — Market Regime  (green=Bull · red=Bear · grey=Sideways)",
                yaxis_title="Price (USD)",
                height=360,
                margin=dict(l=50, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_reg, use_container_width=True)

            rc1, rc2, rc3 = st.columns(3)
            for col, name, color in [(rc1, "Bull", "pos"), (rc2, "Bear", "neg"), (rc3, "Sideways", "neutral")]:
                days = int((regime == name).sum())
                metric_card(col, f"{name} Days", days, color=color)

            if result.trades:
                trade_regime_rows = []
                for t in result.trades:
                    ed = pd.to_datetime(t["entry_date"])
                    reg_label = regime.get(ed, "Unknown") if ed in regime.index else "Unknown"
                    trade_regime_rows.append({**t, "regime": reg_label})
                trd_df = pd.DataFrame(trade_regime_rows)
                reg_summary = []
                for rname in ["Bull", "Bear", "Sideways"]:
                    sub = trd_df[trd_df["regime"] == rname]
                    if sub.empty:
                        continue
                    wins = int((sub["pnl"] > 0).sum())
                    reg_summary.append({
                        "Regime": rname,
                        "# Trades": len(sub),
                        "Win Rate %": round(wins / len(sub) * 100, 1),
                        "Avg PnL %": round(float(sub["pnl_pct"].mean()), 2),
                        "Total PnL %": round(float(sub["pnl_pct"].sum()), 2),
                    })
                if reg_summary:
                    def _rc(val):
                        if isinstance(val, (int, float)):
                            return f"color: {ACCENT}" if val > 0 else f"color: {RED}"
                        return ""
                    st.markdown(
                        f'<p style="color:{MUTED};font-size:0.82rem;margin-top:10px;">'
                        "Trade performance split by detected market regime:</p>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        pd.DataFrame(reg_summary).style.applymap(_rc, subset=["Avg PnL %", "Total PnL %"]),
                        use_container_width=True,
                    )

        # ── Per-trade PnL bars + cumulative line ──────────────────────────────
        if result.trades:
            pnl_vals = [t["pnl_pct"] for t in result.trades]
            trade_nums = list(range(1, len(pnl_vals) + 1))
            cum_pnl = list(np.cumsum(pnl_vals))
            bar_colors = [ACCENT if v >= 0 else RED for v in pnl_vals]

            fig_seq = go.Figure()
            fig_seq.add_trace(go.Bar(
                x=trade_nums, y=pnl_vals,
                marker_color=bar_colors,
                marker_line=dict(color=BORDER, width=0.4),
                name="Trade PnL %",
            ))
            fig_seq.add_trace(go.Scatter(
                x=trade_nums, y=cum_pnl,
                mode="lines", name="Cumulative PnL %",
                line=dict(color=YELLOW, width=2),
                yaxis="y2",
            ))
            fig_seq.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.5)
            apply_plotly_layout(
                fig_seq,
                title="Per-Trade PnL % with Cumulative (right axis)",
                xaxis_title="Trade #",
                yaxis_title="PnL %",
                yaxis2=dict(
                    title="Cumulative PnL %",
                    overlaying="y", side="right",
                    gridcolor=BORDER, showgrid=False,
                    tickfont=dict(color=YELLOW),
                    titlefont=dict(color=YELLOW),
                ),
                height=280,
                margin=dict(l=50, r=60, t=45, b=35),
            )
            st.plotly_chart(fig_seq, use_container_width=True)

        # ── Trade PnL distribution ─────────────────────────────────────────────
        if result.trades:
            pnl_vals = [t["pnl_pct"] for t in result.trades]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=pnl_vals,
                nbinsx=max(10, len(pnl_vals) // 3),
                marker_color=[ACCENT if v >= 0 else RED for v in pnl_vals],
                marker_line=dict(color=BORDER, width=0.5),
                name="Trade PnL %",
            ))
            fig_hist.add_vline(x=0, line_dash="dot", line_color=MUTED, opacity=0.7)
            mean_pnl = float(np.mean(pnl_vals))
            fig_hist.add_vline(
                x=mean_pnl, line_dash="dash", line_color=YELLOW,
                annotation_text=f"Mean: {mean_pnl:.2f}%",
                annotation_font_color=YELLOW, annotation_position="top right",
            )
            apply_plotly_layout(
                fig_hist,
                title="Trade PnL Distribution (%)",
                xaxis_title="PnL %",
                yaxis_title="Count",
                height=240,
                margin=dict(l=50, r=20, t=45, b=35),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Monte Carlo Simulation ─────────────────────────────────────────────
        if result.trades and len(result.trades) >= 2:
            mc_btn = st.button("▶  Run Monte Carlo (1000 sims)", key="mc_btn")
            if mc_btn:
                mc_pnl = np.array([t["pnl_pct"] / 100 for t in result.trades])
                mc_rng = np.random.default_rng(42)
                mc_curves = np.zeros((1000, len(mc_pnl) + 1))
                mc_curves[:, 0] = 1.0
                for mc_i in range(1000):
                    mc_shuf = mc_rng.permutation(mc_pnl)
                    mc_curves[mc_i, 1:] = np.cumprod(1 + mc_shuf)
                st.session_state["mc_curves"] = mc_curves
                st.session_state["mc_pnl"] = mc_pnl
            if "mc_curves" in st.session_state and "mc_pnl" in st.session_state:
                mc_cv = st.session_state["mc_curves"]
                mc_pa = st.session_state["mc_pnl"]
                mc_x = list(range(mc_cv.shape[1]))
                mc_p5 = np.percentile(mc_cv, 5, axis=0)
                mc_p95 = np.percentile(mc_cv, 95, axis=0)
                mc_med = np.percentile(mc_cv, 50, axis=0)
                mc_act = np.concatenate([[1.0], np.cumprod(1 + mc_pa)])
                fig_mc = go.Figure()
                for mc_i in range(0, len(mc_cv), 5):
                    fig_mc.add_trace(go.Scatter(
                        x=mc_x, y=mc_cv[mc_i].tolist(),
                        mode="lines",
                        line=dict(color="rgba(0,212,170,0.04)", width=1),
                        showlegend=False, hoverinfo="skip",
                    ))
                fig_mc.add_trace(go.Scatter(
                    x=mc_x + mc_x[::-1],
                    y=mc_p95.tolist() + mc_p5.tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(0,212,170,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="5th–95th %ile",
                ))
                fig_mc.add_trace(go.Scatter(
                    x=mc_x, y=mc_med.tolist(),
                    mode="lines", name="Median sim",
                    line=dict(color=YELLOW, width=1.5, dash="dash"),
                ))
                fig_mc.add_trace(go.Scatter(
                    x=mc_x, y=mc_act.tolist(),
                    mode="lines", name="Actual sequence",
                    line=dict(color=ACCENT, width=2.5),
                ))
                fig_mc.add_hline(y=1.0, line_dash="dot", line_color=BORDER, opacity=0.6)
                apply_plotly_layout(
                    fig_mc,
                    title="Monte Carlo — 1000 Trade-Sequence Shuffles",
                    xaxis_title="Trade #",
                    yaxis_title="Portfolio Value (normalized)",
                    height=400,
                    margin=dict(l=50, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_mc, use_container_width=True)
                mc_finals = mc_cv[:, -1]
                mc_p50v = float(np.percentile(mc_finals, 50))
                mc_p5v = float(np.percentile(mc_finals, 5))
                mc_p95v = float(np.percentile(mc_finals, 95))
                mc_col1, mc_col2, mc_col3 = st.columns(3)
                metric_card(mc_col1, "Median Final Equity", round(mc_p50v, 3),
                            color=_color_for(mc_p50v - 1))
                metric_card(mc_col2, "P5 Final Equity", round(mc_p5v, 3),
                            color=_color_for(mc_p5v - 1))
                metric_card(mc_col3, "P95 Final Equity", round(mc_p95v, 3),
                            color=_color_for(mc_p95v - 1))

        # ── Walk-Forward Analysis ──────────────────────────────────────────────
        wf_fn_map = {
            "MA Crossover": ma_crossover,
            "EMA Crossover": ema_crossover,
            "RSI Mean Reversion": rsi_mean_reversion,
            "Stochastic Oscillator": stochastic_oscillator,
            "Volume Breakout": volume_breakout,
            "MACD Crossover": macd_crossover,
            "Bollinger Bands": bollinger_bands,
            "Z-Score MR": zscore_mean_reversion,
            "Keltner Channel": keltner_channel,
            "VWAP Crossover": vwap_crossover,
        }
        wf_btn = st.button("▶  Walk-Forward Analysis (5 windows)", key="wf_btn")
        if wf_btn:
            wf_fn = wf_fn_map.get(cur_strat)
            if wf_fn:
                wf_risk = st.session_state.get("bt_risk", {})
                n_wf = len(df)
                chunk = n_wf // 5
                wf_rows = []
                for wfi in range(5):
                    s_i = wfi * chunk
                    e_i = (wfi + 1) * chunk if wfi < 4 else n_wf
                    wdf = df.iloc[s_i:e_i]
                    if len(wdf) < 20:
                        continue
                    try:
                        wr = wf_fn(wdf, cur_asset, **cur_params, **wf_risk)
                        wm = compute_metrics(wr)
                        wf_rows.append({
                            "Window": f"W{wfi+1} ({str(wdf.index[0].date())}–{str(wdf.index[-1].date())})",
                            "Start": str(wdf.index[0].date()),
                            "End": str(wdf.index[-1].date()),
                            "Return %": wm["total_return"],
                            "Sharpe": wm["sharpe"],
                            "# Trades": wm["num_trades"],
                        })
                    except Exception:
                        pass
                st.session_state["wf_rows"] = wf_rows

        if "wf_rows" in st.session_state and st.session_state["wf_rows"]:
            wf_rows = st.session_state["wf_rows"]
            wf_labels = [r["Window"] for r in wf_rows]
            wf_sharpe = [r["Sharpe"] for r in wf_rows]
            wf_colors = [ACCENT if s >= 0 else RED for s in wf_sharpe]

            fig_wf = go.Figure(go.Bar(
                x=wf_labels, y=wf_sharpe,
                marker_color=wf_colors,
                marker_line=dict(color=BORDER, width=0.5),
                text=[f"{s:.2f}" for s in wf_sharpe],
                textposition="outside",
                textfont=dict(color=TEXT),
                name="Sharpe",
            ))
            fig_wf.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.5)
            fig_wf.add_hline(y=1.0, line_dash="dash", line_color=YELLOW, opacity=0.6,
                             annotation_text="Sharpe = 1", annotation_font_color=YELLOW,
                             annotation_position="top right")
            apply_plotly_layout(
                fig_wf,
                title=f"Walk-Forward Sharpe — {cur_strat} on {cur_asset}",
                yaxis_title="Sharpe Ratio",
                height=320,
                margin=dict(l=50, r=20, t=50, b=70),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            wf_display = pd.DataFrame([{
                "Window": r["Window"],
                "Start": r["Start"],
                "End": r["End"],
                "Return %": r["Return %"],
                "Sharpe": r["Sharpe"],
                "# Trades": r["# Trades"],
            } for r in wf_rows])

            def _wf_color(val):
                if not isinstance(val, (int, float)):
                    return ""
                return f"color: {ACCENT}" if val > 0 else f"color: {RED}"

            st.dataframe(
                wf_display.style
                .applymap(_wf_color, subset=["Return %", "Sharpe"])
                .format({"Return %": "{:.2f}%", "Sharpe": "{:.2f}"}),
                use_container_width=True,
            )

        # ── Parameter Heatmap ─────────────────────────────────────────────────
        HM_PARAM_SPECS = {
            "MA Crossover": {
                "fast": {"min": 2, "max": 50, "type": int},
                "slow": {"min": 10, "max": 200, "type": int},
            },
            "EMA Crossover": {
                "fast": {"min": 2, "max": 50, "type": int},
                "slow": {"min": 5, "max": 200, "type": int},
            },
            "RSI Mean Reversion": {
                "period": {"min": 5, "max": 50, "type": int},
                "oversold": {"min": 10, "max": 45, "type": int},
                "overbought": {"min": 55, "max": 90, "type": int},
            },
            "Stochastic Oscillator": {
                "k_period": {"min": 5, "max": 50, "type": int},
                "d_period": {"min": 1, "max": 10, "type": int},
                "oversold": {"min": 5, "max": 40, "type": int},
                "overbought": {"min": 60, "max": 95, "type": int},
            },
            "Volume Breakout": {
                "multiplier": {"min": 1.0, "max": 5.0, "type": float},
                "lookback": {"min": 5, "max": 60, "type": int},
            },
            "MACD Crossover": {
                "fast": {"min": 5, "max": 50, "type": int},
                "slow": {"min": 10, "max": 100, "type": int},
                "signal": {"min": 3, "max": 30, "type": int},
            },
            "Bollinger Bands": {
                "period": {"min": 5, "max": 50, "type": int},
                "std_dev": {"min": 1.0, "max": 4.0, "type": float},
            },
            "Z-Score MR": {
                "window": {"min": 5, "max": 100, "type": int},
                "threshold": {"min": 0.5, "max": 4.0, "type": float},
            },
            "Keltner Channel": {
                "ema_period": {"min": 5, "max": 50, "type": int},
                "atr_mult": {"min": 0.5, "max": 4.0, "type": float},
            },
            "VWAP Crossover": {
                "window": {"min": 5, "max": 60, "type": int},
            },
        }
        with st.expander("🔬  Parameter Heatmap"):
            hm_spec = HM_PARAM_SPECS.get(cur_strat, {})
            hm_keys = list(hm_spec.keys())
            hm_fn = wf_fn_map.get(cur_strat)
            hm_risk = st.session_state.get("bt_risk", {})

            if len(hm_keys) < 2:
                if hm_keys and hm_fn:
                    pk = hm_keys[0]
                    spec0 = hm_spec[pk]
                    if st.button("▶  Run Line Scan", key="hm_btn_line"):
                        hm_vals = np.linspace(spec0["min"], spec0["max"], 20)
                        if spec0["type"] == int:
                            hm_vals = np.unique(np.round(hm_vals).astype(int)).astype(float)
                        sharpes_line = []
                        for hv in hm_vals:
                            tp = dict(cur_params)
                            tp[pk] = spec0["type"](hv)
                            try:
                                sharpes_line.append(compute_metrics(hm_fn(df, cur_asset, **tp, **hm_risk))["sharpe"])
                            except Exception:
                                sharpes_line.append(float("nan"))
                        st.session_state["hm_data"] = {
                            "type": "line", "x": hm_vals.tolist(), "y": sharpes_line, "xlabel": pk,
                        }
            else:
                hm_c1, hm_c2 = st.columns(2)
                hm_p1 = hm_c1.selectbox("X-axis param", hm_keys, key="hm_p1")
                hm_p2 = hm_c2.selectbox("Y-axis param", [k for k in hm_keys if k != hm_p1], key="hm_p2")
                if st.button("▶  Run Heatmap (10 × 10 grid)", key="hm_btn") and hm_fn:
                    sp1, sp2 = hm_spec[hm_p1], hm_spec[hm_p2]
                    x_raw = np.linspace(sp1["min"], sp1["max"], 10)
                    y_raw = np.linspace(sp2["min"], sp2["max"], 10)
                    if sp1["type"] == int:
                        x_raw = np.unique(np.round(x_raw).astype(int)).astype(float)
                    if sp2["type"] == int:
                        y_raw = np.unique(np.round(y_raw).astype(int)).astype(float)
                    x_labels = [str(int(v)) if sp1["type"] == int else f"{v:.2f}" for v in x_raw]
                    y_labels = [str(int(v)) if sp2["type"] == int else f"{v:.2f}" for v in y_raw]
                    hm_grid = np.full((len(y_raw), len(x_raw)), np.nan)
                    total_hm = len(x_raw) * len(y_raw)
                    prog_hm = st.progress(0, text="Running grid…")
                    cell_hm = 0
                    for yi, yv in enumerate(y_raw):
                        for xi, xv in enumerate(x_raw):
                            tp = dict(cur_params)
                            tp[hm_p1] = sp1["type"](xv)
                            tp[hm_p2] = sp2["type"](yv)
                            try:
                                hm_grid[yi, xi] = compute_metrics(
                                    hm_fn(df, cur_asset, **tp, **hm_risk)
                                )["sharpe"]
                            except Exception:
                                pass
                            cell_hm += 1
                            prog_hm.progress(cell_hm / total_hm,
                                             text=f"Running grid… {cell_hm}/{total_hm}")
                    prog_hm.empty()
                    st.session_state["hm_data"] = {
                        "type": "heatmap",
                        "z": hm_grid.tolist(),
                        "x": x_labels, "y": y_labels,
                        "xlabel": hm_p1, "ylabel": hm_p2,
                    }

            if "hm_data" in st.session_state:
                hm_d = st.session_state["hm_data"]
                if hm_d["type"] == "heatmap":
                    hm_text = [
                        [f"{v:.2f}" if v == v else "—" for v in row]
                        for row in hm_d["z"]
                    ]
                    fig_hm = go.Figure(go.Heatmap(
                        z=hm_d["z"],
                        x=hm_d["x"],
                        y=hm_d["y"],
                        colorscale=[[0, RED], [0.5, BG], [1, ACCENT]],
                        zmid=0,
                        text=hm_text,
                        texttemplate="%{text}",
                        textfont=dict(size=10, color=TEXT),
                        showscale=True,
                        colorbar=dict(tickfont=dict(color=TEXT), outlinecolor=BORDER, title="Sharpe"),
                    ))
                    apply_plotly_layout(
                        fig_hm,
                        title=f"Sharpe Heatmap — {cur_strat} on {cur_asset}",
                        xaxis_title=hm_d["xlabel"],
                        yaxis_title=hm_d["ylabel"],
                        height=460,
                        margin=dict(l=70, r=20, t=55, b=50),
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    fig_hm_line = go.Figure(go.Scatter(
                        x=hm_d["x"], y=hm_d["y"],
                        mode="lines+markers",
                        line=dict(color=ACCENT, width=2),
                        marker=dict(size=7, color=ACCENT),
                        name="Sharpe",
                    ))
                    fig_hm_line.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.5)
                    apply_plotly_layout(
                        fig_hm_line,
                        title=f"Sharpe vs {hm_d['xlabel']} — {cur_strat} on {cur_asset}",
                        xaxis_title=hm_d["xlabel"],
                        yaxis_title="Sharpe",
                        height=300,
                        margin=dict(l=50, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_hm_line, use_container_width=True)

        # ── Strategy Comparison ────────────────────────────────────────────────
        st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)

        if result.trades:
            trades_export = pd.DataFrame(result.trades)
            equity_export = result.equity_curve.reset_index()
            equity_export.columns = ["date", "equity"]
            exp_col1, exp_col2, _ = st.columns([1, 1, 5])
            exp_col1.download_button(
                label="⬇  Trades CSV",
                data=trades_export.to_csv(index=False).encode(),
                file_name=f"backtest_{cur_asset}_{cur_strat.replace(' ','_').lower()}_{datetime.date.today()}.csv",
                mime="text/csv",
                key="dl_bt_trades",
            )
            exp_col2.download_button(
                label="⬇  Equity CSV",
                data=equity_export.to_csv(index=False).encode(),
                file_name=f"equity_{cur_asset}_{cur_strat.replace(' ','_').lower()}_{datetime.date.today()}.csv",
                mime="text/csv",
                key="dl_bt_equity",
            )

        cmp_btn = st.button("⚖  Compare All Strategies", key="cmp_btn")

        if cmp_btn:
            with st.spinner("Running all strategies for comparison…"):
                cmp_results = {}
                for strat_name, strat_fn, strat_params in [
                    ("MA Crossover", ma_crossover, {"fast": 10, "slow": 30}),
                    ("EMA Crossover", ema_crossover, {"fast": 9, "slow": 21}),
                    ("RSI Mean Reversion", rsi_mean_reversion, {"period": 14, "oversold": 30, "overbought": 70}),
                    ("Stochastic", stochastic_oscillator, {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80}),
                    ("Volume Breakout", volume_breakout, {"multiplier": 2.0, "lookback": 20}),
                    ("MACD Crossover", macd_crossover, {"fast": 12, "slow": 26, "signal": 9}),
                    ("Bollinger Bands", bollinger_bands, {"period": 20, "std_dev": 2.0}),
                    ("Z-Score MR", zscore_mean_reversion, {"window": 20, "threshold": 2.0}),
                    ("Keltner Channel", keltner_channel, {"ema_period": 20, "atr_mult": 1.5}),
                    ("VWAP Crossover", vwap_crossover, {"window": 20}),
                ]:
                    r = strat_fn(df, cur_asset, **strat_params)
                    cmp_results[strat_name] = (r, compute_metrics(r))
            st.session_state["cmp_results"] = cmp_results

        if "cmp_results" in st.session_state:
            cmp_results = st.session_state["cmp_results"]
            st.divider()
            st.markdown(
                f'<p style="color:{ACCENT};font-family:Space Mono,monospace;font-size:0.95rem;'
                f'font-weight:700;margin-bottom:8px;">Strategy Comparison (default params)</p>',
                unsafe_allow_html=True,
            )

            # Overlaid equity curves
            palette = [ACCENT, YELLOW, "#ff6b81", "#a29bfe", "#fd79a8", "#55efc4", "#fdcb6e", "#74b9ff"]
            fig_cmp = go.Figure()
            for (name, (res, _)), color in zip(cmp_results.items(), palette):
                fig_cmp.add_trace(go.Scatter(
                    x=res.equity_curve.index, y=res.equity_curve,
                    mode="lines", name=name,
                    line=dict(color=color, width=2),
                ))
            bh_base = df["Close"].loc[df.index[0]:]
            bh_cmp = bh_base / float(bh_base.iloc[0])
            fig_cmp.add_trace(go.Scatter(
                x=bh_cmp.index, y=bh_cmp,
                mode="lines", name="Buy & Hold",
                line=dict(color=MUTED, width=1.5, dash="dash"),
            ))
            fig_cmp.add_hline(y=1.0, line_dash="dot", line_color=BORDER, opacity=0.6)
            apply_plotly_layout(
                fig_cmp,
                title=f"{cur_asset} — All Strategies (normalized equity)",
                yaxis_title="Portfolio Value",
                height=360,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Metrics comparison table
            metric_keys = ["total_return", "win_rate", "sharpe", "sortino", "calmar", "max_drawdown", "num_trades", "avg_pnl_pct", "profit_factor"]
            metric_labels = ["Total Return %", "Win Rate %", "Sharpe", "Sortino", "Calmar", "Max Drawdown %", "# Trades", "Avg PnL %", "Profit Factor"]
            rows = {}
            for name, (_, m) in cmp_results.items():
                rows[name] = [m[k] for k in metric_keys]
            cmp_df = pd.DataFrame(rows, index=metric_labels)
            st.dataframe(cmp_df.style.format(lambda v: f"{v}" if isinstance(v, str) else f"{v:.2f}"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Portfolio Analyzer
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns([2, 1])
    with c1:
        pa_assets = st.multiselect(
            "Select Cryptos (2–8)", ASSETS,
            default=["BTC", "ETH", "SOL", "BNB"],
        )
    with c2:
        pa_dates = st.date_input(
            "Date Range",
            value=(D_START, D_END),
            min_value=datetime.date(2018, 1, 1),
            max_value=datetime.date.today(),
            key="pa_dates",
        )

    analyze_btn = st.button("Analyze Portfolio", key="pa_btn")

    if analyze_btn:
        if len(pa_assets) < 2:
            st.warning("Select at least 2 assets.")
        elif not isinstance(pa_dates, (list, tuple)) or len(pa_dates) != 2:
            st.warning("Select a valid date range.")
        else:
            s, e = str(pa_dates[0]), str(pa_dates[1])
            with st.spinner("Fetching price data for all assets…"):
                corr_df = correlation_matrix(pa_assets, s, e)
                stats_df = risk_return_stats(pa_assets, s, e)
                pa_prices = {a: fetch_prices(a, s, e)["Close"] for a in pa_assets}
            st.session_state.update(pa_corr=corr_df, pa_stats=stats_df, pa_prices=pa_prices)
            for _k in ("opt_weights", "opt_metrics"):
                st.session_state.pop(_k, None)

    corr_df: pd.DataFrame = st.session_state.get("pa_corr")
    stats_df: pd.DataFrame = st.session_state.get("pa_stats")

    if corr_df is not None and not corr_df.empty:
        left, right = st.columns(2)

        with left:
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns.tolist(),
                y=corr_df.index.tolist(),
                colorscale=[[0, "#1a2744"], [0.5, "#0a0e1a"], [1, ACCENT]],
                zmid=0,
                zmin=-1, zmax=1,
                text=np.round(corr_df.values, 2),
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(tickfont=dict(color=TEXT), outlinecolor=BORDER),
            ))
            apply_plotly_layout(fig_corr, title="Correlation Matrix", height=400)
            st.plotly_chart(fig_corr, use_container_width=True)

        with right:
            if stats_df is not None and not stats_df.empty:
                sr = stats_df.reset_index()
                abs_sharpe = sr["Sharpe Ratio"].abs()
                max_abs = abs_sharpe.max() or 1
                sizes = (abs_sharpe / max_abs * 28 + 12).tolist()

                fig_rr = go.Figure()
                for i, row in sr.iterrows():
                    c = ACCENT if row["Sharpe Ratio"] >= 0 else RED
                    fig_rr.add_trace(go.Scatter(
                        x=[row["Annual Volatility (%)"]],
                        y=[row["Annual Return (%)"]],
                        mode="markers+text",
                        name=row["Asset"],
                        text=[row["Asset"]],
                        textposition="top center",
                        textfont=dict(size=11, color=TEXT),
                        marker=dict(size=max(sizes[i], 10), color=c, opacity=0.85,
                                    line=dict(color="white", width=1)),
                        showlegend=False,
                    ))
                apply_plotly_layout(
                    fig_rr,
                    title="Risk / Return  (bubble size ∝ |Sharpe|)",
                    xaxis_title="Annual Volatility (%)",
                    yaxis_title="Annual Return (%)",
                    height=400,
                )
                st.plotly_chart(fig_rr, use_container_width=True)

        # Sharpe bar chart
        if stats_df is not None and not stats_df.empty:
            sr = stats_df.reset_index()
            colors = [ACCENT if s >= 0 else RED for s in sr["Sharpe Ratio"]]
            fig_sh = go.Figure(go.Bar(
                x=sr["Asset"],
                y=sr["Sharpe Ratio"],
                marker_color=colors,
                text=sr["Sharpe Ratio"].round(2),
                textposition="outside",
                textfont=dict(color=TEXT),
            ))
            fig_sh.add_hline(y=1.0, line_dash="dash", line_color=YELLOW,
                             annotation_text="Sharpe = 1", annotation_font_color=YELLOW)
            apply_plotly_layout(
                fig_sh,
                title="Sharpe Ratio by Asset (annualized)",
                yaxis_title="Sharpe Ratio",
                height=320,
            )
            st.plotly_chart(fig_sh, use_container_width=True)

            # Summary table
            st.markdown("**Risk / Return Summary**")
            st.dataframe(
                stats_df.style.format({
                    "Annual Return (%)": "{:.2f}%",
                    "Annual Volatility (%)": "{:.2f}%",
                    "Sharpe Ratio": "{:.2f}",
                }).background_gradient(
                    subset=["Sharpe Ratio"],
                    cmap="RdYlGn",
                    vmin=-1, vmax=2,
                ),
                use_container_width=True,
            )

        # ── Normalized price performance ───────────────────────────────────────
        pa_prices: dict = st.session_state.get("pa_prices", {})
        if pa_prices:
            price_palette = [ACCENT, YELLOW, "#ff6b81", "#a29bfe", "#fd79a8", "#55efc4", "#fdcb6e", "#e17055"]
            fig_norm = go.Figure()
            for (asset_name, series), color in zip(pa_prices.items(), price_palette):
                if series.empty:
                    continue
                norm = series / float(series.iloc[0])
                fig_norm.add_trace(go.Scatter(
                    x=norm.index, y=norm,
                    mode="lines", name=asset_name,
                    line=dict(color=color, width=1.8),
                ))
            fig_norm.add_hline(y=1.0, line_dash="dot", line_color=BORDER, opacity=0.5)
            apply_plotly_layout(
                fig_norm,
                title="Normalized Price Performance (base = 1)",
                yaxis_title="Return (normalized)",
                height=380,
            )
            st.plotly_chart(fig_norm, use_container_width=True)

            # ── Portfolio Optimizer ────────────────────────────────────────────
            st.divider()
            st.markdown(
                f'<p style="color:{ACCENT};font-family:Space Mono,monospace;font-size:0.95rem;'
                f'font-weight:700;margin-bottom:8px;">Portfolio Optimizer</p>',
                unsafe_allow_html=True,
            )
            opt_btn = st.button("⚡  Optimize Weights (Max Sharpe)", key="opt_btn")
            if opt_btn:
                price_df = pd.DataFrame({a: s for a, s in pa_prices.items()}).dropna()
                rets = price_df.pct_change().dropna()
                mean_rets = rets.mean() * 252
                cov_mat = rets.cov() * 252
                opt_assets = list(pa_prices.keys())
                n_opt = len(opt_assets)

                def _neg_sharpe(w):
                    pr = float(np.dot(w, mean_rets))
                    pv = float(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))))
                    return -pr / pv if pv > 0 else 0.0

                opt_result = scipy_minimize(
                    _neg_sharpe,
                    x0=np.ones(n_opt) / n_opt,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * n_opt,
                    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                )
                if opt_result.success:
                    opt_w = opt_result.x
                    st.session_state["opt_weights"] = dict(zip(opt_assets, opt_w.tolist()))
                    st.session_state["opt_metrics"] = {
                        "return": round(float(np.dot(opt_w, mean_rets)) * 100, 2),
                        "vol": round(float(np.sqrt(np.dot(opt_w.T, np.dot(cov_mat, opt_w)))) * 100, 2),
                        "sharpe": round(float(-opt_result.fun), 2),
                    }
                    st.session_state["opt_frontier_data"] = {
                        "mean_rets": mean_rets.tolist(),
                        "cov_mat": cov_mat.values.tolist(),
                        "assets": opt_assets,
                        "opt_w": opt_w.tolist(),
                    }
                else:
                    st.warning("Optimizer did not converge — try different assets or date range.")

            if "opt_weights" in st.session_state and "opt_metrics" in st.session_state:
                opt_w = st.session_state["opt_weights"]
                opt_m = st.session_state["opt_metrics"]
                sig_w = {a: w for a, w in opt_w.items() if w > 0.005}

                pie_palette = [ACCENT, YELLOW, "#ff6b81", "#a29bfe", "#fd79a8", "#55efc4", "#fdcb6e", "#e17055"]
                fig_pie = go.Figure(go.Pie(
                    labels=list(sig_w.keys()),
                    values=[round(w * 100, 2) for w in sig_w.values()],
                    marker=dict(colors=pie_palette[:len(sig_w)], line=dict(color=BG, width=2)),
                    textinfo="label+percent",
                    textfont=dict(color=TEXT, family="Space Mono, monospace", size=12),
                    hole=0.38,
                ))
                apply_plotly_layout(fig_pie, title="Optimal Portfolio Weights (Max Sharpe)", height=360)
                st.plotly_chart(fig_pie, use_container_width=True)

                oc1, oc2, oc3 = st.columns(3)
                metric_card(oc1, "Expected Annual Return", f"{opt_m['return']}%",
                            color=_color_for(opt_m["return"]))
                metric_card(oc2, "Annual Volatility", f"{opt_m['vol']}%", color="neutral")
                metric_card(oc3, "Portfolio Sharpe", opt_m["sharpe"],
                            color=_color_for(opt_m["sharpe"]))

                # ── Efficient Frontier ─────────────────────────────────────────
                if "opt_frontier_data" in st.session_state:
                    ef_btn = st.button("📈  Show Efficient Frontier (3000 portfolios)", key="ef_btn")
                    if ef_btn:
                        fd = st.session_state["opt_frontier_data"]
                        ef_mr = np.array(fd["mean_rets"])
                        ef_cov = np.array(fd["cov_mat"])
                        ef_assets = fd["assets"]
                        ef_n = len(ef_assets)
                        ef_rng = np.random.default_rng(7)
                        ef_vols, ef_rets, ef_sharpes = [], [], []
                        for _ in range(3000):
                            w = ef_rng.random(ef_n)
                            w /= w.sum()
                            r = float(np.dot(w, ef_mr))
                            v = float(np.sqrt(np.dot(w.T, np.dot(ef_cov, w))))
                            ef_vols.append(v * 100)
                            ef_rets.append(r * 100)
                            ef_sharpes.append(r / v if v > 0 else 0)
                        st.session_state["ef_plot"] = {
                            "vols": ef_vols, "rets": ef_rets, "sharpes": ef_sharpes,
                            "opt_vol": opt_m["vol"], "opt_ret": opt_m["return"],
                            "opt_sharpe": opt_m["sharpe"],
                        }

                    if "ef_plot" in st.session_state:
                        efp = st.session_state["ef_plot"]
                        sharpe_min = min(efp["sharpes"])
                        sharpe_max = max(efp["sharpes"])
                        sharpe_norm = [
                            (s - sharpe_min) / (sharpe_max - sharpe_min + 1e-9)
                            for s in efp["sharpes"]
                        ]
                        point_colors = [
                            f"rgba({int(255*(1-t))},{int(212*t)},{int(170*t)},0.65)"
                            for t in sharpe_norm
                        ]
                        fig_ef = go.Figure()
                        fig_ef.add_trace(go.Scatter(
                            x=efp["vols"], y=efp["rets"],
                            mode="markers",
                            marker=dict(size=4, color=point_colors, line=dict(width=0)),
                            name="Random portfolios",
                            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
                        ))
                        fig_ef.add_trace(go.Scatter(
                            x=[efp["opt_vol"]], y=[efp["opt_ret"]],
                            mode="markers",
                            marker=dict(size=18, symbol="star", color=YELLOW,
                                        line=dict(color="white", width=1.5)),
                            name=f"Max Sharpe ({efp['opt_sharpe']:.2f})",
                            hovertemplate=f"Max Sharpe Portfolio<br>Vol: {efp['opt_vol']:.1f}%<br>Return: {efp['opt_ret']:.1f}%<extra></extra>",
                        ))
                        apply_plotly_layout(
                            fig_ef,
                            title="Efficient Frontier — 3000 Random Portfolios (color = Sharpe)",
                            xaxis_title="Annual Volatility (%)",
                            yaxis_title="Annual Return (%)",
                            height=420,
                        )
                        st.plotly_chart(fig_ef, use_container_width=True)

    elif analyze_btn is False:
        st.info("Select assets and a date range, then click Analyze Portfolio.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trade Log
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        tl_asset = st.selectbox("Asset", ["All"] + ASSETS, key="tl_asset")
    with c2:
        tl_dates = st.date_input(
            "Date Range",
            value=(datetime.date(2018, 1, 1), datetime.date.today()),
            key="tl_dates",
        )
    with c3:
        tl_wl = st.selectbox("Result", ["All", "Win", "Loss"], key="tl_wl")

    asset_f = None if tl_asset == "All" else tl_asset
    wl_f = None if tl_wl == "All" else tl_wl
    s_f = str(tl_dates[0]) if isinstance(tl_dates, (list, tuple)) and len(tl_dates) == 2 else None
    e_f = str(tl_dates[1]) if isinstance(tl_dates, (list, tuple)) and len(tl_dates) == 2 else None

    trades_df = get_trades(asset=asset_f, start=s_f, end=e_f, win_loss=wl_f)

    if trades_df.empty:
        st.info("No trades found. Run a backtest to populate the trade log.")
    else:
        total = len(trades_df)
        wins = int((trades_df["pnl"] > 0).sum())
        total_pnl = round(float(trades_df["pnl_pct"].sum()), 2)

        sc1, sc2, sc3, sc4 = st.columns(4)
        metric_card(sc1, "Total Trades", total, color="neutral")
        metric_card(sc2, "Wins", wins, color="pos")
        metric_card(sc3, "Losses", total - wins, color="neg" if total - wins > 0 else "neutral")
        metric_card(sc4, "Cumulative PnL%", total_pnl, pct=True,
                    color=_color_for(total_pnl))

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        def _highlight_pnl(val):
            if isinstance(val, (int, float)):
                return f"color: {ACCENT}" if val > 0 else f"color: {RED}"
            return ""

        dl_col, _ = st.columns([1, 4])
        dl_col.download_button(
            label="⬇  Export CSV",
            data=trades_df.to_csv(index=False).encode(),
            file_name=f"trades_{tl_asset.lower()}_{datetime.date.today()}.csv",
            mime="text/csv",
            key="dl_trades",
        )

        styled = trades_df.style.applymap(_highlight_pnl, subset=["pnl", "pnl_pct"])
        st.dataframe(styled, use_container_width=True, height=480)

        # ── Monthly returns heatmap ────────────────────────────────────────────
        monthly = trades_df.copy()
        monthly["month"] = pd.to_datetime(monthly["entry_date"]).dt.to_period("M")
        monthly_pnl = monthly.groupby("month")["pnl_pct"].sum().reset_index()
        monthly_pnl["year"] = monthly_pnl["month"].dt.year
        monthly_pnl["mon"] = monthly_pnl["month"].dt.month

        if len(monthly_pnl) >= 2:
            pivot = monthly_pnl.pivot(index="year", columns="mon", values="pnl_pct").fillna(0)
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            col_labels = [month_names[m - 1] for m in pivot.columns]
            row_labels = [str(y) for y in pivot.index]
            z = pivot.values.tolist()
            text = [[f"{v:.1f}%" for v in row] for row in z]

            fig_heat = go.Figure(data=go.Heatmap(
                z=z,
                x=col_labels,
                y=row_labels,
                colorscale=[[0, RED], [0.5, BG], [1, ACCENT]],
                zmid=0,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=11, color=TEXT),
                showscale=True,
                colorbar=dict(tickfont=dict(color=TEXT), outlinecolor=BORDER, title="PnL %"),
            ))
            apply_plotly_layout(
                fig_heat,
                title="Monthly PnL % (by entry date)",
                height=max(180, 60 + len(row_labels) * 55),
                margin=dict(l=60, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SQL Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.82rem;margin-bottom:6px;">'
        "Run arbitrary SQL against the local SQLite database (read operations only recommended).</p>",
        unsafe_allow_html=True,
    )

    example_key = st.selectbox(
        "Example Queries",
        list(EXAMPLE_QUERIES.keys()),
        key="sql_example",
    )
    preset_sql = EXAMPLE_QUERIES[example_key]

    if "sql_text" not in st.session_state:
        st.session_state["sql_text"] = preset_sql

    # Sync selectbox choice into text area via session state
    if preset_sql and st.session_state.get("last_example") != example_key:
        st.session_state["sql_text"] = preset_sql
        st.session_state["last_example"] = example_key

    sql_query = st.text_area(
        "SQL Query",
        value=st.session_state["sql_text"],
        height=130,
        placeholder="SELECT * FROM trades LIMIT 10",
        key="sql_input",
    )

    run_sql_btn = st.button("▶  Run Query", key="run_sql")

    if run_sql_btn:
        q = sql_query.strip()
        if not q:
            st.warning("Enter a SQL query first.")
        else:
            try:
                res_df = run_raw_sql(q)
                rc = len(res_df)
                st.success(f"✓  {rc} row{'s' if rc != 1 else ''} returned")
                if not res_df.empty:
                    st.dataframe(res_df, use_container_width=True, height=420)
            except Exception as exc:
                st.error(f"SQL error: {exc}")

    # Schema reference
    with st.expander("Database Schema", expanded=False):
        st.markdown(
            f"""
```sql
-- prices
CREATE TABLE prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    UNIQUE(asset, date)
);

-- trades
CREATE TABLE trades (
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

-- portfolio
CREATE TABLE portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    asset TEXT NOT NULL,
    weight REAL,
    daily_return REAL,
    UNIQUE(date, asset)
);
```
""",
            unsafe_allow_html=False,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Batch Backtest
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.82rem;margin-bottom:10px;">'
        "Run one strategy across multiple assets and rank results by Sharpe ratio.</p>",
        unsafe_allow_html=True,
    )

    bb_c1, bb_c2, bb_c3 = st.columns([2, 1, 1])
    with bb_c1:
        batch_assets = st.multiselect("Assets", ASSETS, default=ASSETS, key="batch_assets")
    with bb_c2:
        batch_strategy = st.selectbox("Strategy", STRATEGIES, key="batch_strategy")
    with bb_c3:
        batch_dates = st.date_input(
            "Date Range",
            value=(D_START, D_END),
            min_value=datetime.date(2018, 1, 1),
            max_value=datetime.date.today(),
            key="batch_dates",
        )

    BATCH_DEFAULT_PARAMS = {
        "MA Crossover": {"fast": 10, "slow": 30},
        "EMA Crossover": {"fast": 9, "slow": 21},
        "RSI Mean Reversion": {"period": 14, "oversold": 30, "overbought": 70},
        "Stochastic Oscillator": {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80},
        "Volume Breakout": {"multiplier": 2.0, "lookback": 20},
        "MACD Crossover": {"fast": 12, "slow": 26, "signal": 9},
        "Bollinger Bands": {"period": 20, "std_dev": 2.0},
        "Z-Score MR": {"window": 20, "threshold": 2.0},
        "Keltner Channel": {"ema_period": 20, "atr_mult": 1.5},
        "VWAP Crossover": {"window": 20},
    }
    BATCH_FN_MAP = {
        "MA Crossover": ma_crossover,
        "EMA Crossover": ema_crossover,
        "RSI Mean Reversion": rsi_mean_reversion,
        "Stochastic Oscillator": stochastic_oscillator,
        "Volume Breakout": volume_breakout,
        "MACD Crossover": macd_crossover,
        "Bollinger Bands": bollinger_bands,
        "Z-Score MR": zscore_mean_reversion,
        "Keltner Channel": keltner_channel,
        "VWAP Crossover": vwap_crossover,
    }

    run_batch_btn = st.button("▶  Run Batch", key="run_batch")

    if run_batch_btn:
        if len(batch_assets) < 1:
            st.warning("Select at least one asset.")
        elif not isinstance(batch_dates, (list, tuple)) or len(batch_dates) != 2:
            st.warning("Select a valid date range.")
        else:
            bs, be = str(batch_dates[0]), str(batch_dates[1])
            fn = BATCH_FN_MAP[batch_strategy]
            bp = BATCH_DEFAULT_PARAMS[batch_strategy]
            batch_rows = []
            prog = st.progress(0, text="Running…")
            for idx, a in enumerate(batch_assets):
                prog.progress((idx + 1) / len(batch_assets), text=f"Backtesting {a}…")
                try:
                    adf = fetch_prices(a, bs, be)
                    if adf.empty:
                        continue
                    res = fn(adf, a, **bp)
                    m = compute_metrics(res)
                    batch_rows.append({
                        "Asset": a,
                        "Total Return %": m["total_return"],
                        "Sharpe": m["sharpe"],
                        "Sortino": m["sortino"],
                        "Win Rate %": m["win_rate"],
                        "Max DD %": m["max_drawdown"],
                        "# Trades": m["num_trades"],
                        "Avg PnL %": m["avg_pnl_pct"],
                        "Profit Factor": m["profit_factor"],
                        "_equity": res.equity_curve,
                    })
                except Exception:
                    pass
            prog.empty()
            st.session_state["batch_rows"] = batch_rows
            st.session_state["batch_strategy_label"] = batch_strategy

    if "batch_rows" in st.session_state and st.session_state["batch_rows"]:
        batch_rows = st.session_state["batch_rows"]
        batch_strat_label = st.session_state.get("batch_strategy_label", "")

        display_cols = ["Asset", "Total Return %", "Sharpe", "Sortino", "Win Rate %",
                        "Max DD %", "# Trades", "Avg PnL %", "Profit Factor"]
        batch_df = pd.DataFrame([{c: r[c] for c in display_cols} for r in batch_rows])
        batch_df = batch_df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
        batch_df.index = batch_df.index + 1

        st.divider()
        st.markdown(
            f'<p style="color:{ACCENT};font-family:Space Mono,monospace;font-size:0.95rem;'
            f'font-weight:700;margin-bottom:8px;">{batch_strat_label} — All Assets (default params)</p>',
            unsafe_allow_html=True,
        )

        def _style_batch(val):
            if not isinstance(val, (int, float)):
                return ""
            return f"color: {ACCENT}" if val > 0 else f"color: {RED}"

        styled_batch = batch_df.style.applymap(
            _style_batch,
            subset=["Total Return %", "Sharpe", "Sortino", "Avg PnL %"],
        ).format({
            "Total Return %": "{:.2f}%",
            "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}",
            "Win Rate %": "{:.1f}%",
            "Max DD %": "{:.2f}%",
            "Avg PnL %": "{:.2f}%",
        })
        st.dataframe(styled_batch, use_container_width=True)

        # Sharpe bar chart
        sorted_names = batch_df["Asset"].tolist()
        sorted_sharpe = batch_df["Sharpe"].tolist()
        sharpe_colors = [ACCENT if s >= 0 else RED for s in sorted_sharpe]

        fig_bs = go.Figure(go.Bar(
            x=sorted_names, y=sorted_sharpe,
            marker_color=sharpe_colors,
            marker_line=dict(color=BORDER, width=0.5),
            text=[f"{s:.2f}" for s in sorted_sharpe],
            textposition="outside",
            textfont=dict(color=TEXT),
            name="Sharpe",
        ))
        fig_bs.add_hline(y=1.0, line_dash="dash", line_color=YELLOW,
                         annotation_text="Sharpe = 1", annotation_font_color=YELLOW,
                         annotation_position="top right")
        fig_bs.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.5)
        apply_plotly_layout(
            fig_bs,
            title=f"Sharpe Ratio by Asset — {batch_strat_label}",
            yaxis_title="Sharpe Ratio",
            height=320,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_bs, use_container_width=True)

        # Total return bar chart
        ret_order = batch_df.sort_values("Total Return %", ascending=False)
        ret_colors = [ACCENT if v >= 0 else RED for v in ret_order["Total Return %"]]
        fig_br = go.Figure(go.Bar(
            x=ret_order["Asset"].tolist(),
            y=ret_order["Total Return %"].tolist(),
            marker_color=ret_colors,
            marker_line=dict(color=BORDER, width=0.5),
            text=[f"{v:.1f}%" for v in ret_order["Total Return %"]],
            textposition="outside",
            textfont=dict(color=TEXT),
            name="Total Return",
        ))
        fig_br.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.5)
        apply_plotly_layout(
            fig_br,
            title=f"Total Return % by Asset — {batch_strat_label}",
            yaxis_title="Total Return %",
            height=320,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_br, use_container_width=True)

        # Overlaid equity curves
        equity_rows = {r["Asset"]: r["_equity"] for r in batch_rows}
        if equity_rows:
            palette = [ACCENT, YELLOW, "#ff6b81", "#a29bfe", "#fd79a8", "#55efc4",
                       "#fdcb6e", "#e17055", "#74b9ff", "#00cec9", "#6c5ce7", "#fab1a0", "#dfe6e9"]
            fig_eq_b = go.Figure()
            for (aname, eq), color in zip(equity_rows.items(), palette):
                fig_eq_b.add_trace(go.Scatter(
                    x=eq.index, y=eq,
                    mode="lines", name=aname,
                    line=dict(color=color, width=1.6),
                ))
            fig_eq_b.add_hline(y=1.0, line_dash="dot", line_color=BORDER, opacity=0.5)
            apply_plotly_layout(
                fig_eq_b,
                title=f"Equity Curves — {batch_strat_label} (all assets)",
                yaxis_title="Portfolio Value (normalized)",
                height=380,
            )
            st.plotly_chart(fig_eq_b, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Signal Scanner
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.82rem;margin-bottom:10px;">'
        "Run all 10 strategies on selected assets and see the most recent buy/sell signal from each."
        " Green = last signal was BUY · Red = SELL · Grey = no signal generated.</p>",
        unsafe_allow_html=True,
    )

    sc_c1, sc_c2 = st.columns([2, 1])
    with sc_c1:
        sc_assets = st.multiselect("Assets to scan", ASSETS, default=ASSETS, key="sc_assets")
    with sc_c2:
        sc_dates = st.date_input(
            "Lookback period",
            value=(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31)),
            min_value=datetime.date(2018, 1, 1),
            max_value=datetime.date.today(),
            key="sc_dates",
        )

    _SC_FN_MAP = {
        "MA Cross": (ma_crossover, {"fast": 10, "slow": 30}),
        "EMA Cross": (ema_crossover, {"fast": 9, "slow": 21}),
        "RSI MR": (rsi_mean_reversion, {"period": 14, "oversold": 30, "overbought": 70}),
        "Stoch": (stochastic_oscillator, {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80}),
        "Vol Break": (volume_breakout, {"multiplier": 2.0, "lookback": 20}),
        "MACD": (macd_crossover, {"fast": 12, "slow": 26, "signal": 9}),
        "BB": (bollinger_bands, {"period": 20, "std_dev": 2.0}),
        "Z-Score": (zscore_mean_reversion, {"window": 20, "threshold": 2.0}),
        "Keltner": (keltner_channel, {"ema_period": 20, "atr_mult": 1.5}),
        "VWAP": (vwap_crossover, {"window": 20}),
    }

    scan_btn = st.button("▶  Run Signal Scan", key="scan_btn")

    if scan_btn:
        if not sc_assets:
            st.warning("Select at least one asset.")
        elif not isinstance(sc_dates, (list, tuple)) or len(sc_dates) != 2:
            st.warning("Select a valid date range.")
        else:
            sc_s, sc_e = str(sc_dates[0]), str(sc_dates[1])
            strat_names = list(_SC_FN_MAP.keys())
            signal_matrix = {}
            prog_sc = st.progress(0, text="Scanning…")
            for ai, asset_sc in enumerate(sc_assets):
                prog_sc.progress((ai + 1) / len(sc_assets), text=f"Scanning {asset_sc}…")
                try:
                    adf_sc = fetch_prices(asset_sc, sc_s, sc_e)
                    if adf_sc.empty:
                        signal_matrix[asset_sc] = {s: "—" for s in strat_names}
                        continue
                    row = {}
                    for sname, (sfn, sparams) in _SC_FN_MAP.items():
                        try:
                            sr = sfn(adf_sc, asset_sc, **sparams)
                            sig_col = sr.signals["signal"].dropna()
                            sig_col = sig_col[sig_col != ""]
                            row[sname] = sig_col.iloc[-1] if not sig_col.empty else "—"
                        except Exception:
                            row[sname] = "—"
                    signal_matrix[asset_sc] = row
                except Exception:
                    signal_matrix[asset_sc] = {s: "—" for s in strat_names}
            prog_sc.empty()
            st.session_state["sc_matrix"] = signal_matrix
            st.session_state["sc_strat_names"] = strat_names

    if "sc_matrix" in st.session_state:
        sc_mat = st.session_state["sc_matrix"]
        sc_strats = st.session_state["sc_strat_names"]
        sc_asset_list = list(sc_mat.keys())

        # Build numeric matrix for heatmap: buy=1, sell=-1, none=0
        z_num = []
        z_text = []
        for asset_sc in sc_asset_list:
            row_num, row_txt = [], []
            for sn in sc_strats:
                sig = sc_mat[asset_sc].get(sn, "—")
                row_num.append(1 if sig == "buy" else (-1 if sig == "sell" else 0))
                row_txt.append("BUY" if sig == "buy" else ("SELL" if sig == "sell" else "—"))
            z_num.append(row_num)
            z_text.append(row_txt)

        fig_sc = go.Figure(go.Heatmap(
            z=z_num,
            x=sc_strats,
            y=sc_asset_list,
            colorscale=[[0, RED], [0.5, "#1a2744"], [1, ACCENT]],
            zmin=-1, zmax=1,
            zmid=0,
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=11, color=TEXT, family="Space Mono, monospace"),
            showscale=False,
        ))
        apply_plotly_layout(
            fig_sc,
            title="Signal Matrix — last signal from each strategy (default params)",
            height=max(300, 60 + len(sc_asset_list) * 42),
            margin=dict(l=70, r=20, t=55, b=80),
            xaxis=dict(gridcolor=BORDER, showgrid=False, zeroline=False, tickangle=-30),
            yaxis=dict(gridcolor=BORDER, showgrid=False, zeroline=False),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Bullish / Bearish score per asset
        sc_summary = []
        for asset_sc in sc_asset_list:
            row = sc_mat[asset_sc]
            buys = sum(1 for v in row.values() if v == "buy")
            sells = sum(1 for v in row.values() if v == "sell")
            sc_summary.append({
                "Asset": asset_sc,
                "Buy signals": buys,
                "Sell signals": sells,
                "Neutral": len(sc_strats) - buys - sells,
                "Bias": "Bullish" if buys > sells else ("Bearish" if sells > buys else "Neutral"),
            })
        sc_sum_df = pd.DataFrame(sc_summary).sort_values("Buy signals", ascending=False).reset_index(drop=True)

        def _sc_bias_color(val):
            if val == "Bullish":
                return f"color: {ACCENT}"
            if val == "Bearish":
                return f"color: {RED}"
            return f"color: {MUTED}"

        st.markdown(
            f'<p style="color:{MUTED};font-size:0.82rem;margin-top:8px;">Signal tally per asset:</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            sc_sum_df.style.applymap(_sc_bias_color, subset=["Bias"]),
            use_container_width=True,
        )
