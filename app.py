import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import get_trades, init_db, run_raw_sql, save_trades
from data_fetcher import fetch_prices
from portfolio import correlation_matrix, risk_return_stats
from strategies import (
    BacktestResult,
    bollinger_bands,
    compute_metrics,
    ema_crossover,
    ma_crossover,
    macd_crossover,
    rsi_mean_reversion,
    stochastic_oscillator,
    volume_breakout,
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
STRATEGIES = ["MA Crossover", "EMA Crossover", "RSI Mean Reversion", "Stochastic Oscillator", "Volume Breakout", "MACD Crossover", "Bollinger Bands"]
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


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈  Strategy Backtester", "🗂  Portfolio Analyzer", "📋  Trade Log", "🔍  SQL Explorer"]
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
    else:  # Bollinger Bands
        p1, p2, p3 = st.columns([1, 1, 2])
        bb_period = p1.slider("Period", 5, 50, 20)
        bb_std = p2.slider("Std Dev", 1.0, 4.0, 2.0, 0.1)
        params = {"period": bb_period, "std_dev": bb_std}

    st.markdown(
        f'<p style="color:{MUTED};font-size:0.8rem;margin:6px 0 2px;">Risk Controls</p>',
        unsafe_allow_html=True,
    )
    rc1, rc2, rc3 = st.columns([1, 1, 2])
    stop_loss_pct = rc1.slider("Stop Loss %", 0.0, 30.0, 0.0, 0.5,
                                help="0 = disabled. Exit trade if loss exceeds this %.")
    take_profit_pct = rc2.slider("Take Profit %", 0.0, 100.0, 0.0, 1.0,
                                  help="0 = disabled. Exit trade if gain exceeds this %.")

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
            sl_tp = {"stop_loss": stop_loss_pct, "take_profit": take_profit_pct}
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
            else:
                result = bollinger_bands(df, asset, **params, **sl_tp)

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

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines", name="Price",
            line=dict(color=ACCENT, width=1.5),
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

        if not buys.empty:
            fig_p.add_trace(go.Scatter(
                x=buys.index, y=buys["Close"] * 0.985,
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=11, color=ACCENT,
                            line=dict(color="white", width=1)),
            ))
        if not sells.empty:
            fig_p.add_trace(go.Scatter(
                x=sells.index, y=sells["Close"] * 1.015,
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=11, color=RED,
                            line=dict(color="white", width=1)),
            ))

        apply_plotly_layout(
            fig_p,
            title=f"{cur_asset} — {cur_strat}",
            yaxis_title="Price (USD)",
            height=420,
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

        # ── Trade PnL distribution ─────────────────────────────────────────────
        if result.trades:
            pnl_vals = [t["pnl_pct"] for t in result.trades]
            bar_colors = [ACCENT if v >= 0 else RED for v in pnl_vals]
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
            palette = [ACCENT, YELLOW, "#ff6b81", "#a29bfe", "#fd79a8", "#55efc4", "#fdcb6e"]
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
            st.session_state.update(pa_corr=corr_df, pa_stats=stats_df)

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
