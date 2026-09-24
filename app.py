"""
Bullish Pattern Scanner
Scan stock price histories for bullish chart patterns, then view the chart
and key stats for every match.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Bullish Pattern Scanner", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------- #
# Stock universes
# --------------------------------------------------------------------------- #
FALLBACK_TICKERS = (
    "AAPL MSFT NVDA AMZN GOOGL META TSLA BRK-B JPM V UNH XOM JNJ WMT MA PG HD CVX "
    "MRK ABBV LLY AVGO KO PEP COST ADBE CRM NFLX AMD INTC CSCO ORCL QCOM TXN IBM "
    "BAC WFC GS MS C DIS NKE MCD SBUX T VZ CMCSA PFE TMO ABT DHR BMY AMGN GILD CAT "
    "DE BA GE HON UPS LMT RTX NEE DUK SO LIN SPGI BLK AXP PYPL INTU NOW AMAT MU"
).split()

HEADERS = {"User-Agent": "Mozilla/5.0 (pattern-scanner)"}


def _wiki_tickers(url: str, min_rows: int) -> list[str]:
    html = requests.get(url, headers=HEADERS, timeout=20).text
    for table in pd.read_html(io.StringIO(html)):
        for col in ("Symbol", "Ticker"):
            if col in table.columns and len(table) >= min_rows:
                syms = table[col].astype(str).str.strip().str.replace(".", "-", regex=False)
                return sorted(set(syms))
    raise ValueError("ticker table not found")


@st.cache_data(ttl=86400, show_spinner=False)
def get_universe(name: str) -> list[str]:
    try:
        if name == "S&P 500":
            return _wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 400)
        if name == "Nasdaq-100":
            return _wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100", 90)
    except Exception:
        pass
    return FALLBACK_TICKERS


# --------------------------------------------------------------------------- #
# Data download (cached, chunked)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def _download_chunk(tickers: tuple[str, ...], start: str) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        list(tickers), start=start, auto_adjust=True, group_by="ticker",
        threads=True, progress=False,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(0):
                    continue
                df = raw[t]
            else:
                df = raw
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(
                subset=["Open", "High", "Low", "Close"]
            )
            if len(df) < 60:
                continue
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            out[t] = df.astype(float)
        except Exception:
            continue
    return out


def load_prices(tickers: list[str], start: str, chunk: int = 50) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    bar = st.progress(0.0, text="Downloading price history…")
    chunks = [tickers[i : i + chunk] for i in range(0, len(tickers), chunk)]
    for i, ch in enumerate(chunks, 1):
        data.update(_download_chunk(tuple(ch), start))
        done = min(i * chunk, len(tickers))
        bar.progress(i / len(chunks), text=f"Downloading price history… {done}/{len(tickers)}")
    bar.empty()
    return data


@st.cache_data(ttl=86400, show_spinner=False)
def get_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    for n in (20, 50, 200):
        df[f"SMA{n}"] = c.rolling(n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["VolAvg50"] = df["Volume"].rolling(50).mean()
    return df


# --------------------------------------------------------------------------- #
# Pattern detectors
# Each takes (df, window_start, tolerance, pivot_order) and returns [Signal].
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    date: pd.Timestamp
    detail: str
    points: list = field(default_factory=list)  # (date, price, label)
    lines: list = field(default_factory=list)   # (date0, price0, date1, price1, label)
    start: pd.Timestamp | None = None


def _prior_down(df: pd.DataFrame, lag: int = 1, span: int = 5) -> pd.Series:
    c = df["Close"]
    return c.shift(lag) < c.shift(lag + span)


def _hits(df, mask, ws):
    return df.index[mask.fillna(False) & (df.index >= ws)]


def detect_golden_cross(df, ws, tol, k):
    a, b = df["SMA50"], df["SMA200"]
    m = (a > b) & (a.shift(1) <= b.shift(1))
    return [
        Signal(d, f"50-day SMA ({a[d]:.2f}) crossed above 200-day SMA ({b[d]:.2f})",
               points=[(d, a[d], "Golden Cross")])
        for d in _hits(df, m, ws)
    ]


def detect_macd(df, ws, tol, k):
    m, s = df["MACD"], df["MACDSignal"]
    mask = (m > s) & (m.shift(1) <= s.shift(1)) & (m < 0)
    return [
        Signal(d, f"MACD ({m[d]:.2f}) crossed above signal line ({s[d]:.2f}) below zero",
               points=[(d, df.at[d, "Low"], "MACD cross")])
        for d in _hits(df, mask, ws)
    ]


def detect_rsi(df, ws, tol, k):
    r = df["RSI"]
    mask = (r > 30) & (r.shift(1) <= 30)
    return [
        Signal(d, f"RSI climbed back above 30 (now {r[d]:.1f}) after being oversold",
               points=[(d, df.at[d, "Low"], "RSI rebound")])
        for d in _hits(df, mask, ws)
    ]


def detect_engulfing(df, ws, tol, k):
    o, c = df["Open"], df["Close"]
    o1, c1 = o.shift(1), c.shift(1)
    mask = (c1 < o1) & (c > o) & (o <= c1) & (c >= o1) & ((c - o) > (o1 - c1)) & _prior_down(df)
    return [
        Signal(d, f"Green candle ({o[d]:.2f}→{c[d]:.2f}) engulfed prior red candle after a decline",
               points=[(d, df.at[d, "Low"], "Engulfing")])
        for d in _hits(df, mask, ws)
    ]


def detect_hammer(df, ws, tol, k):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body, rng = (c - o).abs(), h - l
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    mask = (rng > 0) & (body > 0) & (lower >= 2 * body) & (upper <= body) & (body <= 0.35 * rng) & _prior_down(df)
    return [
        Signal(d, f"Hammer: long lower wick to {l[d]:.2f}, closed at {c[d]:.2f} after a decline",
               points=[(d, l[d], "Hammer")])
        for d in _hits(df, mask, ws)
    ]


def detect_morning_star(df, ws, tol, k):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    o1, c1 = o.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)
    body2 = o2 - c2
    rng2 = h.shift(2) - l.shift(2)
    mask = (
        (c2 < o2) & (body2 >= 0.5 * rng2)
        & ((c1 - o1).abs() <= 0.3 * body2)
        & (c > o) & (c >= (o2 + c2) / 2)
        & _prior_down(df, 2, 5)
    )
    return [
        Signal(d, f"Three-candle reversal completed with close at {c[d]:.2f}",
               points=[(d, l[d], "Morning Star")])
        for d in _hits(df, mask, ws)
    ]


def detect_52w_breakout(df, ws, tol, k):
    prior_high = df["High"].shift(1).rolling(252, min_periods=200).max()
    raw = (df["Close"] > prior_high) & (df["Volume"] > 1.5 * df["VolAvg50"])
    raw = raw.fillna(False)
    # keep only the first breakout in any 20-day stretch
    mask = raw & ~raw.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
    return [
        Signal(d, f"Closed at {df.at[d, 'Close']:.2f}, above prior 52-week high {prior_high[d]:.2f}, "
                  f"on {df.at[d, 'Volume'] / df.at[d, 'VolAvg50']:.1f}× average volume",
               points=[(d, df.at[d, "Close"], "Breakout")],
               lines=[(d - pd.Timedelta(days=90), prior_high[d], d, prior_high[d], "Prior 52w high")])
        for d in _hits(df, mask, ws)
    ]


def _pivots(arr: np.ndarray, k: int, kind: str) -> list[int]:
    s = pd.Series(arr)
    roll = s.rolling(2 * k + 1, center=True, min_periods=2 * k + 1)
    ext = roll.min() if kind == "low" else roll.max()
    out: list[int] = []
    for i in np.where(s.values == ext.values)[0]:
        if not out or i - out[-1] > k:
            out.append(int(i))
    return out


def detect_double_bottom(df, ws, tol, k):
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    idx, n = df.index, len(df)
    lows = _pivots(L, k, "low")
    sigs, last_t = [], -1
    for i, j in zip(lows, lows[1:]):
        if not 10 <= j - i <= 150:
            continue
        l1, l2 = L[i], L[j]
        floor = min(l1, l2)
        if abs(l1 - l2) / floor > tol:
            continue
        mid = i + int(np.argmax(H[i : j + 1]))
        peak = H[mid]
        if peak < max(l1, l2) * 1.05:
            continue
        if H[max(0, i - 40) : i + 1].max() < l1 * 1.08:  # needs a decline into bottom 1
            continue
        for t in range(j + 1, min(j + 60, n)):
            if L[t] < floor * (1 - tol):
                break
            if C[t] > peak:
                if t > last_t and idx[t] >= ws:
                    sigs.append(Signal(
                        idx[t],
                        f"Bottoms at {l1:.2f} and {l2:.2f}, neckline {peak:.2f}, breakout close {C[t]:.2f}",
                        points=[(idx[i], l1, "Bottom 1"), (idx[j], l2, "Bottom 2"), (idx[t], C[t], "Breakout")],
                        lines=[(idx[mid], peak, idx[t], peak, "Neckline")],
                        start=idx[i],
                    ))
                    last_t = t
                break
    return sigs


def detect_inverse_hs(df, ws, tol, k):
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    idx, n = df.index, len(df)
    lows = _pivots(L, k, "low")
    sigs, last_t = [], -1
    for a, b, c in zip(lows, lows[1:], lows[2:]):
        if c - a > 250 or b - a < 5 or c - b < 5:
            continue
        la, lb, lc = L[a], L[b], L[c]
        if not (lb < la * 0.97 and lb < lc * 0.97):
            continue
        if abs(la - lc) / min(la, lc) > tol * 2:
            continue
        p1 = a + int(np.argmax(H[a : b + 1]))
        p2 = b + int(np.argmax(H[b : c + 1]))
        slope = (H[p2] - H[p1]) / max(p2 - p1, 1)
        for t in range(c + 1, min(c + 60, n)):
            if L[t] < lb:
                break
            neck = H[p1] + slope * (t - p1)
            if C[t] > neck:
                if t > last_t and idx[t] >= ws:
                    sigs.append(Signal(
                        idx[t],
                        f"Shoulders {la:.2f} / {lc:.2f}, head {lb:.2f}, broke neckline at {C[t]:.2f}",
                        points=[(idx[a], la, "L. shoulder"), (idx[b], lb, "Head"),
                                (idx[c], lc, "R. shoulder"), (idx[t], C[t], "Breakout")],
                        lines=[(idx[p1], H[p1], idx[t], neck, "Neckline")],
                        start=idx[a],
                    ))
                    last_t = t
                break
    return sigs


def detect_cup_handle(df, ws, tol, k):
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    idx, n = df.index, len(df)
    highs = _pivots(H, k, "high")
    sigs, used_until = [], -1
    for ai, a in enumerate(highs):
        if a <= used_until:
            continue
        for c in highs[ai + 1 :]:
            span = c - a
            if span < 30:
                continue
            if span > 260:
                break
            A, Cr = H[a], H[c]
            if abs(A - Cr) / A > max(tol * 1.5, 0.05):
                continue
            rim = max(A, Cr)
            if H[a + 1 : c].max() > rim * 1.02:
                break
            b = a + int(np.argmin(L[a : c + 1]))
            B = L[b]
            if not 0.12 <= (rim - B) / rim <= 0.45:
                continue
            if not a + 0.25 * span <= b <= a + 0.75 * span:
                continue
            found = False
            for t in range(c + 1, min(c + 45, n)):
                if C[t] > rim:
                    if t - c >= 5:
                        handle_low = L[c:t].min()
                        if (rim - handle_low) / rim <= 0.15 and handle_low >= B + 0.5 * (rim - B) and idx[t] >= ws:
                            sigs.append(Signal(
                                idx[t],
                                f"Cup depth {(rim - B) / rim:.0%} over {span} sessions, handle low {handle_low:.2f}, "
                                f"breakout close {C[t]:.2f}",
                                points=[(idx[a], A, "Left rim"), (idx[b], B, "Cup bottom"),
                                        (idx[c], Cr, "Right rim"), (idx[t], C[t], "Breakout")],
                                lines=[(idx[a], rim, idx[t], rim, "Rim")],
                                start=idx[a],
                            ))
                            found, used_until = True, t
                    break
            if found:
                break
    return sigs


PATTERNS = {
    "Golden Cross": (detect_golden_cross,
        "The 50-day moving average crosses above the 200-day — a classic long-term trend shift."),
    "Double Bottom": (detect_double_bottom,
        "Price tests a low twice, rallies between, then closes above the middle peak (the neckline)."),
    "Inverse Head & Shoulders": (detect_inverse_hs,
        "Three lows with the middle one deepest, confirmed by a close above the neckline."),
    "Cup and Handle": (detect_cup_handle,
        "A rounded 12–45% dip that recovers to the prior high, a shallow pullback, then a breakout."),
    "52-Week High Breakout": (detect_52w_breakout,
        "A close above the prior 52-week high on at least 1.5× average volume."),
    "Bullish Engulfing": (detect_engulfing,
        "After a decline, a green candle's body fully covers the previous red candle's body."),
    "Hammer": (detect_hammer,
        "After a decline, a small body near the top of the day's range with a long lower wick."),
    "Morning Star": (detect_morning_star,
        "Big red candle, small indecisive candle, then a green candle closing past the red one's midpoint."),
    "MACD Bullish Crossover": (detect_macd,
        "MACD crosses above its signal line while still below zero (momentum turning up from weakness)."),
    "RSI Oversold Rebound": (detect_rsi,
        "14-day RSI climbs back above 30 after being oversold."),
}


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
def run_scan(data, pattern, ws, tol, recent_days, min_price):
    fn = PATTERNS[pattern][0]
    rows, sigmap, frames = [], {}, {}
    for t, raw in data.items():
        df = add_indicators(raw)
        win = df[df.index >= ws]
        if len(win) < 20 or win["Close"].iloc[-1] < min_price:
            continue
        k = int(np.clip(len(win) // 50, 3, 10))
        try:
            sigs = fn(df, ws, tol, k)
        except Exception:
            continue
        if not sigs:
            continue
        last = sigs[-1]
        if recent_days and (df.index[-1] - last.date).days > recent_days:
            continue
        close = win["Close"].iloc[-1]
        sig_close = df.at[last.date, "Close"]
        rows.append({
            "Ticker": t,
            "Last signal": last.date.date(),
            "Signals in window": len(sigs),
            "Last close": close,
            "Since signal %": (close / sig_close - 1) * 100,
            "Window return %": (close / win["Close"].iloc[0] - 1) * 100,
            "RSI": df["RSI"].iloc[-1],
        })
        sigmap[t], frames[t] = sigs, df
    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["Last signal", "Since signal %"], ascending=[False, False]).reset_index(drop=True)
    return res, sigmap, frames


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
def make_chart(ticker, df, ws, sigs, focus, pattern):
    view = df[df.index >= ws - pd.Timedelta(days=7)]
    extra = pattern in ("MACD Bullish Crossover", "RSI Oversold Rebound")
    rows = 3 if extra else 2
    heights = [0.62, 0.18, 0.2] if extra else [0.78, 0.22]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
        name="Price", increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
    ), 1, 1)
    for col, color in (("SMA50", "#f59e0b"), ("SMA200", "#7c3aed")):
        fig.add_trace(go.Scatter(x=view.index, y=view[col], name=col.replace("SMA", "") + "-day SMA",
                                 line=dict(color=color, width=1.4)), 1, 1)

    in_view = [s for s in sigs if s.date in view.index]
    fig.add_trace(go.Scatter(
        x=[s.date for s in in_view], y=[view.at[s.date, "Low"] * 0.97 for s in in_view],
        mode="markers", name="Signal", marker=dict(symbol="triangle-up", size=12, color="#0ea5e9"),
        hovertext=[s.detail for s in in_view], hoverinfo="text+x",
    ), 1, 1)

    if focus is not None:
        start = focus.start or focus.date - pd.Timedelta(days=5)
        fig.add_vrect(x0=max(start, view.index[0]), x1=focus.date, fillcolor="#0ea5e9",
                      opacity=0.08, line_width=0, row=1, col=1)
        for d0, p0, d1, p1, label in focus.lines:
            fig.add_trace(go.Scatter(x=[max(d0, view.index[0]), d1], y=[p0, p1], mode="lines",
                                     name=label, line=dict(color="#0f172a", dash="dash", width=1.5)), 1, 1)
        if focus.points:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in focus.points], y=[p[1] for p in focus.points],
                mode="markers+text", text=[p[2] for p in focus.points], textposition="bottom center",
                marker=dict(size=9, color="#0f172a", symbol="circle-open", line=dict(width=2)),
                name="Pattern points", showlegend=False,
            ), 1, 1)

    vol_colors = np.where(view["Close"] >= view["Open"], "#86efac", "#fca5a5")
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], marker_color=vol_colors, name="Volume",
                         showlegend=False), 2, 1)

    if pattern == "MACD Bullish Crossover":
        fig.add_trace(go.Scatter(x=view.index, y=view["MACD"], name="MACD", line=dict(color="#2563eb")), 3, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view["MACDSignal"], name="Signal line",
                                 line=dict(color="#f97316")), 3, 1)
    elif pattern == "RSI Oversold Rebound":
        fig.add_trace(go.Scatter(x=view.index, y=view["RSI"], name="RSI", line=dict(color="#2563eb")), 3, 1)
        for lvl in (30, 70):
            fig.add_hline(y=lvl, line_dash="dot", line_color="#94a3b8", row=3, col=1)

    fig.update_layout(
        title=f"{ticker} — {pattern}", height=700 if extra else 620, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_big(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.2f}{suf}"
    return f"{x:,.0f}"


def fmt_num(x, fmt="{:.2f}"):
    try:
        return fmt.format(float(x)) if x is not None and not np.isnan(float(x)) else "—"
    except (TypeError, ValueError):
        return "—"


def render_info(ticker, df, ws):
    info = get_info(ticker)
    win = df[df.index >= ws]
    last_year = df[df.index >= df.index[-1] - pd.Timedelta(days=365)]
    close = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]

    name = info.get("longName") or info.get("shortName") or ticker
    st.subheader(f"{name} ({ticker})")
    meta = " · ".join(x for x in (info.get("sector"), info.get("industry"), info.get("exchange")) if x)
    if meta:
        st.caption(meta)

    div_rate, price = info.get("dividendRate"), info.get("currentPrice") or close
    div_yield = (div_rate / price * 100) if div_rate and price else None

    r1 = st.columns(4)
    r1[0].metric("Last close", f"${close:,.2f}", f"{(close / prev - 1) * 100:+.2f}%")
    r1[1].metric("Market cap", fmt_big(info.get("marketCap")))
    r1[2].metric("P/E (trailing)", fmt_num(info.get("trailingPE")))
    r1[3].metric("P/E (forward)", fmt_num(info.get("forwardPE")))
    r2 = st.columns(4)
    r2[0].metric("52-week range", f"${last_year['Low'].min():,.2f} – ${last_year['High'].max():,.2f}")
    r2[1].metric("Window return", f"{(close / win['Close'].iloc[0] - 1) * 100:+.1f}%")
    r2[2].metric("Dividend yield", f"{div_yield:.2f}%" if div_yield else "—")
    r2[3].metric("Beta", fmt_num(info.get("beta")))
    r3 = st.columns(4)
    r3[0].metric("Avg volume (50d)", fmt_big(df["VolAvg50"].iloc[-1]))
    r3[1].metric("RSI (14)", fmt_num(df["RSI"].iloc[-1], "{:.1f}"))
    r3[2].metric("vs 50-day SMA", f"{(close / df['SMA50'].iloc[-1] - 1) * 100:+.1f}%"
                 if not np.isnan(df["SMA50"].iloc[-1]) else "—")
    r3[3].metric("vs 200-day SMA", f"{(close / df['SMA200'].iloc[-1] - 1) * 100:+.1f}%"
                 if not np.isnan(df["SMA200"].iloc[-1]) else "—")

    summary = info.get("longBusinessSummary")
    if summary:
        with st.expander("Company overview"):
            st.write(summary)
            if info.get("website"):
                st.write(info["website"])
    if not info:
        st.caption("Company fundamentals unavailable right now (Yahoo may be rate-limiting); price stats shown.")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("📈 Bullish Pattern Scanner")

with st.sidebar:
    st.header("Scan settings")
    pattern = st.selectbox("Bullish pattern", list(PATTERNS))
    st.caption(PATTERNS[pattern][1])

    c1, c2 = st.columns([1, 1])
    length = c1.number_input("History length", min_value=1, max_value=240, value=12, step=1)
    unit = c2.selectbox("Unit", ["Months", "Years"])

    universe = st.radio("Stocks to scan", ["S&P 500", "Nasdaq-100", "Custom list"])
    custom = ""
    if universe == "Custom list":
        custom = st.text_area("Tickers (comma or space separated)", "AAPL, MSFT, NVDA, AMZN, GOOGL")

    with st.expander("Advanced"):
        tol = st.slider("Pattern tolerance", 1, 8, 3, help="How closely matching lows/rims must line up (%).") / 100
        recent_days = st.number_input("Only signals from the last N days (0 = anywhere in history window)",
                                      min_value=0, max_value=3650, value=0)
        min_price = st.number_input("Minimum share price ($)", min_value=0.0, value=5.0, step=1.0)

    run = st.button("Scan", type="primary", use_container_width=True)

if run:
    years = length if unit == "Years" else 0
    months = length if unit == "Months" else 0
    if years > 20:
        st.warning("Capping history at 20 years.")
        years = 20
    today = pd.Timestamp.today().normalize()
    ws = today - pd.DateOffset(years=years, months=months)
    dl_start = (ws - pd.Timedelta(days=420)).strftime("%Y-%m-%d")  # warm-up for 200-day SMA & 52w high

    if universe == "Custom list":
        tickers = sorted({t.strip().upper() for t in custom.replace(",", " ").split() if t.strip()})
    else:
        with st.spinner(f"Loading {universe} tickers…"):
            tickers = get_universe(universe)
            if tickers == FALLBACK_TICKERS:
                st.info(f"Couldn't fetch the {universe} list, so scanning a built-in list of large caps instead.")

    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()

    data = load_prices(tickers, dl_start)
    if not data:
        st.error("No price data came back. Yahoo Finance may be rate-limiting — wait a minute and try again.")
        st.stop()

    with st.spinner("Looking for patterns…"):
        res, sigmap, frames = run_scan(data, pattern, ws, tol, recent_days, min_price)

    st.session_state.update(res=res, sigmap=sigmap, frames=frames, ws=ws, pattern=pattern,
                            scanned=len(data), label=f"{length} {unit.lower()}")

if "res" not in st.session_state:
    st.info("Pick a pattern and a history length in the sidebar, then hit **Scan**.")
    st.stop()

res, sigmap, frames = st.session_state.res, st.session_state.sigmap, st.session_state.frames
ws, pat = st.session_state.ws, st.session_state.pattern

st.markdown(f"**{len(res)}** of {st.session_state.scanned} stocks showed a **{pat}** "
            f"in the last {st.session_state.label} (since {ws.date()}).")
if res.empty:
    st.warning("No matches. Try a longer history, a wider universe, or a looser tolerance.")
    st.stop()

st.dataframe(
    res, use_container_width=True, hide_index=True, height=min(38 * (len(res) + 1), 400),
    column_config={
        "Last close": st.column_config.NumberColumn(format="$%.2f"),
        "Since signal %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Window return %": st.column_config.NumberColumn(format="%+.1f%%"),
        "RSI": st.column_config.NumberColumn(format="%.1f"),
    },
)

st.divider()
left, right = st.columns([1, 1])
ticker = left.selectbox("Show stock", res["Ticker"].tolist(),
                        format_func=lambda t: f"{t}  ·  last signal {res.set_index('Ticker').at[t, 'Last signal']}")
sigs = sigmap[ticker]
choice = right.selectbox("Highlight occurrence", list(range(len(sigs)))[::-1],
                         format_func=lambda i: f"{sigs[i].date.date()}")
focus = sigs[choice]

st.plotly_chart(make_chart(ticker, frames[ticker], ws, sigs, focus, pat), use_container_width=True)
st.caption(f"**{focus.date.date()}:** {focus.detail}")

render_info(ticker, frames[ticker], ws)

st.caption("Pattern detection uses rule-based heuristics on daily prices from Yahoo Finance. "
           "It's a screening tool, not investment advice.")
