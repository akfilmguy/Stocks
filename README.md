# Bullish Pattern Scanner

## Run it on your computer
1. Install Python 3.10 or newer (python.org).
2. Open a terminal in this folder and run:
   pip install -r requirements.txt
   streamlit run app.py
3. Your browser opens to the app (http://localhost:8501).

## Put it online (free)
Push these three files to a GitHub repo, then deploy it at share.streamlit.io.

## Using it
- Sidebar: pick a pattern, enter a history length (months or years), choose S&P 500, Nasdaq-100, or your own tickers, hit Scan.
- Results table: every stock where the pattern appeared in that window, newest signal first.
- Pick a stock to see its candlestick chart (pattern points and neckline marked) and key stats.
- Advanced: "Only signals from the last N days" narrows long lookbacks to fresh setups.

First scan of the full S&P 500 takes a minute or two; results are cached for an hour.
Data comes from Yahoo Finance via yfinance. If a scan returns nothing, Yahoo may be rate-limiting; wait a minute.
