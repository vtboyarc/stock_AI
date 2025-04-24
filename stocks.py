#!/usr/bin/env python3
"""
stocks.py

An AI-powered financial analyst CLI and built-in HTTP server with:
1) /api/data  → returns {name, price, recommendation, factors}
2) /api/analyze → returns {analysis}
NaN factors are sanitized to null; missing values default the local recommendation to HOLD.
"""

import os
import sys
import json
import math
from datetime import datetime

import openai
import yfinance as yf
import pandas as pd

# —————————————————————————————————————————————————————————————————————
# 1) Configure OpenAI
# —————————————————————————————————————————————————————————————————————
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

# —————————————————————————————————————————————————————————————————————
# 2) Fetch company info, price, history
# —————————————————————————————————————————————————————————————————————
def fetch_data(ticker: str):
    asset = yf.Ticker(ticker)
    # company name
    try:
        info = asset.info or {}
    except Exception:
        info = {}
    name = info.get("shortName") or info.get("longName") or ticker

    # 6mo history
    hist = asset.history(period="6mo")
    if hist.empty:
        raise ValueError(f"No data for ticker '{ticker}'")
    price = float(hist["Close"].iloc[-1])
    return name, price, hist

# —————————————————————————————————————————————————————————————————————
# 3) Compute technical indicators
# —————————————————————————————————————————————————————————————————————
def compute_indicators(hist: pd.DataFrame):
    df = hist.copy()
    start = df["Close"].iloc[0]
    df["Return_6mo"] = (df["Close"] / start - 1) * 100

    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + avg_gain/avg_loss))

    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_UP"] = mid + 2 * std
    df["BB_LO"] = mid - 2 * std

    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low  - prev_close).abs()
    df["ATR_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    L = df.iloc[-1]
    return {
        "return_6mo": L["Return_6mo"],
        "rsi_14":     L["RSI_14"],
        "sma_50":     L["SMA_50"],
        "sma_200":    L["SMA_200"],
        "macd":       L["MACD"],
        "macd_signal":L["MACD_signal"],
        "bb_upper":   L["BB_UP"],
        "bb_lower":   L["BB_LO"],
        "atr_14":     L["ATR_14"],
    }

# —————————————————————————————————————————————————————————————————————
# 4) Rule-based recommendation with guards
# —————————————————————————————————————————————————————————————————————
def local_recommend(tech: dict) -> str:
    """
    • If SMA or RSI missing → HOLD
    • strong buy if uptrend & RSI < 30
    • strong sell if downtrend & RSI > 70
    • buy if uptrend
    • sell otherwise
    """
    sma50 = tech.get("sma_50")
    sma200 = tech.get("sma_200")
    rsi = tech.get("rsi_14")

    # if any essential indicator missing or NaN → hold
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in (sma50, sma200, rsi)):
        return "hold"

    uptrend = sma50 > sma200
    if uptrend and rsi < 30:
        return "strong buy"
    if not uptrend and rsi > 70:
        return "strong sell"
    if uptrend:
        return "buy"
    return "sell"

# —————————————————————————————————————————————————————————————————————
# 5) Full GPT-4o analysis (rationale)
# —————————————————————————————————————————————————————————————————————
def analyze_ticker(ticker: str) -> str:
    _, price, hist = fetch_data(ticker)
    tech = compute_indicators(hist)
    prompt = f"""
You are a professional financial analyst. Given the data for {ticker} as of {datetime.today().date()}:

— Technical Indicators —
• Current Price: ${price:,.2f}
• 6-Month Return: {tech['return_6mo']:.2f}%
• 14-Day RSI: {tech['rsi_14']:.2f}
• SMA (50/200): ${tech['sma_50']:.2f} / ${tech['sma_200']:.2f}
• MACD: {tech['macd']:.4f} (Signal: {tech['macd_signal']:.4f})
• Bollinger Bands (20d): Lower ${tech['bb_lower']:.2f}, Upper ${tech['bb_upper']:.2f}
• ATR (14-day): {tech['atr_14']:.4f}

Please assign one of: **strong sell**, **sell**, **hold**, **buy**, **strong buy**
and provide a **concise rationale** (2–3 sentences).
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# —————————————————————————————————————————————————————————————————————
# 6) CLI entrypoint
# —————————————————————————————————————————————————————————————————————
def main():
    if len(sys.argv) != 2:
        print("Usage: python stocks.py <TICKER>", file=sys.stderr)
        sys.exit(1)
    ticker = sys.argv[1].upper()
    name, price, hist = fetch_data(ticker)
    tech = compute_indicators(hist)
    rec = local_recommend(tech)
    print(f"\n{name} — Current Price: ${price:,.2f}")
    print(f"Local Recommendation: {rec.title()}")

# —————————————————————————————————————————————————————————————————————
# 7) Built-in HTTP server
# —————————————————————————————————————————————————————————————————————
def run_server(port: int = 8000):
    import http.server, socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Methods', 'POST,OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length) or b'{}')
            ticker = payload.get('ticker', '').strip().upper()
            if not ticker:
                return self.send_error(400, "Missing 'ticker'")

            # /api/data
            if self.path == '/api/data':
                try:
                    name, price, hist = fetch_data(ticker)
                    tech_raw = compute_indicators(hist)
                    # sanitize NaN → None
                    tech = {
                        k: (None if (v is None or (isinstance(v, float) and math.isnan(v))) else v)
                        for k, v in tech_raw.items()
                    }
                    rec = local_recommend(tech)
                    resp_obj = {
                        'name': name,
                        'price': price,
                        'recommendation': rec,
                        'factors': tech
                    }
                    body = json.dumps(resp_obj).encode()
                    print(f"[DEBUG] /api/data {ticker} → {resp_obj}")
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as e:
                    print(f"[ERROR] /api/data {ticker}: {e}", file=sys.stderr)
                    self.send_error(500, str(e))

            # /api/analyze
            elif self.path == '/api/analyze':
                try:
                    analysis = analyze_ticker(ticker)
                    resp_obj = {'analysis': analysis}
                    body = json.dumps(resp_obj).encode()
                    print(f"[DEBUG] /api/analyze {ticker}")
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as e:
                    print(f"[ERROR] /api/analyze {ticker}: {e}", file=sys.stderr)
                    self.send_error(500, str(e))
            else:
                self.send_error(404)

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"▶ Serving static files & API at http://localhost:{port}/")
        httpd.serve_forever()

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'serve':
        run_server()
    else:
        main()