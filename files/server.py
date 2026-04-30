"""
DayTrade Terminal - Flask Backend Server
yfinanceで日本株の1分足データを取得してフロントエンドに提供します。

起動方法:
  pip install flask flask-cors yfinance pandas numpy
  python server.py
"""

import json
import os
from datetime import datetime, timedelta
import concurrent.futures

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # フロントエンドからのCORSリクエストを許可

CONFIG_FILE = 'config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'stocks': [], 'line_token': ''}

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

config_data = load_config()
WATCH_STOCKS = config_data.get('stocks', [])


# ===== テクニカル指標計算 =====

def calc_sma(closes, period):
    """単純移動平均"""
    result = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        result[i] = float(np.mean(closes[i - period + 1:i + 1]))
    return result


def calc_bollinger(closes, period=20, mult=2.0):
    """ボリンジャーバンド (SMA, Upper, Lower, %B, BandWidth)"""
    sma = calc_sma(closes, period)
    upper, lower, pct_b, bandwidth = [], [], [], []
    for i in range(len(closes)):
        if sma[i] is None:
            upper.append(None)
            lower.append(None)
            pct_b.append(None)
            bandwidth.append(None)
        else:
            std = float(np.std(closes[i - period + 1:i + 1], ddof=0))
            u = sma[i] + mult * std
            l = sma[i] - mult * std
            upper.append(u)
            lower.append(l)
            bw = (u - l) / sma[i] * 100 if sma[i] != 0 else None
            pb = (closes[i] - l) / (u - l) * 100 if (u - l) != 0 else 50
            bandwidth.append(bw)
            pct_b.append(pb)
    return {"sma": sma, "upper": upper, "lower": lower, "pct_b": pct_b, "bandwidth": bandwidth}


def calc_rsi(closes, period=14):
    """RSI"""
    result = [None] * len(closes)
    if len(closes) < period + 1:
        return result
    for i in range(period, len(closes)):
        diffs = [closes[j] - closes[j - 1] for j in range(i - period + 1, i + 1)]
        gains = sum(d for d in diffs if d > 0)
        losses = sum(-d for d in diffs if d < 0)
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = float(100 - 100 / (1 + rs))
    return result


def calc_macd(closes, fast=12, slow=26, signal=9):
    """MACD"""
    def ema(data, span):
        s = [None] * len(data)
        k = 2 / (span + 1)
        for i in range(len(data)):
            if i == 0:
                s[i] = data[i]
            elif s[i - 1] is None:
                s[i] = data[i]
            else:
                s[i] = data[i] * k + s[i - 1] * (1 - k)
        return s

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [
        (f - s) if f is not None and s is not None else None
        for f, s in zip(ema_fast, ema_slow)
    ]
    valid_macd = [m if m is not None else 0.0 for m in macd_line]
    signal_line = ema(valid_macd, signal)
    histogram = [
        (m - s) if m is not None and s is not None else None
        for m, s in zip(macd_line, signal_line)
    ]
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

def calc_atr(candles, period=14):
    """ATR (Average True Range)"""
    n = len(candles)
    tr = [0.0] * n
    atr = [None] * n
    if n == 0:
        return atr
    
    for i in range(1, n):
        h = candles[i]['high']
        l = candles[i]['low']
        pc = candles[i - 1]['close']
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        
    if n > period:
        atr[period] = sum(tr[1:period + 1]) / period
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr

def calc_vwap(df):
    """当日のVWAPを計算"""
    vwaps = []
    if df.empty:
        return vwaps
    
    # 日付ごとにグループ化して累積を計算
    df['Date'] = df.index.date
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['PV'] = df['Typical_Price'] * df['Volume']
    
    cum_pv = df.groupby('Date')['PV'].cumsum()
    cum_vol = df.groupby('Date')['Volume'].cumsum()
    
    vwap_values = cum_pv / cum_vol
    return vwap_values.tolist()

def calc_signals(candles, bb, rsi):
    """BB + RSI の複合シグナル生成"""
    signals = []
    n = len(candles)
    last_sig_idx = -10
    for i in range(20, n):
        if bb["lower"][i] is None or rsi[i] is None:
            continue
        c = candles[i]
        prev = candles[i - 1]
        touch_lower = prev["low"] <= bb["lower"][i - 1] and c["close"] > bb["lower"][i]
        touch_upper = prev["high"] >= bb["upper"][i - 1] and c["close"] < bb["upper"][i]
        rsi_os = rsi[i] < 40
        rsi_ob = rsi[i] > 60
        gap = i - last_sig_idx
        if (touch_lower or rsi_os) and gap >= 5:
            signals.append({"type": "buy", "idx": i, "price": c["low"], "rsi": round(rsi[i], 1)})
            last_sig_idx = i
        elif (touch_upper or rsi_ob) and gap >= 5:
            signals.append({"type": "sell", "idx": i, "price": c["high"], "rsi": round(rsi[i], 1)})
            last_sig_idx = i
    return signals


# ===== データ取得 =====

def fetch_data(ticker_symbol, interval="1m", bars=70):
    """yfinanceでデータを取得"""
    tk = yf.Ticker(ticker_symbol)
    
    # 間隔によってperiodを調整
    if interval in ["1m", "2m", "5m"]:
        period = "5d"
    elif interval in ["15m", "30m", "60m", "90m"]:
        period = "1mo"
    elif interval in ["1d", "5d", "1wk", "1mo", "3mo"]:
        period = "1y"
    else:
        period = "5d"
        
    df = tk.history(period=period, interval=interval)
    if df.empty:
        # 市場時間外の場合は長めの期間でリトライ
        if interval == "1m":
            df = tk.history(period="7d", interval="1m")
    if df.empty:
        return None, None, "データ取得失敗"
        
    vwap = calc_vwap(df)
    
    # 最新のbars本を使用
    df = df.tail(bars).copy()
    vwap = vwap[-bars:] if len(vwap) > bars else vwap
    
    try:
        df.index = df.index.tz_convert("Asia/Tokyo")
    except Exception:
        pass  # 日足等の場合はTZ情報がないことがある
        
    candles = []
    for i, (ts, row) in enumerate(df.iterrows()):
        t_str = ts.strftime("%H:%M") if "m" in interval or "h" in interval else ts.strftime("%m/%d")
        candles.append({
            "time": t_str,
            "open":  round(float(row["Open"]),  2),
            "high":  round(float(row["High"]),  2),
            "low":   round(float(row["Low"]),   2),
            "close": round(float(row["Close"]), 2),
            "vol":   int(row["Volume"]),
            "vwap":  round(float(vwap[i]), 2) if i < len(vwap) and not np.isnan(vwap[i]) else None
        })
    return candles, df, None

def get_stock_overview(ticker_symbol):
    """銘柄の基本情報"""
    try:
        tk = yf.Ticker(ticker_symbol)
        info = tk.fast_info
        return {
            "marketCap": getattr(info, "market_cap", None),
            "fiftyTwoWeekHigh": getattr(info, "year_high", None),
            "fiftyTwoWeekLow": getattr(info, "year_low", None),
        }
    except Exception:
        return {}


# ===== API エンドポイント =====

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    """全銘柄の最新価格・変化率を返す（スキャン用）"""
    global WATCH_STOCKS
    WATCH_STOCKS = load_config().get('stocks', [])
    result = []
    
    def fetch_stock_data(s):
        try:
            candles, _, err = fetch_data(s["ticker"], interval="1m", bars=30)
            if err or not candles:
                return {**s, "price": None, "change": None, "error": err or "no data"}
            closes = [c["close"] for c in candles]
            first_open = candles[0]["open"]
            last_close = candles[-1]["close"]
            change_pct = (last_close - first_open) / first_open * 100 if first_open else 0
            sma10 = calc_sma(closes, min(10, len(closes)))
            trend_slope = 0
            if sma10[-1] is not None and sma10[-5] is not None:
                trend_slope = sma10[-1] - sma10[-5]
            return {
                "code": s["code"],
                "ticker": s["ticker"],
                "name": s["name"],
                "price": last_close,
                "change": round(change_pct, 2),
                "trendSlope": round(trend_slope, 4),
                "volume": candles[-1]["vol"],
                "time": candles[-1]["time"],
            }
        except Exception as e:
            return {**s, "price": None, "change": None, "error": str(e)}

    # 並列処理で全銘柄のデータを取得
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_stock_data, s) for s in WATCH_STOCKS]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            result.append(res)
            
    # 上昇トレンド順（傾き降順）
    result.sort(key=lambda x: x.get("trendSlope", -999), reverse=True)
    return jsonify({"status": "ok", "data": result, "fetchedAt": datetime.now().isoformat()})


@app.route("/api/chart/<code>", methods=["GET"])
def get_chart(code):
    """指定銘柄のチャートデータ（BB・RSI・MACD・シグナル・VWAP・ATR含む）"""
    interval = request.args.get('interval', '1m')
    stock = next((s for s in WATCH_STOCKS if s["code"] == code), None)
    if not stock:
        return jsonify({"status": "error", "message": f"銘柄コード {code} は未登録です"}), 404

    candles, _, err = fetch_data(stock["ticker"], interval=interval, bars=70)
    if err or not candles:
        return jsonify({"status": "error", "message": err or "データなし"}), 500

    closes = [c["close"] for c in candles]
    bb     = calc_bollinger(closes, 20, 2)
    rsi    = calc_rsi(closes, 14)
    macd   = calc_macd(closes)
    atr    = calc_atr(candles, 14)
    signals = calc_signals(candles, bb, rsi)
    info   = get_stock_overview(stock["ticker"])

    # 最新値
    last = candles[-1]
    last_rsi = next((v for v in reversed(rsi) if v is not None), None)
    last_bbu = next((v for v in reversed(bb["upper"]) if v is not None), None)
    last_bbl = next((v for v in reversed(bb["lower"]) if v is not None), None)
    last_sma = next((v for v in reversed(bb["sma"]) if v is not None), None)
    last_pb = next((v for v in reversed(bb["pct_b"]) if v is not None), None)

    return jsonify({
        "status":  "ok",
        "code":    code,
        "name":    stock["name"],
        "ticker":  stock["ticker"],
        "candles": candles,
        "indicators": {
            "bb":   bb,
            "rsi":  rsi,
            "macd": macd,
        },
        "signals": signals,
        "summary": {
            "price": last["close"],
            "open": last["open"],
            "high": last["high"],
            "low": last["low"],
            "volume": last["vol"],
            "vwap": last.get("vwap"),
            "atr": round(atr[-1], 2) if len(atr) > 0 and atr[-1] is not None else None,
            "rsi": round(last_rsi, 1) if last_rsi is not None else None,
            "bb_upper": round(last_bbu, 1) if last_bbu is not None else None,
            "bb_lower": round(last_bbl, 1) if last_bbl is not None else None,
            "bb_mid": round(last_sma, 1) if last_sma is not None else None,
            "pct_b": round(last_pb, 1) if last_pb is not None else None,
            "year_high": info.get("fiftyTwoWeekHigh"),
            "year_low": info.get("fiftyTwoWeekLow"),
        },
        "fetchedAt": datetime.now().isoformat(),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

def resolve_stock_info(code):
    """銘柄コードからティッカーと名前を解決する"""
    ticker = f"{code}.T"
    try:
        tk = yf.Ticker(ticker)
        # infoは取得に時間がかかる場合があるため、最小限の試行
        name = tk.info.get('shortName') or tk.info.get('longName') or f"銘柄 {code}"
    except Exception:
        name = f"銘柄 {code}"
    return {"code": code, "ticker": ticker, "name": name}

@app.route("/api/settings", methods=["GET", "POST"])
def settings_api():
    global WATCH_STOCKS
    if request.method == "GET":
        return jsonify({"status": "ok", "data": load_config()})
    else:
        data = request.json
        line_token = data.get("line_token", "")
        
        new_stocks = []
        if "codes" in data:
            for c in data["codes"]:
                if not c: continue
                # 既存のキャッシュにあればそれを使う（高速化）
                existing = next((s for s in WATCH_STOCKS if s["code"] == c), None)
                if existing:
                    new_stocks.append(existing)
                else:
                    new_stocks.append(resolve_stock_info(c))
        else:
            new_stocks = data.get("stocks", [])
            
        new_conf = {"stocks": new_stocks, "line_token": line_token}
        save_config(new_conf)
        WATCH_STOCKS = new_stocks
        return jsonify({"status": "ok"})

@app.route("/api/notify", methods=["POST"])
def notify_api():
    data = request.json
    message = data.get("message", "")
    config = load_config()
    token = config.get("line_token", "")
    if not token:
        return jsonify({"status": "error", "message": "LINE Token not configured."}), 400
    
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"message": message}
    try:
        r = requests.post(url, headers=headers, data=payload)
        if r.status_code == 200:
            return jsonify({"status": "ok"})
        else:
            return jsonify({"status": "error", "message": r.text}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/backtest/<code>", methods=["GET"])
def backtest_api(code):
    stock = next((s for s in WATCH_STOCKS if s["code"] == code), None)
    if not stock:
        return jsonify({"status": "error", "message": "銘柄が見つかりません"}), 404
    
    interval = request.args.get('interval', '15m')
    candles, df, err = fetch_data(stock["ticker"], interval=interval, bars=300)  # 300本でテスト
    if err or not candles:
        return jsonify({"status": "error", "message": err}), 500
    
    closes = [c["close"] for c in candles]
    bb = calc_bollinger(closes, 20, 2)
    rsi = calc_rsi(closes, 14)
    sigs = calc_signals(candles, bb, rsi)
    
    trades = []
    pos = None
    for sig in sigs:
        if sig["type"] == "buy" and pos is None:
            pos = sig
        elif sig["type"] == "sell" and pos is not None:
            pl = sig["price"] - pos["price"]
            pl_pct = pl / pos["price"] * 100
            trades.append({"entry": pos["price"], "exit": sig["price"], "pl": pl, "pl_pct": pl_pct})
            pos = None
            
    win_trades = [t for t in trades if t["pl"] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    total_pl_pct = sum(t["pl_pct"] for t in trades)
    
    return jsonify({
        "status": "ok",
        "code": code,
        "interval": interval,
        "trades_count": len(trades),
        "win_rate": round(win_rate, 2),
        "total_pl_pct": round(total_pl_pct, 2)
    })

STATE_FILE = 'state.json'

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"positions": {}, "signals": []}
    return {"positions": {}, "signals": []}

def save_state(state):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

@app.route("/api/state", methods=["GET", "POST"])
def state_api():
    if request.method == "GET":
        return jsonify({"status": "ok", "data": load_state()})
    else:
        try:
            data = request.json
            save_state(data)
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("=" * 50)
    print("  DayTrade Terminal - Backend Server")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
