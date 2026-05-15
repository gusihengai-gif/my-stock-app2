import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

# --- 網頁基礎設定 ---
st.set_page_config(page_title="台股量化策略觀測站", layout="wide")

st.title("📈 台股量化交易策略觀測站")
st.caption("使用說明：輸入台股代碼後，系統將自動抓取 Yahoo Finance 真實數據並計算買賣訊號。")

# --- 自定義指標計算 (不依賴 pandas-ta) ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    
    # 1. 移動平均線 (MA)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    # 2. RSI 計算 (標準 Wilder 算法)
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    df['RSI5'] = compute_rsi(df['Close'], 5)
    df['RSI10'] = compute_rsi(df['Close'], 10)
    
    # 3. KDJ 計算 (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

# --- 策略訊號判斷 ---
def check_signals(df):
    signals = []
    # 至少需要 15 天數據才能計算指標
    for i in range(15, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        buy_triggered = False
        sell_triggered = False

        # --- 買入條件邏輯 ---
        # 1. MA 黃金交叉
        if prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']:
            buy_triggered = True
        # 2. KD 低檔金叉
        elif row['K'] < 25 and row['D'] < 25 and prev_row['K'] <= prev_row['D'] and row['K'] > row['D']:
            buy_triggered = True
        # 3. RSI 黃金交叉且站上 50
        elif row['RSI5'] > 50 and prev_row['RSI5'] <= 50:
            buy_triggered = True

        # --- 賣出條件邏輯 ---
        # 1. 價格跌破 MA5 或 MA5 轉弱
        if row['Close'] < row['MA5'] or row['MA5'] < row['MA10']:
            sell_triggered = True
        # 2. KD 高檔死叉
        elif row['K'] > 75 and row['D'] > 75 and prev_row['K'] >= prev_row['D'] and row['K'] < row['D']:
            sell_triggered = True
        # 3. RSI 轉弱 (跌破 50)
        elif row['RSI5'] < 50:
            sell_triggered = True

        date_str = df.index[i].strftime('%Y-%m-%d')
        if buy_triggered and not sell_triggered:
            signals.append({"日期": date_str, "動作": "🔴 買入", "價格": round(float(row['Close']), 2), "原因": "技術指標轉強"})
        elif sell_triggered and not buy_triggered:
            signals.append({"日期": date_str, "動作": "🟢 賣出", "價格": round(float(row['Close']), 2), "原因": "技術指標轉弱"})
            
    return signals

# --- 側邊欄：輸入區 ---
st.sidebar.header("設定")
symbol = st.sidebar.text_input("輸入台股代碼", value="2330")
period = st.sidebar.selectbox("觀測期間", ["1y", "2y", "6m"], index=0)

if symbol:
    with st.spinner(f"正在分析 {symbol} ..."):
        # 嘗試上市代碼
        data = yf.download(f"{symbol}.TW", period=period, auto_adjust=True)
        # 若無資料則嘗試上櫃代碼
        if data.empty:
            data = yf.download(f"{symbol}.TWO", period=period, auto_adjust=True)

        if not data.empty:
            # 修正 yfinance 可能產生的多重索引問題
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # 標準化欄位名稱
            data.columns = [str(c).capitalize() for c in data.columns]
            
            # 計算指標
            df_processed = calculate_indicators(data)
            
            # --- 儀表板顯示 ---
            latest = df_processed.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("當前股價", f"{float(latest['Close']):.2f}")
            c2.metric("MA5 / MA10", f"{float(latest['MA5']):.1f}", f"{float(latest['MA5']-latest['MA10']):.1f}")
            c3.metric("RSI(5)", f"{float(latest['RSI5']):.1f}")
            c4.metric("K / D", f"{float(latest['K']):.1f} / {float(latest['D']):.1f}")

            # 股價走勢圖
            st.subheader("價格走勢與移動平均線")
            st.line_chart(df_processed[['Close', 'MA5', 'MA10']])

            # 訊號列表
            sig_results = check_signals(df_processed)
            if sig_results:
                st.subheader("🚩 最近交易訊號紀錄")
                # 倒序顯示最近 10 筆
                st.table(pd.DataFrame(sig_results).iloc[::-1].head(10))
            else:
                st.info("目前尚無明確的買賣訊號。")
        else:
            st.error("無法取得數據。請確認輸入代碼是否正確（例：上市輸入 2330，上櫃輸入 8069）。")
