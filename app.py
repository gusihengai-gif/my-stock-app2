import streamlit as st
import pandas as pd
import yfinance as yf

# 設定網頁標題
st.set_page_config(page_title="台股量化交易策略觀測站", layout="wide")

st.title("🚀 台股真實價格量化策略 (穩定部署版)")

# --- 自定義指標計算公式 (取代 pandas-ta) ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    
    # 1. 移動平均線 MA
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    # 2. RSI 計算
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI5'] = compute_rsi(df['Close'], 5)
    df['RSI10'] = compute_rsi(df['Close'], 10)
    
    # 3. KDJ 計算 (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    
    df['K'] = rsv.ewm(com=2, adjust=False).mean() # 這裡用 ewm 模擬平滑
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

# --- 訊號判斷邏輯 (與原本一致) ---
def check_signals(df):
    signals = []
    for i in range(11, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if pd.isna(row['MA10']) or pd.isna(row['J']) or pd.isna(row['RSI10']):
            continue

        buy_triggered = False
        sell_triggered = False

        # 買入條件
        if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) or \
           (row['K'] < 20 and row['D'] < 20 and prev_row['K'] <= prev_row['D'] and row['K'] > row['D']) or \
           (prev_row['J'] < 0 and row['J'] > prev_row['J'] and row['J'] > row['K']) or \
           ((20 <= prev_row['RSI5'] <= 30) and row['RSI5'] > row['RSI10']) or \
           (row['RSI5'] > 50 and prev_row['RSI5'] <= 50):
            buy_triggered = True

        # 賣出條件
        if (row['Close'] < row['MA5']) or (row['MA5'] < row['MA10']) or \
           (row['K'] > 80 and row['D'] > 80 and row['K'] < row['D']) or \
           (prev_row['J'] > 100 and row['J'] < 100) or \
           (row['RSI5'] < 50):
            sell_triggered = True

        if buy_triggered and not sell_triggered:
            signals.append({"日期": df.index[i].strftime('%Y-%m-%d'), "動作": "🔴 買入", "價格": round(row['Close'], 2)})
        elif sell_triggered and not buy_triggered:
            signals.append({"日期": df.index[i].strftime('%Y-%m-%d'), "動作": "🟢 賣出", "價格": round(row['Close'], 2)})
            
    return signals

# --- Streamlit UI 介面 ---
user_input = st.sidebar.text_input("輸入台股代碼", value="2330")
if user_input:
    with st.spinner("獲取真實數據中..."):
        # 自動判斷上市或上櫃
        ticker = f"{user_input}.TW"
        data = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        if data.empty:
            ticker = f"{user_input}.TWO"
            data = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            df_final = calculate_indicators(data)
            st.success(f"已讀取 {ticker} 最新數據")
            
            # 顯示資訊卡片
            latest = df_final.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("收盤價", f"{latest['Close']:.2f}")
            c2.metric("RSI5", f"{latest['RSI5']:.1f}")
            c3.metric("K/D", f"{latest['K']:.1f} / {latest['D']:.1f}")
            
            st.line_chart(df_final['Close'])
            
            sig_list = check_signals(df_final)
            if sig_list:
                st.write("### 🚩 最近策略訊號")
                st.table(pd.DataFrame(sig_list).iloc[::-1].head(10))
        else:
            st.error("找不到該股票代碼，請檢查輸入。")
