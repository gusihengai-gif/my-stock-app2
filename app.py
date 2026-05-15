import pandas as pd
import pandas_ta as ta  # 確保這行在 yfinance 之前
import yfinance as yf

# 設定網頁標題與寬度
st.set_page_config(page_title="台股量化交易策略觀測站", layout="wide")

st.title("🚀 台股真實價格量化策略網頁版")
st.markdown("本系統採用真實還原權值價格計算，自動過濾除權息干擾。")

# 📥 側邊欄：使用者輸入與參數設定
st.sidebar.header("⚙️ 設定參數")
user_input = st.sidebar.text_input("請輸入台股代碼 (例如: 2330, 0050, 8069)", value="2330").strip()
period_choice = st.sidebar.selectbox("選取資料歷史範圍", ["1y", "6m", "2y"], index=0)

def calculate_indicators(df):
    """計算所有策略所需的技術指標"""
    df = df.astype(float)
    df['MA5'] = ta.sma(df['Close'], length=5)
    df['MA10'] = ta.sma(df['Close'], length=10)
    
    kdj = ta.kdj(df['High'], df['Low'], df['Close'], length=9, signal=3, bounded=True)
    df['K'] = kdj['K_9_3']
    df['D'] = kdj['D_9_3']
    df['J'] = kdj['J_9_3']
    
    df['RSI5'] = ta.rsi(df['Close'], length=5)
    df['RSI10'] = ta.rsi(df['Close'], length=10)
    return df

def check_signals(df):
    """根據你的條件判斷買入與賣出訊號"""
    signals = []
    for i in range(11, len(df)):
        buy_ma = buy_kd = buy_kdj = buy_rsi = False
        sell_ma = sell_kd = sell_kdj = sell_rsi = False
        
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if pd.isna(row['MA10']) or pd.isna(row['J']) or pd.isna(row['RSI10']):
            continue

        # --- 買入條件 ---
        if prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']:
            buy_ma = True
        if row['K'] < 20 and row['D'] < 20 and prev_row['K'] <= prev_row['D'] and row['K'] > row['D']:
            buy_kd = True
        if prev_row['J'] < 0 and row['J'] > prev_row['J'] and row['J'] > row['K'] and row['J'] > row['D']:
            buy_kdj = True
        if (20 <= prev_row['RSI5'] <= 30) and (prev_row['RSI5'] <= prev_row['RSI10']) and (row['RSI5'] > row['RSI10']):
            buy_rsi = True
        elif row['RSI5'] > 50 and prev_row['RSI5'] <= 50:
            buy_rsi = True

        # --- 賣出條件 ---
        if row['Close'] < row['MA5'] or (row['MA5'] < row['MA10']):
            sell_ma = True
        if row['K'] > 80 and row['D'] > 80 and prev_row['K'] >= prev_row['D'] and row['K'] < row['D']:
            sell_kd = True
        if prev_row['J'] > 100 and (row['J'] < 100 or (row['J'] < row['K'] and row['J'] < row['D'])):
            sell_kdj = True
        if ((70 <= prev_row['RSI5'] <= 80) and (prev_row['RSI5'] >= prev_row['RSI10']) and (row['RSI5'] < row['RSI10'])) or (row['RSI5'] < 50):
            sell_rsi = True

        buy_triggered = any([buy_ma, buy_kd, buy_kdj, buy_rsi])
        sell_triggered = any([sell_ma, sell_kd, sell_kdj, sell_rsi])
        
        if buy_triggered and not sell_triggered:
            signals.append({"日期": df.index[i].strftime('%Y-%m-%d'), "動作": "🔴 買入 (BUY)", "價格": round(row['Close'], 2)})
        elif sell_triggered and not buy_triggered:
            signals.append({"日期": df.index[i].strftime('%Y-%m-%d'), "動作": "🟢 賣出 (SELL)", "價格": round(row['Close'], 2)})
            
    return signals

# 🛠️ 執行資料撈取邏輯
if user_input:
    with st.spinner("正在下載並分析數據中..."):
        # 嘗試上市 (.TW)
        ticker_tw = f"{user_input}.TW"
        data = yf.download(ticker_tw, period=period_choice, interval="1d", auto_adjust=True, progress=False)
        target_ticker = ticker_tw
        
        # 失敗則嘗試上櫃 (.TWO)
        if data.empty:
            ticker_two = f"{user_input}.TWO"
            data = yf.download(ticker_two, period=period_choice, interval="1d", auto_adjust=True, progress=False)
            target_ticker = ticker_two

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            st.error(f"❌ 在台股市場找不到代碼 【{user_input}】，請檢查是否輸入正確。")
        else:
            st.success(f"📊 成功分析標的：{target_ticker}")
            
            # 計算指標
            data = calculate_indicators(data)
            latest = data.iloc[-1]
            
            # --- 區塊 1：最新狀態卡片 ---
            st.subheader("📌 最新收盤狀態統計")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("真實收盤價", f"{latest['Close']:.2f} 元")
            col2.metric("MA 狀態", f"MA5: {latest['MA5']:.1f}", f"MA10: {latest['MA10']:.1f}")
            col3.metric("KDJ 狀態", f"J 線: {latest['J']:.1f}", f"K:{latest['K']:.1f} / D:{latest['D']:.1f}")
            col4.metric("RSI5 數值", f"{latest['RSI5']:.1f}")
            
            # --- 區塊 2：走勢圖表 ---
            st.subheader("📈 歷史還原收盤價走勢")
            st.line_chart(data['Close'])
            
            # --- 區塊 3：策略訊號輸出 ---
            st.subheader("🚩 策略觸發歷史訊號 (僅顯示最新 10 筆)")
            signal_list = check_signals(data)
            
            if not signal_list:
                st.info("該股票在觀測期間內未觸發任何訊號。")
            else:
                signal_df = pd.DataFrame(signal_list)
                # 倒序排列，讓最新的訊號在最上面
                st.dataframe(signal_df.iloc[::-1].head(10), use_container_width=True)
