import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 網頁基礎配置 ---
st.set_page_config(page_title="台股實價量化觀測站", layout="wide")

# --- 1. 技術指標計算 (確保使用原始價格) ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    # 移動平均線
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    # RSI (5)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    df['RSI5'] = 100 - (100 / (1 + (gain / loss)))
    # KD (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    return df

# --- 2. 訊號標記邏輯 (修正價格偏移) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 買入條件：MA金叉 或 KD低檔金叉 或 RSI站上50
        if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) or \
           (row['K'] < 25 and prev_row['K'] <= prev_row['D'] and row['K'] > row['D']) or \
           (prev_row['RSI5'] <= 50 and row['RSI5'] > 50):
            # 買點標註在當日最低價 (Low) 的下方
            buy_markers[i] = row['Low'] * 0.985
            
        # 賣出條件：跌破MA5 或 KD高檔死叉 或 RSI跌破50
        elif (row['Close'] < row['MA5']) or \
             (row['K'] > 75 and prev_row['K'] >= prev_row['D'] and row['K'] < row['D']) or \
             (prev_row['RSI5'] >= 50 and row['RSI5'] < 50):
            # 賣點標註在當日最高價 (High) 的上方
            sell_markers[i] = row['High'] * 1.015
            
    return buy_markers, sell_markers

# --- 3. UI 介面與數據抓取 ---
st.title("📊 台股實價對齊監控系統")

symbol = st.sidebar.text_input("輸入代碼", value="2330")
if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

# 週期切換按鈕
cols = st.columns(6)
periods = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}
for i, (lab, val) in enumerate(periods.items()):
    if cols[i].button(lab):
        st.session_state.view_days = val

if symbol:
    with st.spinner("同步市場實價中..."):
        # 核心修正：auto_adjust=False 確保抓到的是真實市價
        df = yf.download(f"{symbol}.TW", period="2y", auto_adjust=False)
        if df.empty:
            df = yf.download(f"{symbol}.TWO", period="2y", auto_adjust=False)
            
        if not df.empty:
            # 解決 yfinance v1.3.0+ 的多重索引標籤問題
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 清理並統一欄位名稱
            df.columns = [str(c).strip().capitalize() for c in df.columns]
            
            # 計算指標與訊號
            df = calculate_indicators(df)
            df['Buy_Sig'], df['Sell_Sig'] = get_signal_markers(df)
            
            # 擷取顯示視窗
            view_df = df.tail(st.session_state.view_days)
            
            # --- Plotly 繪圖 ---
            fig = go.Figure()
            
            # K線圖
            fig.add_trace(go.Candlestick(
                x=view_df.index, 
                open=view_df['Open'], high=view_df['High'],
                low=view_df['Low'], close=view_df['Close'], 
                name='價格'
            ))
            
            # 均線
            fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA5'], name='MA5', line=dict(color='#FFA500', width=1.2)))
            fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA10'], name='MA10', line=dict(color='#00FFFF', width=1.2)))
            
            # 買入標記 (精準對位)
            fig.add_trace(go.Scatter(
                x=view_df.index, y=view_df['Buy_Sig'],
                mode='markers', name='買入',
                marker=dict(symbol='triangle-up', size=13, color='#FF3131')
            ))
            
            # 賣出標記 (精準對位)
            fig.add_trace(go.Scatter(
                x=view_df.index, y=view_df['Sell_Sig'],
                mode='markers', name='賣出',
                marker=dict(symbol='triangle-down', size=13, color='#39FF14')
            ))
            
            fig.update_layout(
                height=650,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                yaxis=dict(autorange=True, fixedrange=False, tickformat='.2f'),
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 今日數據卡片
            curr = view_df.iloc[-1]
            st.write("#### 📝 今日結算摘要")
            c1, c2, c3 = st.columns(3)
            c1.metric("當前收盤", f"{curr['Close']:.2f}")
            c2.metric("MA5 支撐", f"{curr['MA5']:.2f}")
            c3.metric("RSI(5) 強度", f"{curr['RSI5']:.1f}")
        else:
            st.error("找不到該股票代碼，請確認輸入是否正確。")
