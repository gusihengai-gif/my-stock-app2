import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 網頁配置 ---
st.set_page_config(page_title="台股精準訊號觀測站", layout="wide")

# --- 1. 計算指標函數 ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    # 計算 MA (使用原始 Close)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    # RSI (5)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    df['RSI5'] = 100 - (100 / (1 + (gain / loss)))
    
    # KDJ (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    return df

# --- 2. 策略訊號標記 (修正價格不符問題) ---
def get_signal_markers(df):
    # 建立與 df 等長的空陣列
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 買入條件邏輯
        if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) or \
           (row['K'] < 25 and prev_row['K'] <= df.iloc[i-1]['D'] and row['K'] > row['D']) or \
           (prev_row['RSI5'] <= 50 and row['RSI5'] > 50):
            # 買點：精準鎖定當日最低價下方
            buy_markers[i] = row['Low'] * 0.99 
            
        # 賣出條件邏輯
        elif (row['Close'] < row['MA5']) or \
             (row['K'] > 75 and prev_row['K'] >= df.iloc[i-1]['D'] and row['K'] < row['D']) or \
             (prev_row['RSI5'] >= 50 and row['RSI5'] < 50):
            # 賣點：精準鎖定當日最高價上方
            sell_markers[i] = row['High'] * 1.01
            
    return buy_markers, sell_markers

# --- 3. UI 主程式 ---
st.title("📊 台股實價訊號對齊版")

symbol = st.sidebar.text_input("輸入代碼 (2330)", value="2330")
if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

# 天數按鈕
p_cols = st.columns(6)
for i, (lab, val) in enumerate({"10天":10, "20天":20, "30天":30, "60天":60, "120天":120, "240天":240}.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

if symbol:
    # 關鍵修正：auto_adjust=False 以獲取真實市價
    df = yf.download(f"{symbol}.TW", period="2y", auto_adjust=False)
    if df.empty:
        df = yf.download(f"{symbol}.TWO", period="2y", auto_adjust=False)
        
    if not df.empty:
        # 扁平化處理 yfinance 的資料結構
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = calculate_indicators(df)
        df['Buy_Marker'], df['Sell_Marker'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days)
        
        # --- 繪圖 ---
        fig = go.Figure()
        
        # K線 (使用真實 Open, High, Low, Close)
        fig.add_trace(go.Candlestick(
            x=view_df.index, 
            open=view_df['Open'], high=view_df['High'],
            low=view_df['Low'], close=view_df['Close'], 
            name='真實價格'
        ))
        
        # 均線 (MA5, MA10)
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA5'], name='MA5', line=dict(color='#FFA500', width=1.5)))
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA10'], name='MA10', line=dict(color='#00FFFF', width=1.5)))
        
        # 標註買賣點
        fig.add_trace(go.Scatter(
            x=view_df.index, y=view_df['Buy_Marker'],
            mode='markers', name='策略買入',
            marker=dict(symbol='triangle-up', size=14, color='#FF0000')
        ))
        fig.add_trace(go.Scatter(
            x=view_df.index, y=view_df['Sell_Marker'],
            mode='markers', name='策略賣出',
            marker=dict(symbol='triangle-down', size=14, color='#00FF00')
        ))
        
        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            yaxis=dict(autorange=True, fixedrange=False, title="價格 (TWD)"),
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"目前顯示為 {symbol} 的真實交易價格。")
