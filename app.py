import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 網頁配置 ---
st.set_page_config(page_title="台股量化訊號觀測站", layout="wide")

# --- 1. 計算指標函數 ---
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
    
    # KDJ (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

# --- 2. 策略訊號標記函數 ---
def get_signal_markers(df):
    buy_prices = []
    sell_prices = []
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 預設為空值
        buy_val = np.nan
        sell_val = np.nan
        
        # --- 買入條件 (🔴) ---
        # 邏輯：MA金叉 OR KD低檔金叉 OR RSI強勢站上50
        if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) or \
           (row['K'] < 25 and prev_row['K'] <= prev_row['D'] and row['K'] > row['D']) or \
           (prev_row['RSI5'] <= 50 and row['RSI5'] > 50):
            buy_val = row['Low'] * 0.98 # 標註在最低價下方 2%
            
        # --- 賣出條件 (🟢) ---
        # 邏輯：跌破MA5 OR KD高檔死叉 OR RSI跌破50
        elif (row['Close'] < row['MA5']) or \
             (row['K'] > 75 and prev_row['K'] >= prev_row['D'] and row['K'] < row['D']) or \
             (prev_row['RSI5'] >= 50 and row['RSI5'] < 50):
            sell_val = row['High'] * 1.02 # 標註在最高價上方 2%
            
        buy_prices.append(buy_val)
        sell_prices.append(sell_val)
        
    return [np.nan] + buy_prices, [np.nan] + sell_prices

# --- 3. UI 主程式 ---
st.title("🚀 台股自動化訊號 K 線圖")

symbol = st.sidebar.text_input("輸入代碼", value="2330")
st.write("### 觀測週期切換")
p_cols = st.columns(6)
periods = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

for i, (lab, val) in enumerate(periods.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

if symbol:
    df = yf.download(f"{symbol}.TW", period="2y", auto_adjust=True)
    if df.empty:
        df = yf.download(f"{symbol}.TWO", period="2y", auto_adjust=True)
        
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).capitalize() for c in df.columns]
        
        df = calculate_indicators(df)
        df['Buy_Marker'], df['Sell_Marker'] = get_signal_markers(df)
        
        # 截取顯示區間
        view_df = df.tail(st.session_state.view_days)
        
        # --- 繪圖 ---
        fig = go.Figure()
        
        # K線
        fig.add_trace(go.Candlestick(
            x=view_df.index, open=view_df['Open'], high=view_df['High'],
            low=view_df['Low'], close=view_df['Close'], name='K線'
        ))
        
        # 均線
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA5'], name='MA5', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA10'], name='MA10', line=dict(color='cyan', width=1.5)))
        
        # 🔴 買入訊號標記 (朝上三角形)
        fig.add_trace(go.Scatter(
            x=view_df.index, y=view_df['Buy_Marker'],
            mode='markers', name='買入訊號',
            marker=dict(symbol='triangle-up', size=12, color='red', line=dict(width=1, color='white'))
        ))
        
        # 🟢 賣出訊號標記 (朝下三角形)
        fig.add_trace(go.Scatter(
            x=view_df.index, y=view_df['Sell_Marker'],
            mode='markers', name='賣出訊號',
            marker=dict(symbol='triangle-down', size=12, color='lime', line=dict(width=1, color='white'))
        ))
        
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            yaxis=dict(autorange=True, fixedrange=False),
            template="plotly_dark" # 改用深色模式，訊號會更亮眼
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 底部摘要
        st.info(f"💡 目前顯示最近 {st.session_state.view_days} 天交易日。紅色三角形代表策略建議買進，綠色代表建議賣出。")
