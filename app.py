import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. 網頁基礎配置 ---
st.set_page_config(page_title="2364 倫飛實價監控", layout="wide")

# --- 2. 技術指標計算 ---
def calculate_indicators(df):
    df = df.copy().astype(float)
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

# --- 3. 訊號標記邏輯 (僅標記位置，不顯示數值) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 買入觸發
        if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) or \
           (row['K'] < 25 and prev_row['K'] <= prev_row['D'] and row['K'] > row['D']) or \
           (prev_row['RSI5'] <= 50 and row['RSI5'] > 50):
            buy_markers[i] = row['Low'] * 0.985
            
        # 賣出觸發
        elif (row['Close'] < row['MA5']) or \
             (row['K'] > 75 and prev_row['K'] >= prev_row['D'] and row['K'] < row['D']) or \
             (prev_row['RSI5'] >= 50 and row['RSI5'] < 50):
            sell_markers[i] = row['High'] * 1.015
            
    return buy_markers, sell_markers

# --- 4. UI 介面 ---
st.title("📈 台股量化策略觀測站")

symbol = st.sidebar.text_input("輸入代碼 (例: 2364)", value="2364")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測期間切換")
p_cols = st.columns(6)
for i, (lab, val) in enumerate({"10天":10, "20天":20, "30天":30, "60天":60, "120天":120, "240天":240}.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

if symbol:
    data = yf.download(f"{symbol}.TW", period="2y", auto_adjust=False)
    if data.empty:
        data = yf.download(f"{symbol}.TWO", period="2y", auto_adjust=False)
        
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 中文化處理
        df = calculate_indicators(data)
        df['買入訊號'], df['賣出訊號'] = get_signal_markers(df)
        view_df = df.tail(st.session_state.view_days)
        
        # 繪圖
        fig = go.Figure()
        
        # K線 (標題中文化)
        fig.add_trace(go.Candlestick(
            x=view_df.index, 
            open=view_df['Open'], high=view_df['High'],
            low=view_df['Low'], close=view_df['Close'], 
            name='K線價格'
        ))
        
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA5'], name='5日均線', line=dict(color='#FFA500', width=1.2)))
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA10'], name='10日均線', line=dict(color='#00FFFF', width=1.2)))
        
        # 買賣訊號 (hoverinfo='none' 隱藏懸浮價格，僅保留時間對齊)
        fig.add_trace(go.Scatter(
            x=view_df.index, y=view_df['買入訊號'],
            mode='markers', name='買入觸發',
            hoverinfo='none', # 核心修正：不顯示價格
            marker=dict(symbol='triangle-up', size=13, color='#FF3131')
        ))
        
        fig.add_trace(go.Scatter(
            x=view_df.index, y=view_df['賣出訊號'],
            mode='markers', name='賣出觸發',
            hoverinfo='none', # 核心修正：不顯示價格
            marker=dict(symbol='triangle-down', size=13, color='#39FF14')
        ))
        
        # 圖表設定
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            # 禁用所有縮放與選擇框
            yaxis=dict(autorange=True, fixedrange=True, title="真實成交價 (數字)", tickformat='.2f'),
            xaxis=dict(fixedrange=True, title="交易日期", tickformat="%Y-%m-%d"),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # 徹底移除工具列
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 今日資訊
        latest = view_df.iloc[-1]
        st.write(f"#### 📝 {symbol} 今日數據摘要")
        c1, c2, c3 = st.columns(3)
        c1.metric("當前收盤", f"{latest['Close']:.2f}")
        c2.metric("5日均價", f"{latest['MA5']:.2f}")
        c3.metric("RSI(5)強度", f"{latest['RSI5']:.1f}")
