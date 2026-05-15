import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. 網頁基礎配置 ---
st.set_page_config(page_title="股票買賣時機", layout="wide")

# --- 2. 技術指標計算 ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    # RSI 計算
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    df['RSI5'] = 100 - (100 / (1 + (gain.rolling(5).mean() / loss.rolling(5).mean())))
    df['RSI10'] = 100 - (100 / (1 + (gain.rolling(10).mean() / loss.rolling(10).mean())))
    
    # KDJ 計算
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    
    return df

# --- 3. 核心訊號邏輯 (嚴格共振：賣出需同時成立) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    in_position = False 
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 買入：MA金叉 + RSI強勢
        if not in_position:
            if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) and (row['RSI5'] > 50):
                buy_markers[i] = row['Close']
                in_position = True
                
        # 賣出：RSI死叉 + KDJ死叉 + 跌破5日線 (同時成立)
        elif in_position:
            rsi_weak = (row['RSI5'] < row['RSI10'])
            price_weak = (row['Close'] < row['MA5'])
            kdj_weak = (row['K'] < row['D'])
            
            if rsi_weak and price_weak and kdj_weak:
                sell_markers[i] = row['Close']
                in_position = False 
                
    return buy_markers, sell_markers

# --- 4. UI 介面 ---
st.sidebar.title("🚀 股票買賣時機")
symbol_input = st.sidebar.text_input("輸入股票代碼", value="2364")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期選擇")
p_cols = st.columns(6)
for i, (lab, val) in enumerate({"10天":10, "20天":20, "30天":30, "60天":60, "120天":120, "240天":240}.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

if symbol_input:
    # 判斷輸入格式並自動補強
    full_symbol = f"{symbol_input}.TW" if "." not in symbol_input else symbol_input
    ticker = yf.Ticker(full_symbol)
    data = ticker.history(period="2y")
    
    # 如果 .TW 沒資料，嘗試 .TWO (上櫃)
    if data.empty and ".TW" in full_symbol:
        full_symbol = full_symbol.replace(".TW", ".TWO")
        ticker = yf.Ticker(full_symbol)
        data = ticker.history(period="2y")

    if not data.empty:
        # 顯示股票名稱與代號
        stock_name = ticker.info.get('shortName', '')
        st.subheader(f"📈 {stock_name} ({full_symbol})")
        
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期顯示'] = view_df.index.strftime('%Y-%m-%d')
        
        fig = go.Figure()
        
        # 收盤價連線 (自定義 hovertemplate 顯示時間與價格)
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['Close'],
            mode='lines', name='收盤價',
            line=dict(color='#FFFFFF', width=2),
            hovertemplate="時間: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        # 買賣點標記
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['買入點'],
            mode='markers', name='重要買入',
            marker=dict(symbol='triangle-up', size=15, color='#FF3131'),
            hovertemplate="買入時間: %{x}<br>觸發價格: %{y:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['賣出點'],
            mode='markers', name='重要賣出',
            marker=dict(symbol='triangle-down', size=15, color='#39FF14'),
            hovertemplate="賣出時間: %{x}<br>觸發價格: %{y:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='closest',
            xaxis=dict(type='category', title="交易日期", fixedrange=True),
            yaxis=dict(autorange=True, fixedrange=True, title="實價", tickformat='.2f'),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.warning("⚠️ 賣出門檻：RSI死叉、KDJ死叉、跌破5日線「同時成立」才會顯示。")
    else:
        st.error(f"無法找到股票代碼: {symbol_input}")
