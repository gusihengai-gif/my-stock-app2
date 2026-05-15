import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. 網頁基礎配置 ---
st.set_page_config(page_title="2364 趨勢決策版", layout="wide")

# --- 2. 技術指標計算 ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    # RSI (14天) - 使用較長天數來降低靈敏度
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI14'] = 100 - (100 / (1 + (gain / loss)))
    
    # KDJ (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    return df

# --- 3. 訊號標記邏輯 (降低靈敏度，強化每筆價值) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # --- 買入條件：必須同時滿足 (共振買入) ---
        # 1. MA5 金叉 MA10
        # 2. RSI14 站上 50 (動能確認)
        ma_gold_cross = (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10'])
        rsi_strong = (row['RSI14'] > 50)
        
        if ma_gold_cross and rsi_strong:
            buy_markers[i] = row['Close'] * 0.98
            
        # --- 賣出條件：重勢不重量 (保護利潤) ---
        # 1. 收盤價「跌破」10日線 (長期趨勢走壞)
        # 2. 或 KD 處於極高檔死叉 (80以上)
        trend_broken = (row['Close'] < row['MA10'])
        kd_dead_cross = (prev_row['K'] >= prev_row['D'] and row['K'] < row['D'] and row['K'] > 80)
        
        if trend_broken or kd_dead_cross:
            sell_markers[i] = row['Close'] * 1.02
            
    return buy_markers, sell_markers

# --- 4. UI 介面 ---
st.title("📈 2364 倫飛 - 趨勢決策觀測站")
symbol = st.sidebar.text_input("輸入代碼", value="2364")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測期間")
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
        
        df = calculate_indicators(data)
        df['買入訊號'], df['賣出訊號'] = get_signal_markers(df)
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期標籤'] = view_df.index.strftime('%m/%d') # 簡化日期顯示
        
        fig = go.Figure()
        
        # 收盤價連線
        fig.add_trace(go.Scatter(
            x=view_df['日期標籤'], y=view_df['Close'],
            mode='lines', name='收盤價',
            line=dict(color='#FFFFFF', width=2),
            hoverinfo='y'
        ))
        
        # 關鍵趨勢線 (MA10)
        fig.add_trace(go.Scatter(x=view_df['日期標籤'], y=view_df['MA10'], name='10日趨勢線', line=dict(color='#00FFFF', width=1)))
        
        # 買賣訊號 (僅顯示符號，不顯示價格)
        fig.add_trace(go.Scatter(
            x=view_df['日期標籤'], y=view_df['買入訊號'],
            mode='markers', name='【重要】建議買進',
            hoverinfo='none',
            marker=dict(symbol='triangle-up', size=16, color='#FF3131', line=dict(width=1, color='white'))
        ))
        
        fig.add_trace(go.Scatter(
            x=view_df['日期標籤'], y=view_df['賣出訊號'],
            mode='markers', name='【重要】建議賣出',
            hoverinfo='none',
            marker=dict(symbol='triangle-down', size=16, color='#39FF14', line=dict(width=1, color='white'))
        ))
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            xaxis=dict(type='category', title="交易日", fixedrange=True),
            yaxis=dict(autorange=True, fixedrange=True, title="價格", tickformat='.2f'),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.success("✅ 已過濾短線雜訊，目前僅顯示「趨勢轉折」級別的重要訊號。")
