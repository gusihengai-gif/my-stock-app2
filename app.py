import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. 網頁基礎配置 ---
st.set_page_config(page_title="2364 實價監控-簡潔版", layout="wide")

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

# --- 3. 訊號標記邏輯 (對齊收盤價) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 買入觸發 (與先前邏輯一致)
        if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) or \
           (row['K'] < 25 and prev_row['K'] <= prev_row['D'] and row['K'] > row['D']) or \
           (prev_row['RSI5'] <= 50 and row['RSI5'] > 50):
            buy_markers[i] = row['Close'] * 0.985 # 對齊收盤價下方
            
        # 賣出觸發
        elif (row['Close'] < row['MA5']) or \
             (row['K'] > 75 and prev_row['K'] >= prev_row['D'] and row['K'] < row['D']) or \
             (prev_row['RSI5'] >= 50 and row['RSI5'] < 50):
            sell_markers[i] = row['Close'] * 1.015 # 對齊收盤價上方
            
    return buy_markers, sell_markers

# --- 4. UI 介面 ---
st.title("📈 2364 倫飛-實價連線觀測站")

symbol = st.sidebar.text_input("輸入代碼", value="2364")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測期間 (點擊切換)")
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
        
        # 轉換時間格式為字串，確保 X 軸不跳過假日
        view_df['日期標籤'] = view_df.index.strftime('%Y-%m-%d')
        
        fig = go.Figure()
        
        # --- 核心更改：改用 Scatter 線圖代替 K 線 ---
        fig.add_trace(go.Scatter(
            x=view_df['日期標籤'], 
            y=view_df['Close'],
            mode='lines+markers',
            name='收盤價連線',
            line=dict(color='#FFFFFF', width=2),
            marker=dict(size=4, color='#FFFFFF'),
            hoverinfo='text',
            text=view_df['Close']
        ))
        
        # 均線
        fig.add_trace(go.Scatter(x=view_df['日期標籤'], y=view_df['MA5'], name='5日均線', line=dict(color='#FFA500', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=view_df['日期標籤'], y=view_df['MA10'], name='10日均線', line=dict(color='#00FFFF', width=1, dash='dot')))
        
        # 買賣訊號
        fig.add_trace(go.Scatter(
            x=view_df['日期標籤'], y=view_df['買入訊號'],
            mode='markers', name='買入觸發',
            hoverinfo='none',
            marker=dict(symbol='triangle-up', size=14, color='#FF3131')
        ))
        
        fig.add_trace(go.Scatter(
            x=view_df['日期標籤'], y=view_df['賣出訊號'],
            mode='markers', name='賣出觸發',
            hoverinfo='none',
            marker=dict(symbol='triangle-down', size=14, color='#39FF14')
        ))
        
        # 圖表佈局
        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            # type='category' 是扣除非交易日的關鍵
            xaxis=dict(type='category', title="交易日期 (不含假日)", fixedrange=True, tickangle=45),
            yaxis=dict(autorange=True, fixedrange=True, title="收盤價格 (數字)", tickformat='.2f'),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 底部摘要
        st.info(f"💡 目前為「收盤連線模式」。X 軸僅顯示實際交易日。")
