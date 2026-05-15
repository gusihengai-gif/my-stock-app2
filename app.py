import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. 網頁基礎配置 ---
st.set_page_config(page_title="台股實價對齊監控系統", layout="wide")

# --- 2. 技術指標計算 (確保使用真實市價) ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    # 移動平均線 MA
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

# --- 3. 訊號標記邏輯 (實價對齊核心修正) ---
def get_signal_markers(df):
    # 建立與數據等長的空陣列，確保索引(Index)完全對齊日期
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 買入條件：MA金叉 或 KD低檔金叉 或 RSI站上50
        if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) or \
           (row['K'] < 25 and prev_row['K'] <= prev_row['D'] and row['K'] > row['D']) or \
           (prev_row['RSI5'] <= 50 and row['RSI5'] > 50):
            # 修正：將 Y 軸座標精準鎖定在當日最低價 (Low) 的下方
            buy_markers[i] = row['Low'] * 0.985
            
        # 賣出條件：跌破MA5 或 KD高檔死叉 或 RSI跌破50
        elif (row['Close'] < row['MA5']) or \
             (row['K'] > 75 and prev_row['K'] >= prev_row['D'] and row['K'] < row['D']) or \
             (prev_row['RSI5'] >= 50 and row['RSI5'] < 50):
            # 修正：將 Y 軸座標精準鎖定在當日最高價 (High) 的上方
            sell_markers[i] = row['High'] * 1.015
            
    return buy_markers, sell_markers

# --- 4. UI 介面與數據處理 ---
st.title("🚀 台股實價精準對齊觀測站")

# 定義變數防止 NameError
symbol = st.sidebar.text_input("輸入台股代碼 (例: 2330)", value="2330")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

# 週期切換按鈕
st.write("### 觀測週期切換")
cols = st.columns(6)
periods = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}
for i, (lab, val) in enumerate(periods.items()):
    if cols[i].button(lab):
        st.session_state.view_days = val

if symbol:
    with st.spinner(f"正在搜尋 {symbol} 的真實市場數據..."):
        # 核心：auto_adjust=False 確保獲取真實成交價而非除權息調整價
        data = yf.download(f"{symbol}.TW", period="2y", auto_adjust=False)
        if data.empty:
            data = yf.download(f"{symbol}.TWO", period="2y", auto_adjust=False)
            
        if not data.empty:
            # 處理 yfinance 多重索引問題
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(c).strip().capitalize() for c in data.columns]
            
            # 執行計算
            df = calculate_indicators(data)
            df['Buy_Sig'], df['Sell_Sig'] = get_signal_markers(df)
            
            # 擷取顯示區段
            view_df = df.tail(st.session_state.view_days)
            
            # --- 5. Plotly 互動圖表 ---
            fig = go.Figure()
            
            # K線圖 (使用真實實價)
            fig.add_trace(go.Candlestick(
                x=view_df.index, 
                open=view_df['Open'], high=view_df['High'],
                low=view_df['Low'], close=view_df['Close'], 
                name='K線價格'
            ))
            
            # 均線
            fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA5'], name='MA5', line=dict(color='#FFA500', width=1.5)))
            fig.add_trace(go.Scatter(x=view_df.index, y=view_df['MA10'], name='MA10', line=dict(color='#00FFFF', width=1.5)))
            
            # 買入標記 (🔴 三角形貼在 Low 下方)
            fig.add_trace(go.Scatter(
                x=view_df.index, y=view_df['Buy_Sig'],
                mode='markers', name='買入觸發',
                marker=dict(symbol='triangle-up', size=14, color='#FF3131', line=dict(width=1, color='white'))
            ))
            
            # 賣出標記 (🟢 三角形貼在 High 上方)
            fig.add_trace(go.Scatter(
                x=view_df.index, y=view_df['Sell_Sig'],
                mode='markers', name='賣出觸發',
                marker=dict(symbol='triangle-down', size=14, color='#39FF14', line=dict(width=1, color='white'))
            ))
            
            fig.update_layout(
                height=700,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                yaxis=dict(autorange=True, fixedrange=False, tickformat='.2f', title="真實成交價 (TWD)"),
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示當日細節卡片
            curr = view_df.iloc[-1]
            st.write(f"#### 📝 {symbol} 今日市場摘要")
            c1, c2, c3 = st.columns(3)
            c1.metric("收盤價", f"{curr['Close']:.2f}")
            c2.metric("MA5 均價", f"{curr['MA5']:.2f}")
            c3.metric("RSI(5) 強度", f"{curr['RSI5']:.1f}")
        else:
            st.error(f"❌ 找不到股票代碼 '{symbol}'，請確認輸入是否正確。")
