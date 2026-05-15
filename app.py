import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. 網頁基礎配置 ---
st.set_page_config(page_title="股票買賣時機", layout="wide")

# --- 2. 技術指標計算 (新增 BIAS 與 布林) ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    # 均線
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean() # 月線，布林中軌
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    df['RSI5'] = 100 - (100 / (1 + (gain.rolling(5).mean() / loss.rolling(5).mean())))
    df['RSI10'] = 100 - (100 / (1 + (gain.rolling(10).mean() / loss.rolling(10).mean())))
    
    # KDJ
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    # --- 新增：BIAS (5日乖離率) ---
    df['BIAS5'] = ((df['Close'] - df['MA5']) / df['MA5']) * 100

    # --- 新增：布林通道 (20日, 2標準差) ---
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std20 * 2)
    
    return df

# --- 3. 核心訊號邏輯 (包含獨立壓力判斷) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    in_position = False 
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 買入：維持嚴謹
        if not in_position:
            if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) and (row['RSI5'] > 50):
                buy_markers[i] = row['Close']
                in_position = True
                
        # 賣出：原本的三重共振 OR 新增的壓力指標
        elif in_position:
            # 原本的嚴格條件
            strict_sell = (row['RSI5'] < row['RSI10']) and (row['Close'] < row['MA5']) and (row['K'] < row['D'])
            
            # --- 新增：單獨列出的壓力指標 ---
            # 1. 5日正乖離過大 (例如 > 6%，代表短線噴發過頭)
            bias_over = (row['BIAS5'] > 6.0)
            
            # 2. 觸碰布林上軌 (股價碰觸或穿過 BB_Upper)
            bb_pressure = (row['Close'] >= row['BB_Upper'])
            
            if strict_sell or bias_over or bb_pressure:
                sell_markers[i] = row['Close']
                in_position = False 
                
    return buy_markers, sell_markers

# --- 4. 數據與 UI 邏輯 ---
st.sidebar.title("🚀 股票買賣時機")
symbol_input = st.sidebar.text_input("輸入股票代碼", value="2454")

# 這裡省略 fetch_stock_data 函數(與前次相同)...
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    target_sym = f"{symbol}.TW" if "." not in symbol else symbol
    data = yf.download(target_sym, period="2y", auto_adjust=False)
    if data.empty and ".TW" in target_sym:
        target_sym = target_sym.replace(".TW", ".TWO")
        data = yf.download(target_sym, period="2y", auto_adjust=False)
    return data, target_sym

if symbol_input:
    data, final_symbol = fetch_stock_data(symbol_input)
    if not data.empty:
        st.subheader(f"📈 目前查看：{final_symbol}")
        
        # 處理索引與指標
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'] = get_signal_markers(df)
        
        # 時間軸過濾邏輯...
        view_days = st.session_state.get('view_days', 60)
        view_df = df.tail(view_days).copy()
        view_df['日期顯示'] = view_df.index.strftime('%Y-%m-%d')

        # 繪製圖表
        fig = go.Figure()
        
        # 實價連線
        fig.add_trace(go.Scatter(x=view_df['日期顯示'], y=view_df['Close'], mode='lines', name='實價', line=dict(color='#FFFFFF', width=2)))
        
        # --- 視覺化新增：布林上軌 (顯示為暗紅色虛線，代表壓力) ---
        fig.add_trace(go.Scatter(x=view_df['日期顯示'], y=view_df['BB_Upper'], name='布林壓力線', line=dict(color='rgba(255, 0, 0, 0.3)', dash='dash')))

        # 買賣標記
        fig.add_trace(go.Scatter(x=view_df['日期顯示'], y=view_df['買入點'], mode='markers', name='重要買點', marker=dict(symbol='triangle-up', size=15, color='#FF3131')))
        fig.add_trace(go.Scatter(x=view_df['日期顯示'], y=view_df['賣出點'], mode='markers', name='重要賣點', marker=dict(symbol='triangle-down', size=15, color='#39FF14')))

        fig.update_layout(
            height=600,
            xaxis=dict(type='category', title=""),
            template="plotly_dark",
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 顯示指標狀態
        curr = view_df.iloc[-1]
        st.write("#### 🔍 壓力指標監控")
        c1, c2 = st.columns(2)
        c1.metric("5日乖離率 (BIAS5)", f"{curr['BIAS5']:.2f}%", help="超過 6% 代表短線過熱")
        c2.metric("布林上軌 (壓力位)", f"{curr['BB_Upper']:.2f}")

        st.warning("📊 當前賣出依據：1.三重共振成立 或 2.BIAS > 6% 或 3.觸碰布林上軌。")
