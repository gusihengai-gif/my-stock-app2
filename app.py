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
    # 移動平均線
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

# --- 3. 核心訊號邏輯 (賣出需「同時成立」) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    in_position = False 
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 買入：MA5金叉MA10 且 RSI5 > 50
        if not in_position:
            if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) and (row['RSI5'] > 50):
                buy_markers[i] = row['Close']
                in_position = True
                
        # 賣出：RSI死叉 + KDJ死叉 + 跌破5日線 (三重共振)
        elif in_position:
            cond_rsi = (row['RSI5'] < row['RSI10'])
            cond_price = (row['Close'] < row['MA5'])
            cond_kdj = (row['K'] < row['D'])
            
            if cond_rsi and cond_price and cond_kdj:
                sell_markers[i] = row['Close']
                in_position = False 
                
    return buy_markers, sell_markers

# --- 4. UI 介面 ---
st.sidebar.title("🚀 股票買賣時機")
symbol_input = st.sidebar.text_input("輸入股票代碼 (例: 2364)", value="2364")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期選擇")
p_cols = st.columns(6)
periods = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}
for i, (lab, val) in enumerate(periods.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

# --- 5. 資料抓取函數 (修正了 NameError 變數問題) ---
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    # 確保代碼格式正確
    target_sym = f"{symbol}.TW" if "." not in symbol else symbol
    data = yf.download(target_sym, period="2y", auto_adjust=False)
    
    # 支援上櫃股票 (.TWO)
    if data.empty and ".TW" in target_sym:
        target_sym = target_sym.replace(".TW", ".TWO")
        data = yf.download(target_sym, period="2y", auto_adjust=False)
    
    return data, target_sym

if symbol_input:
    with st.spinner('連線至 Yahoo Finance...'):
        data, final_symbol = fetch_stock_data(symbol_input)

    if not data.empty:
        # 動態顯示代號標題
        st.subheader(f"📈 目前查看：{final_symbol}")
        
        # 處理多重索引
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期標籤'] = view_df.index.strftime('%Y-%m-%d')
        
        # --- 6. 繪圖 (顯示時間與價格) ---
        fig = go.Figure()
        
        # 收盤價連線
        fig.add_trace(go.Scatter(
            x=view_df['日期標籤'], y=view_df['Close'],
            mode='lines', name='實價連線',
            line=dict(color='#FFFFFF', width=2),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        # 買入標記 (🔴)
        fig.add_trace(go.Scatter(
            x=view_df['日期標籤'], y=view_df['買入點'],
            mode='markers', name='重要買點',
            marker=dict(symbol='triangle-up', size=15, color='#FF3131'),
            hovertemplate="買入日期: %{x}<br>成交價格: %{y:.2f}<extra></extra>"
        ))
        
        # 賣出標記 (🟢)
        fig.add_trace(go.Scatter(
            x=view_df['日期標籤'], y=view_df['賣出點'],
            mode='markers', name='重要賣點',
            marker=dict(symbol='triangle-down', size=15, color='#39FF14'),
            hovertemplate="賣出日期: %{x}<br>成交價格: %{y:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='closest',
            xaxis=dict(type='category', title="交易日期", fixedrange=True, tickangle=45),
            yaxis=dict(autorange=True, fixedrange=True, title="價格 (TWD)", tickformat='.2f'),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.info("💡 提示：賣出訊號需同時符合 RSI死叉、KDJ死叉、跌破5日線才會觸發。")
    else:
        st.error(f"找不到代碼 {symbol_input}，請檢查輸入是否正確。")
