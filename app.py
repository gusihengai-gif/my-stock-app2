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
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    df['RSI5'] = 100 - (100 / (1 + (gain.rolling(5).mean() / loss.rolling(5).mean())))
    df['RSI10'] = 100 - (100 / (1 + (gain.rolling(10).mean() / loss.rolling(10).mean())))
    
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
        
        if not in_position:
            if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) and (row['RSI5'] > 50):
                buy_markers[i] = row['Close']
                in_position = True
        elif in_position:
            if (row['RSI5'] < row['RSI10']) and (row['Close'] < row['MA5']) and (row['K'] < row['D']):
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
periods = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}
for i, (lab, val) in enumerate(periods.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

# --- 5. 資料抓取 ---
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
        st.subheader(f"📈 {final_symbol}")
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期顯示'] = view_df.index.strftime('%Y-%m-%d')
        view_df['月份'] = view_df.index.strftime('%Y-%m')

        # --- 6. X軸標籤優化邏輯 ---
        tick_vals = []
        tick_texts = []
        days = st.session_state.view_days

        if days == 10:
            tick_vals = view_df['日期顯示'].tolist()
            tick_texts = view_df['日期顯示'].tolist()
        elif days == 20:
            tick_vals = view_df['日期顯示'].tolist()[::5] # 每5天
            tick_texts = view_df['日期顯示'].tolist()[::5]
        elif days == 30:
            tick_vals = view_df['日期顯示'].tolist()[::10] # 每10天
            tick_texts = view_df['日期顯示'].tolist()[::10]
        else: # 60天以上：只顯示每月第一個交易日
            # 找出每個月分組後的第一個索引
            first_days = view_df.groupby('月份').head(1)
            tick_vals = first_days['日期顯示'].tolist()
            tick_texts = first_days['日期顯示'].tolist()

        fig = go.Figure()
        
        # 實價連線
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['Close'],
            mode='lines', name='實價',
            line=dict(color='#FFFFFF', width=2),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        # 買賣點
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['買入點'],
            mode='markers', name='買入',
            marker=dict(symbol='triangle-up', size=14, color='#FF3131'),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['賣出點'],
            mode='markers', name='賣出',
            marker=dict(symbol='triangle-down', size=14, color='#39FF14'),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        # --- 7. 更新佈局 (關鍵修正) ---
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='closest',
            xaxis=dict(
                type='category',
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_texts,
                tickangle=45,
                fixedrange=True,
                title=""
            ),
            yaxis=dict(autorange=True, fixedrange=True, title="實價", tickformat='.2f'),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.warning("⚠️ 賣出條件：RSI死叉 + KDJ死叉 + 跌破5日線 (三重共振成立)")
