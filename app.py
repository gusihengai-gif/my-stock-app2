import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. 網頁基礎配置 ---
st.set_page_config(page_title="2364 倫飛決策系統-終極版", layout="wide")

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
    
    # KDJ 計算 (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    
    return df

# --- 3. 核心訊號邏輯 (買賣一對一，每筆交易最重要) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    
    in_position = False # 倉位狀態
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # --- 買入觸發：趨勢共振 ---
        if not in_position:
            # 條件：MA5 金叉 MA10 且 RSI5 > 50 且 K > D (全面轉強)
            if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) and \
               (row['RSI5'] > 50 and row['K'] > row['D']):
                buy_markers[i] = row['Close'] * 0.985
                in_position = True
                
        # --- 賣出觸發：三重預警 (新增 KDJ 死叉) ---
        elif in_position:
            # 1. RSI 向下交叉
            rsi_death = (prev_row['RSI5'] >= prev_row['RSI10'] and row['RSI5'] < row['RSI10'])
            # 2. 跌破 5 日線
            price_break = (row['Close'] < row['MA5'])
            # 3. KDJ 死亡交叉 (K 跌破 D)
            kdj_death = (prev_row['K'] >= prev_row['D'] and row['K'] < row['D'])
            
            if rsi_death or price_break or kdj_death:
                sell_markers[i] = row['Close'] * 1.015
                in_position = False # 結清這筆重要交易
                
    return buy_markers, sell_markers

# --- 4. UI 介面與資料呈現 ---
st.title("📊 2364 倫飛 - 終極實價決策系統")
symbol = st.sidebar.text_input("輸入股票代碼", value="2364")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期")
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
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期'] = view_df.index.strftime('%Y-%m-%d')
        
        fig = go.Figure()
        
        # 收盤價主線
        fig.add_trace(go.Scatter(
            x=view_df['日期'], y=view_df['Close'],
            mode='lines', name='收盤價',
            line=dict(color='#FFFFFF', width=2.5),
            hoverinfo='y'
        ))
        
        # 輔助線 (MA5)
        fig.add_trace(go.Scatter(
            x=view_df['日期'], y=view_df['MA5'], name='5日線', 
            line=dict(color='#FFA500', width=1, dash='dot'), hoverinfo='none'
        ))
        
        # 重要買賣點 (加大符號，強化視覺)
        fig.add_trace(go.Scatter(
            x=view_df['日期'], y=df.loc[view_df.index, '買入點'],
            mode='markers', name='【重要買入】',
            hoverinfo='none',
            marker=dict(symbol='triangle-up', size=16, color='#FF3131', line=dict(width=1, color='white'))
        ))
        
        fig.add_trace(go.Scatter(
            x=view_df['日期'], y=df.loc[view_df.index, '賣出點'],
            mode='markers', name='【重要賣出】',
            hoverinfo='none',
            marker=dict(symbol='triangle-down', size=16, color='#39FF14', line=dict(width=1, color='white'))
        ))
        
        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            xaxis=dict(type='category', title="交易日 (扣除假日)", fixedrange=True, tickangle=45),
            yaxis=dict(autorange=True, fixedrange=True, title="實價 (數字)", tickformat='.2f'),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 狀態摘要
        curr = view_df.iloc[-1]
        st.write(f"#### 📝 今日監測指標 ({symbol})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("收盤實價", f"{curr['Close']:.2f}")
        c2.metric("RSI5 強度", f"{curr['RSI5']:.1f}")
        c3.metric("K值 (轉折)", f"{curr['K']:.1f}")
        c4.metric("D值 (平滑)", f"{curr['D']:.1f}")
        
        st.success("✅ 賣出依據：RSI 死叉、KDJ 死叉、或跌破 5 日均線。三道防線已就緒。")
