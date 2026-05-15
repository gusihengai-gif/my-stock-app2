import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. 網頁基礎配置 ---
st.set_page_config(page_title="2364 倫飛決策系統", layout="wide")

# --- 2. 技術指標計算 ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    # 移動平均線
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    # RSI 計算邏輯 (5日與10日用於判斷交叉)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    # 避免除以零，使用滾動平均
    df['RSI5'] = 100 - (100 / (1 + (gain.rolling(5).mean() / loss.rolling(5).mean())))
    df['RSI10'] = 100 - (100 / (1 + (gain.rolling(10).mean() / loss.rolling(10).mean())))
    
    return df

# --- 3. 核心訊號邏輯 (每筆交易最重要：買賣一對一) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    
    in_position = False # 倉位追蹤
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # --- 買入觸發：趨勢共振 ---
        # 條件：MA5 金叉 MA10 且 短線動能 RSI5 站上 50
        if not in_position:
            ma_gold_cross = (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10'])
            if ma_gold_cross and row['RSI5'] > 50:
                buy_markers[i] = row['Close'] * 0.985 # 標註在收盤價下方
                in_position = True
                
        # --- 賣出觸發：預警撤退 (你的核心要求) ---
        # 條件 1：RSI 向下交叉 (5日跌破10日)
        # 條件 2：收盤價跌破 5 日線
        elif in_position:
            rsi_death_cross = (prev_row['RSI5'] >= prev_row['RSI10'] and row['RSI5'] < row['RSI10'])
            price_break_ma5 = (row['Close'] < row['MA5'])
            
            if rsi_death_cross or price_break_ma5:
                sell_markers[i] = row['Close'] * 1.015 # 標註在收盤價上方
                in_position = False # 結清此筆交易
                
    return buy_markers, sell_markers

# --- 4. UI 介面與數據抓取 ---
st.title("📊 2364 倫飛 - 實價趨勢決策觀測")

symbol = st.sidebar.text_input("輸入股票代碼", value="2364")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期選擇")
p_cols = st.columns(6)
periods = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}
for i, (lab, val) in enumerate(periods.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

if symbol:
    # 抓取實價 (auto_adjust=False)
    data = yf.download(f"{symbol}.TW", period="2y", auto_adjust=False)
    if data.empty:
        data = yf.download(f"{symbol}.TWO", period="2y", auto_adjust=False)
        
    if not data.empty:
        # 處理多重索引
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        
        # 計算資料
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'] = get_signal_markers(df)
        
        # 截取顯示段落並處理 X 軸 (扣除假日)
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['交易日期'] = view_df.index.strftime('%Y-%m-%d')
        
        # --- 5. 繪製圖表 ---
        fig = go.Figure()
        
        # 收盤價主連線
        fig.add_trace(go.Scatter(
            x=view_df['交易日期'], y=view_df['Close'],
            mode='lines', name='收盤價',
            line=dict(color='#FFFFFF', width=2.5),
            hoverinfo='y'
        ))
        
        # 輔助均線 (MA5 虛線)
        fig.add_trace(go.Scatter(
            x=view_df['交易日期'], y=view_df['MA5'], 
            name='5日支撐線', 
            line=dict(color='#FFA500', width=1, dash='dot'),
            hoverinfo='none'
        ))
        
        # 買入標記 (🔴)
        fig.add_trace(go.Scatter(
            x=view_df['交易日期'], y=df.loc[view_df.index, '買入點'],
            mode='markers', name='【重要】買入點',
            hoverinfo='none',
            marker=dict(symbol='triangle-up', size=16, color='#FF3131', line=dict(width=1, color='white'))
        ))
        
        # 賣出標記 (🟢)
        fig.add_trace(go.Scatter(
            x=view_df['交易日期'], y=df.loc[view_df.index, '賣出點'],
            mode='markers', name='【重要】賣出點',
            hoverinfo='none',
            marker=dict(symbol='triangle-down', size=16, color='#39FF14', line=dict(width=1, color='white'))
        ))
        
        # 圖表佈局設定
        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            # 關鍵：使用 category 排除非交易日
            xaxis=dict(type='category', title="交易日 (不含假日)", fixedrange=True, tickangle=45),
            yaxis=dict(autorange=True, fixedrange=True, title="真實成交價 (數字)", tickformat='.2f'),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # 顯示圖表並移除工具列
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 今日結算
        latest = view_df.iloc[-1]
        st.write(f"#### 📝 {symbol} 即時監測摘要")
        c1, c2, c3 = st.columns(3)
        c1.metric("當前實價", f"{latest['Close']:.2f}")
        c2.metric("MA5 價格", f"{latest['MA5']:.2f}")
        c3.metric("RSI5 強度", f"{latest['RSI5']:.1f}")
        
        st.info("💡 賣出準則：當 RSI5 向下穿過 RSI10，或收盤價跌破橘色虛線(MA5)時觸發重要交易賣出。")
    else:
        st.error("找不到該股票代碼。")
