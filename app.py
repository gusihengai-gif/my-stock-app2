import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="台股互動式量化觀測站", layout="wide")

# --- 計算指標函數 ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    # 移動平均線 (MA)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    df['RSI5'] = 100 - (100 / (1 + (gain / loss)))
    return df

# --- UI 介面 ---
st.title("📊 台股互動式 K 線觀測站")

# 側邊欄設定
st.sidebar.header("查詢設定")
symbol = st.sidebar.text_input("輸入台股代碼", value="2330")

# --- 新增功能：天數切換按鈕 ---
st.write("### 選擇觀測範圍")
period_col = st.columns(6)
days_map = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}
selected_days = 60 # 預設

# 用按鈕切換天數 (存放在 Session State 中)
if 'days' not in st.session_state:
    st.session_state.days = 60

# 建立按鈕橫列
for i, (label, val) in enumerate(days_map.items()):
    if period_col[i].button(label):
        st.session_state.days = val

# 抓取資料 (為了計算指標穩定，固定抓取 1 年，顯示時再切換範圍)
if symbol:
    with st.spinner("抓取最新數據中..."):
        ticker = f"{symbol}.TW"
        df = yf.download(ticker, period="2y", auto_adjust=True)
        if df.empty:
            ticker = f"{symbol}.TWO"
            df = yf.download(ticker, period="2y", auto_adjust=True)

        if not df.empty:
            # 處理多重索引與欄位名
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).capitalize() for c in df.columns]
            
            # 計算指標
            df = calculate_indicators(df)
            
            # 根據選擇的天數截取顯示範圍
            display_df = df.tail(st.session_state.days)

            # --- Plotly 互動圖表製作 ---
            fig = go.Figure()

            # 1. 繪製 K 線 (Candlestick)
            fig.add_trace(go.Candlestick(
                x=display_df.index,
                open=display_df['Open'],
                high=display_df['High'],
                low=display_df['Low'],
                close=display_df['Close'],
                name='K線',
                increasing_line_color='#FF3333', # 紅色上漲
                decreasing_line_color='#00AA00'  # 綠色下跌
            ))

            # 2. 疊加 MA5 均線
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['MA5'],
                mode='lines', name='MA5', line=dict(color='orange', width=1)
            ))

            # 3. 疊加 MA10 均線
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['MA10'],
                mode='lines', name='MA10', line=dict(color='blue', width=1)
            ))

            # --- 圖表佈局設定 (包含自動縮放) ---
            fig.update_layout(
                title=f"{ticker} - {st.session_state.days}天走勢 (可滑動縮放)",
                yaxis_title="價格 (TWD)",
                xaxis_title="日期",
                height=600,
                xaxis_rangeslider_visible=False, # 關閉下方滑桿以獲得更乾淨的視野
                hovermode='x unified', # 滑鼠指到哪裡就顯示該日所有數值
                yaxis=dict(autorange=True, fixedrange=False), # Y 軸自動縮放
                template="plotly_white"
            )

            # 顯示圖表
            st.plotly_chart(fig, use_container_width=True)

            # --- 顯示今日數值摘要 ---
            latest = display_df.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("收盤價", f"{latest['Close']:.2f}")
            c2.metric("今日漲跌", f"{latest['Close'] - display_df.iloc[-2]['Close']:.2f}")
            c3.metric("RSI5", f"{latest['RSI5']:.1f}")
        else:
            st.error("查無代碼，請重新輸入。")
