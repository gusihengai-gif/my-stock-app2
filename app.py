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
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
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

    # BIAS 5日乖離
    df['BIAS5'] = ((df['Close'] - df['MA5']) / df['MA5']) * 100

    # 布林通道
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std20 * 2)
    
    return df

# --- 3. 雙軌訊號邏輯 ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_strict = np.full(len(df), np.nan) # 綠色三角 (趨勢反轉)
    sell_pressure = np.full(len(df), np.nan) # 黃色標記 (過熱預警)
    
    in_position = False 
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 買入
        if not in_position:
            if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) and (row['RSI5'] > 50):
                buy_markers[i] = row['Close']
                in_position = True
                
        # 賣出判定
        elif in_position:
            # 1. 嚴格賣出 (三重共振)
            is_strict = (row['RSI5'] < row['RSI10']) and (row['Close'] < row['MA5']) and (row['K'] < row['D'])
            
            # 2. 壓力賣出 (BIAS > 6% 或 觸碰布林上軌)
            is_pressure = (row['BIAS5'] > 6.0) or (row['Close'] >= row['BB_Upper'])
            
            if is_strict:
                sell_strict[i] = row['Close']
                in_position = False 
            elif is_pressure:
                sell_pressure[i] = row['Close']
                in_position = False 
                
    return buy_markers, sell_strict, sell_pressure

# --- 4. 資料抓取 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    target_sym = f"{symbol}.TW" if "." not in symbol else symbol
    data = yf.download(target_sym, period="2y", auto_adjust=False)
    if data.empty and ".TW" in target_sym:
        target_sym = target_sym.replace(".TW", ".TWO")
        data = yf.download(target_sym, period="2y", auto_adjust=False)
    return data, target_sym

# --- 5. UI 與 繪圖 ---
st.sidebar.title("🚀 股票買賣時機")
symbol_input = st.sidebar.text_input("輸入股票代碼", value="2454")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期選擇")
p_cols = st.columns(6)
for i, (lab, val) in enumerate({"10天":10, "20天":20, "30天":30, "60天":60, "120天":120, "240天":240}.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

if symbol_input:
    data, final_symbol = fetch_stock_data(symbol_input)
    if not data.empty:
        st.subheader(f"📈 {final_symbol}")
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        
        df = calculate_indicators(data)
        df['買入'], df['嚴格賣出'], df['壓力賣出'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期'] = view_df.index.strftime('%Y-%m-%d')

        fig = go.Figure()
        
        # 主線
        fig.add_trace(go.Scatter(x=view_df['日期'], y=view_df['Close'], mode='lines', name='實價', line=dict(color='#FFFFFF', width=2), hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"))
        
        # 布林壓力線 (淡淡的紅色虛線)
        fig.add_trace(go.Scatter(x=view_df['日期'], y=view_df['BB_Upper'], name='布林壓力', line=dict(color='rgba(255, 80, 80, 0.2)', dash='dash'), hoverinfo='none'))

        # 買入標記
        fig.add_trace(go.Scatter(x=view_df['日期'], y=view_df['買入'], mode='markers', name='重要買入', marker=dict(symbol='triangle-up', size=15, color='#FF3131')))
        
        # 標記 A：嚴格賣出 (綠色三角)
        fig.add_trace(go.Scatter(x=view_df['日期'], y=view_df['嚴格賣出'], mode='markers', name='趨勢轉弱賣出', marker=dict(symbol='triangle-down', size=15, color='#39FF14')))
        
        # 標記 B：壓力賣出 (黃色圓圈，內部帶驚嘆號感)
        fig.add_trace(go.Scatter(x=view_df['日期'], y=view_df['壓力賣出'], mode='markers', name='高檔壓力賣出', marker=dict(symbol='hexagram', size=16, color='#FFFF00', line=dict(width=1, color='white'))))

        fig.update_layout(
            height=600,
            xaxis=dict(type='category', title="", tickangle=45),
            template="plotly_dark",
            hovermode='closest',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.info("💡 標記說明：\n1. **綠色三角**：趨勢全面轉弱 (RSI+KDJ+MA5共振)。\n2. **黃色六角**：高檔壓力警戒 (BIAS過大或觸碰布林上軌)。")
