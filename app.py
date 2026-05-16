import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import urllib.request
import json

# --- 1. 網頁基礎配置 ---
st.set_page_config(page_title="股票買賣時機", layout="wide")

# --- 2. 繁體中文名稱全自動查詢（對照表 + 證交所官方 API 雙防線） ---
STOCK_NAME_DICT = {
    "2364": "倫飛", "2454": "聯發科", "2330": "台積電", "2317": "鴻海",
    "3481": "群創", "2409": "面板", "00919": "群益台灣精選高息",
    "0056": "元大高股息", "00878": "國泰永續高股息", "00929": "復華台灣科技優息",
    "0050": "元大台灣50"
}

@st.cache_data(ttl=86400) # 快取一天，避免重複聯網查詢，執行速度極快
def get_stock_display_name(symbol):
    # 提取純數字代碼，例如 "3481.TW" -> "3481"
    pure_code = symbol.split('.')[0].strip()
    
    # 第一道防線：如果核心表裡有，直接秒回傳
    if pure_code in STOCK_NAME_DICT:
        return f"{STOCK_NAME_DICT[pure_code]} ({symbol})"
    
    # 第二道防線：全自動聯網向台灣證券交易所 API 查詢中文簡稱
    try:
        # 1. 查詢上市公司名單
        url_l = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        req_l = urllib.request.Request(url_l, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_l) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            for item in res_data:
                if item.get('公司代號') == pure_code:
                    c_name = item.get('公司簡稱', '').strip()
                    return f"{c_name} ({symbol})"
                    
        # 2. 若上市找不到，查詢上櫃公司名單
        url_o = "https://openapi.twse.com.tw/v1/opendata/t187ap03_O"
        req_o = urllib.request.Request(url_o, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_o) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            for item in res_data:
                if item.get('公司代號') == pure_code:
                    c_name = item.get('公司簡稱', '').strip()
                    return f"{c_name} ({symbol})"
    except Exception:
        pass
        
    # 第三道防線：萬一網路異常，回傳原始代碼，確保程式絕不崩潰
    return symbol

# --- 3. 技術指標計算 ---
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

# --- 4. 核心訊號邏輯 (賣出需「同時成立」) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    sell_reasons = [""] * len(df)
    in_position = False 
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if not in_position:
            if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) and (row['RSI5'] > 50):
                buy_markers[i] = row['Close']
                in_position = True
        elif in_position:
            cond_rsi = (row['RSI5'] < row['RSI10'])
            cond_price = (row['Close'] < row['MA5'])
            cond_kdj = (row['K'] < row['D'])
            
            if cond_rsi and cond_price and cond_kdj:
                sell_markers[i] = row['Close']
                sell_reasons[i] = "RSI死叉 + KDJ死叉 + 跌破MA5"
                in_position = False 
    return buy_markers, sell_markers, sell_reasons

# --- 5. UI 介面 ---
st.sidebar.title("🚀 股票買賣時機")
symbol_input = st.sidebar.text_input("輸入股票代碼", value="3481")

if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期選擇")
p_cols = st.columns(6)
periods = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}
for i, (lab, val) in enumerate(periods.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

# --- 6. 資料抓取 ---
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
        # 【功能升級】自動取得完全繁體中文的股票/ETF名稱
        display_title = get_stock_display_name(final_symbol)
        st.subheader(f"📈 {display_title}")
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'], df['賣出原因'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期顯示'] = view_df.index.strftime('%Y-%m-%d')
        view_df['月份'] = view_df.index.strftime('%Y-%m')

        # --- X軸標籤優化邏輯 ---
        tick_vals = []
        tick_texts = []
        days = st.session_state.view_days

        if days == 10:
            tick_vals = view_df['日期顯示'].tolist()
            tick_texts = view_df['日期顯示'].tolist()
        elif days == 20:
            tick_vals = view_df['日期顯示'].tolist()[::5]
            tick_texts = view_df['日期顯示'].tolist()[::5]
        elif days == 30:
            tick_vals = view_df['日期顯示'].tolist()[::10]
            tick_texts = view_df['日期顯示'].tolist()[::10]
        else: 
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
        
        # 買入點
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['買入點'],
            mode='markers', name='買入',
            marker=dict(symbol='triangle-up', size=14, color='#FF3131'),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        # 賣出點（🟢 綠色圓點 + 懸浮動態條件）
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['賣出點'],
            mode='markers', name='賣出',
            marker=dict(symbol='circle', size=14, color='#39FF14'),
            customdata=view_df['賣出原因'],
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<br><b>滿足條件: %{customdata}</b><extra></extra>"
        ))
        
        # --- 7. 更新佈局 ---
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
