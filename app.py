import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

# --- 1. 全域網頁設定 ---
st.set_page_config(page_title="台股買賣時機觀測", layout="wide")

# 隱藏 Streamlit 預設的 Menu 與 Footer
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# --- 2. 動態抓取最新台股上市櫃與主要 ETF 資料庫 (拋棄 lxml 改用 BeautifulSoup) ---
@st.cache_data(ttl=86400)  # 股票清單一天更新一次即可
def fetch_taiwan_stocks():
    urls = {
        "上市": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "上櫃": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    }
    
    stock_list = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for market_type, url in urls.items():
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'ms950'
            
            # 使用內建的 html.parser 替代 lxml，徹底解決套件遺失問題
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': ' there is no class, look for rows '}) 
            if not table:
                tables = soup.find_all('table')
                for t in tables:
                    if '有價證券代號及名稱' in t.text:
                        table = t
                        break
            
            if table:
                rows = table.find_all('tr')
                is_valid_section = False
                for row in rows[1:]:  # 跳過標題列
                    cols = row.find_all('td')
                    if not cols:
                        continue
                    text = cols[0].text.strip()
                    
                    if "股 票" in text or "股票" in text or "受益憑證" in text:
                        is_valid_section = True
                        continue
                    elif "認購" in text or "認售" in text or "權證" in text or "債券" in text or "創櫃" in text:
                        is_valid_section = False
                        continue
                    
                    if is_valid_section and "　" in text:
                        parts = text.split("　")
                        stock_id = parts[0].strip()
                        stock_name = parts[1].strip()
                        
                        is_normal_stock = (len(stock_id) == 4 and stock_id.isdigit())
                        is_etf = (len(stock_id) in [5, 6] and stock_id.startswith("00"))
                        
                        if is_normal_stock or is_etf:
                            if "特" not in stock_name and "債" not in stock_name:
                                stock_list.append({"id": stock_id, "name": stock_name})
        except Exception as e:
            st.error(f"無法從 {market_type} 網址獲取資料: {e}")
            
    unique_stocks = {item['id']: item['name'] for item in stock_list}
    sorted_list = [{"id": k, "name": v} for k, v in sorted(unique_stocks.items())]
    return sorted_list

# 載入資料
with st.spinner("正在即時同步最新台股與 ETF 清單..."):
    ALL_STOCKS_DATA = fetch_taiwan_stocks()

# 防呆機制：若證交所網頁抓取失敗，提供基本熱門股避免選單掛掉
if not ALL_STOCKS_DATA:
    ALL_STOCKS_DATA = [{"id": "2330", "name": "台積電"}, {"id": "2317", "name": "鴻海"}, {"id": "2454", "name": "聯發科"}]

ALL_STOCKS = {item['id']: item['name'] for item in ALL_STOCKS_DATA}
ALL_STOCKS_LIST = [f"{item['id']} {item['name']}" for item in ALL_STOCKS_DATA]

def get_stock_display_name(symbol):
    pure_code = symbol.split('.')[0].strip()
    if pure_code in ALL_STOCKS:
        return f"{ALL_STOCKS[pure_code]} ({symbol})"
    return symbol

# --- 3. 技術指標計算 ---
def calculate_indicators(df):
    df = df.copy()
    
    # 計算均線
    df['MA5'] = df['Close'].rolling(window=5).mean().bfill().ffill()
    df['MA10'] = df['Close'].rolling(window=10).mean().bfill().ffill()
    
    # 計算 RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    roll_gain = gain.rolling(5).mean()
    roll_loss = loss.rolling(5).mean()
    df['RSI5'] = (100 - (100 / (1 + (roll_gain / roll_loss.replace(0, np.nan))))).bfill().ffill()
    
    roll_gain10 = gain.rolling(10).mean()
    roll_loss10 = loss.rolling(10).mean()
    df['RSI10'] = (100 - (100 / (1 + (roll_gain10 / roll_loss10.replace(0, np.nan))))).bfill().ffill()
    
    # 計算 KDJ
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean().bfill().ffill()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean().bfill().ffill()
    
    return df

# --- 4. 核心買賣訊號邏輯（三重共振） ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    sell_reasons = [""] * len(df)
    in_position = False 
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if pd.isna(row['MA5']) or pd.isna(row['MA10']):
            continue
            
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

# --- 5. 主畫面：股票搜尋與選擇欄 ---
st.title("📊 台灣股市即時決策儀表板")

if "final_target_code" not in st.session_state:
    st.session_state.final_target_code = "2330"

default_index = 0
for idx, item in enumerate(ALL_STOCKS_LIST):
    if item.startswith("2330"):
        default_index = idx
        break

selected_stock_str = st.selectbox(
    "請選擇或輸入股票代碼/名稱：",
    options=ALL_STOCKS_LIST,
    index=default_index
)

if selected_stock_str:
    st.session_state.final_target_code = selected_stock_str.split(" ")[0].strip()

final_target_code = st.session_state.final_target_code

# --- 6. 觀測週期選擇區塊 ---
if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期選擇")

row1_cols = st.columns(3)
if row1_cols[0].button("10天", use_container_width=True):
    st.session_state.view_days = 10
if row1_cols[1].button("20天", use_container_width=True):
    st.session_state.view_days = 20
if row1_cols[2].button("30天", use_container_width=True):
    st.session_state.view_days = 30

row2_cols = st.columns(3)
if row2_cols[0].button("60天", use_container_width=True):
    st.session_state.view_days = 60
if row2_cols[1].button("120天", use_container_width=True):
    st.session_state.view_days = 120
if row2_cols[2].button("240天", use_container_width=True):
    st.session_state.view_days = 240

# --- 7. 資料抓取與圖表渲染 ---
@st.cache_data(ttl=10)
def fetch_stock_data(symbol):
    target_sym = f"{symbol}.TW" if "." not in symbol else symbol
    raw_data = yf.download(target_sym, period="2y", auto_adjust=False)
    
    if raw_data.empty and ".TW" in target_sym:
        target_sym = target_sym.replace(".TW", ".TWO")
        raw_data = yf.download(target_sym, period="2y", auto_adjust=False)
        
    if raw_data.empty:
        return pd.DataFrame(), target_sym

    # 【核心索引清洗防線】相容新舊版 yfinance，精準提取真實價格
    clean_df = pd.DataFrame(index=raw_data.index)
    
    for col in ['Open', 'High', 'Low', 'Close']:
        if isinstance(raw_data.columns, pd.MultiIndex):
            # 如果是雙層索引，精準取出對應型態的第一層欄位
            clean_df[col] = raw_data.xs(col, axis=1, level=0).iloc[:, 0]
        else:
            clean_df[col] = raw_data[col]
            
    return clean_df, target_sym

if final_target_code:
    data, final_symbol = fetch_stock_data(final_target_code)

    if not data.empty:
        display_title = get_stock_display_name(final_symbol)
        st.subheader(f"📈 {display_title}")
        
        # 移除時區
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # 轉為純數值型態
        for col in ['Open', 'High', 'Low', 'Close']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 【最新交易日安全補丁】解決盤後時差導致最新一天 Close 是 NaN 的問題
        if len(data) >= 2:
            for col in ['Open', 'High', 'Low', 'Close']:
                if pd.isna(data[col].iloc[-1]):
                    data[col].iloc[-1] = data[col].iloc[-2]
        
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'], df['賣出原因'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期顯示'] = view_df.index.strftime('%Y-%m-%d')
        view_df['月份'] = view_df.index.strftime('%Y-%m')

        # 計算 X 軸刻度
        all_dates = view_df['日期顯示'].tolist()
        days = st.session_state.view_days

        if days == 10:
            tick_vals = all_dates
            tick_texts = all_dates
        elif days == 20:
            tick_vals = all_dates[::5]
            tick_texts = all_dates[::5]
        elif days == 30:
            tick_vals = all_dates[::10]
            tick_texts = all_dates[::10]
        else: 
            first_days = view_df.groupby('月份').head(1)
            tick_vals = first_days['日期顯示'].tolist()
            tick_texts = first_days['日期顯示'].tolist()

        if all_dates and (all_dates[-1] not in tick_vals):
            tick_vals.append(all_dates[-1])
            tick_texts.append(all_dates[-1])

        fig = go.Figure()
        
        # 繪製真實收盤價實價線
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['Close'],
            mode='lines+markers', name='實價',
            line=dict(color='#FFFFFF', width=2),
            marker=dict(size=4),
            connectgaps=True,
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['買入點'],
            mode='markers', name='買入',
            marker=dict(symbol='triangle-up', size=14, color='#FF3131'),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['賣出點'],
            mode='markers', name='賣出',
            marker=dict(symbol='circle', size=14, color='#39FF14'),
            customdata=view_df['賣出原因'],
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<br><b>滿足條件: %{customdata}</b><extra></extra>"
        ))
        
        fig.update_layout(
            height=500,
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
    else:
        st.error("無法取得該股票的市場真實價格資料，請確認代號是否正確。")
