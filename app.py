import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io

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

# --- 2. 動態抓取最新台股上市櫃與主要 ETF 資料庫 ---
@st.cache_data(ttl=86400)  # 快取 24 小時
def fetch_taiwan_stocks():
    urls = {
        "上市": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "上櫃": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    }
    
    stock_list = []
    # 偽裝成真人 Chrome 瀏覽器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    
    for market_type, url in urls.items():
        try:
            # 安全破防 403：先用 requests 把網頁文字下載下來
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'ms950'  # 修正台股網頁萬年 CP950 編碼問題
            
            # 將下載下來的網頁純文字，包裝成記憶體緩衝流（StringIO）餵給 pandas
            # 這樣 pandas 就不會直接去撞證交所牆壁，完美破解 403 Forbidden
            html_data = io.StringIO(response.text)
            dfs = pd.read_html(html_data)
            
            target_df = None
            # 遍歷所有表格，尋找真正含有股票資料的表格
            for table in dfs:
                if table.shape[1] > 0 and table.iloc[0].astype(str).str.contains('有價證券代號及名稱').any():
                    target_df = table.copy()
                    break
            
            if target_df is None:
                continue
                
            # 設定標準欄位標題
            target_df.columns = target_df.iloc[0]
            target_df = target_df.iloc[1:]
            target_df = target_df.dropna(subset=['有價證券代號及名稱'])
            
            is_valid_section = False
            for _, row in target_df.iterrows():
                text = str(row['有價證券代號及名稱']).strip()
                
                # 判斷是否進入股票或受益憑證區段
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
                    
                    # 條件一：剛好 4 碼純數字（一般個股）
                    is_normal_stock = (len(stock_id) == 4 and stock_id.isdigit())
                    
                    # 條件二：5 或 6 碼，且必須是 00 開頭（精準保留各類 ETF，允許帶英文後綴）
                    is_etf = (len(stock_id) in [5, 6] and stock_id.startswith("00"))
                    
                    if is_normal_stock or is_etf:
                        if "特" not in stock_name and "債" not in stock_name:
                            stock_list.append({"id": stock_id, "name": stock_name})
        except Exception as e:
            st.error(f"無法從 {market_type} 網址獲取資料: {e}")
            
    # 去除可能重複的代碼並排序
    unique_stocks = {item['id']: item['name'] for item in stock_list}
    sorted_list = [{"id": k, "name": v} for k, v in sorted(unique_stocks.items())]
    return sorted_list

# 載入資料
with st.spinner("正在即時同步最新台股與 ETF 清單..."):
    ALL_STOCKS_DATA = fetch_taiwan_stocks()

ALL_STOCKS = {item['id']: item['name'] for item in ALL_STOCKS_DATA}
ALL_STOCKS_LIST = [f"{item['id']} {item['name']}" for item in ALL_STOCKS_DATA]

def get_stock_display_name(symbol):
    pure_code = symbol.split('.')[0].strip()
    if pure_code in ALL_STOCKS:
        return f"{ALL_STOCKS[pure_code]} ({symbol})"
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

# --- 4. 核心買賣訊號邏輯（三重共振） ---
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

# --- 5. 主畫面：股票搜尋與選擇欄 ---
st.title("🚀 股票買賣時機觀測")

if "final_target_code" not in st.session_state:
    st.session_state.final_target_code = "2330"

# 安全尋找預設值索引
default_index = 0
for idx, item in enumerate(ALL_STOCKS_LIST):
    if item.startswith("2330"):
        default_index = idx
        break

selected_stock_str = st.selectbox(
    "請輸入或選擇股票代碼：",
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
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    target_sym = f"{symbol}.TW" if "." not in symbol else symbol
    data = yf.download(target_sym, period="2y", auto_adjust=False)
    if data.empty and ".TW" in target_sym:
        target_sym = target_sym.replace(".TW", ".TWO")
        data = yf.download(target_sym, period="2y", auto_adjust=False)
    return data, target_sym

if final_target_code:
    data, final_symbol = fetch_stock_data(final_target_code)

    if not data.empty:
        display_title = get_stock_display_name(final_symbol)
        st.subheader(f"📈 {display_title}")
        
        if hasattr(data.columns, 'levels') or ('MultiIndex' in type(data.columns).__name__):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'], df['賣出原因'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期顯示'] = view_df.index.strftime('%Y-%m-%d')
        view_df['月份'] = view_df.index.strftime('%Y-%m')

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
        
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['Close'],
            mode='lines', name='實價',
            line=dict(color='#FFFFFF', width=2),
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
