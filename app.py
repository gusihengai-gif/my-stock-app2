import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 設定網頁標題與整體外觀
st.set_page_config(page_title="台股 K 線與買賣時機觀測站", layout="wide")

# --- 1. 完整台股清單資料池 ---
ALL_STOCKS = {
    "1101": "台泥", "1102": "亞泥", "1216": "統一", "1301": "台塑", "1303": "南亞",
    "1326": "台化", "1402": "遠東新", "1503": "士電", "1504": "東元", "1513": "中興電",
    "1519": "華城", "1590": "亞德客-KY", "1605": "華新", "1722": "台肥", "2002": "中鋼",
    "2023": "志聯", "2101": "泰豐", "2102": "泰豐二", "2105": "正新", "2201": "裕隆",
    "2204": "中華", "2206": "三陽工業", "2231": "為升", "2233": "宇隆", "2236": "百達-KY",
    "2239": "英利-KY", "2301": "光寶科", "2303": "聯電", "2308": "台達電", "2317": "鴻海",
    "2324": "仁寶", "2327": "國巨", "2330": "台積電", "2344": "華邦電", "2345": "智邦",
    "2352": "佳世達", "2353": "宏碁", "2356": "英業達", "2357": "華碩", "2360": "致茂",
    "2371": "大同", "2376": "技嘉", "2377": "微星", "2379": "瑞昱", "2382": "廣達",
    "2383": "台光電", "2395": "研華", "2408": "南亞科", "2409": "友達", "2412": "中華電",
    "2423": "固緯", "2454": "聯發科", "2474": "可成", "2498": "宏達電", "2603": "長榮",
    "2606": "裕民", "2609": "陽明", "2610": "華航", "2615": "萬海", "2618": "長榮航",
    "2801": "彰銀", "2880": "華南金", "2881": "富邦金", "2882": "國泰金", "2883": "開發金",
    "2884": "玉山金", "2885": "元大金", "2886": "兆豐金", "2887": "台新金", "2888": "新光金",
    "2890": "永豐金", "2891": "中信金", "2892": "第一金", "2912": "統一超", "3008": "大立光",
    "3017": "奇鋐", "3034": "聯詠", "3037": "欣興", "3045": "台灣大", "3231": "緯創",
    "3443": "創意", "3661": "世芯-KY", "3711": "日月光投控", "4904": "遠傳", "4938": "和碩",
    "4958": "臻鼎-KY", "4966": "譜瑞-KY", "5871": "中租-KY", "5876": "上海商銀", "5880": "合庫金",
    "6415": "矽力*-KY", "6505": "台塑化", "6669": "緯穎", "8046": "南電", "8454": "富邦媒",
    "9904": "寶成", "9910": "豐泰", "9921": "巨大", "9945": "潤泰新",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息", "00919": "群益台灣精選高息"
}

def get_stock_display_name(symbol):
    pure_code = symbol.split('.')[0].strip()
    if pure_code in ALL_STOCKS:
        return f"{ALL_STOCKS[pure_code]} ({symbol})"
    return symbol

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

# --- 3. 核心買賣訊號邏輯 ---
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

# --- 4. UI 側邊欄：【唯一純淨搜尋欄 ＆ 免按 Enter 自動觸發機制】 ---
st.sidebar.title("🚀 股票買賣時機")

# 使用 session_state 在後台牢牢記憶當前鎖定的最終代碼（預設台積電）
if "current_confirmed_code" not in st.session_state:
    st.session_state.current_confirmed_code = "2330"

# 畫面上唯一的、乾乾淨淨的輸入框
user_input = st.sidebar.text_input("輸入股票代碼：", value=st.session_state.current_confirmed_code, key="pure_single_search_bar").strip()

# 【免按 Enter 動態追蹤心臟】
if user_input:
    # 情況 A：如果輸入的內容正好完美對齊清單中的某檔股票（例如輸入 2231 或 0050）
    if user_input in ALL_STOCKS:
        if user_input != st.session_state.current_confirmed_code:
            st.session_state.current_confirmed_code = user_input
            st.rerun()
    else:
        # 情況 B：進行字首模糊強鎖定
        matched_keys = [k for k in ALL_STOCKS.keys() if k.startswith(user_input)]
        
        # 如果打字打到一半，剛好篩到只剩下一檔股票，就自動幫你切換（免按 Enter）
        if len(matched_keys) == 1:
            if st.session_state.current_confirmed_code != matched_keys[0]:
                st.session_state.current_confirmed_code = matched_keys[0]
                st.rerun()
        # 如果輸入的是打到一半的字首（例如 1、11、22），不做任何跳轉，保持當前畫面，絕不噴錯！
        else:
            pass

final_target_code = st.session_state.current_confirmed_code

# --- 5. 主頁面：觀測天數週期切換面板 ---
if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期選擇")
p_cols = st.columns(6)
periods = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}
for i, (lab, val) in enumerate(periods.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

# --- 6. 股票數據抓取 與 圖表渲染 ---
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
            mode='markers', name='買入信號點',
            marker=dict(symbol='triangle-up', size=14, color='#FF3131'),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['賣出點'],
            mode='markers', name='賣出信號點',
            marker=dict(symbol='circle', size=14, color='#39FF14'),
            customdata=view_df['賣出原因'],
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<br><b>滿足條件: %{customdata}</b><extra></extra>"
        ))
        
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
            yaxis=dict(autorange=True, fixedrange=True, title="實價 (TWD)", tickformat='.2f'),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.warning("⚠️ 賣出核心提醒：RSI死叉 + KDJ死叉 + 跌破5日線 (三個條件須在同一天完全共振成立)")
    else:
        st.error("❌ 無法從 Yahoo Finance 獲取該股票的即時實價數據。")
