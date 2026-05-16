import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 設定網頁標題與整體外觀
st.set_page_config(page_title="台股 K 線與買賣時機觀測站", layout="wide")

# --- 1. 完整台股清單資料池 (包含你先前提供的所有股票代碼) ---
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

# --- 2. 輔助函式：格式化名稱與計算指標 ---
ALL_STOCKS_LIST = [f"{k} {v}" for k, v in ALL_STOCKS.items()]

def get_stock_display_name(symbol):
    pure_code = symbol.split('.')[0].strip()
    if pure_code in ALL_STOCKS:
        return f"{ALL_STOCKS[pure_code]} ({symbol})"
    return symbol

def calculate_indicators(df):
    df = df.copy().astype(float)
    # 計算 5日、10日移動平均線 (MA)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    # 計算 RSI5、RSI10
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    df['RSI5'] = 100 - (100 / (1 + (gain.rolling(5).mean() / loss.rolling(5).mean())))
    df['RSI10'] = 100 - (100 / (1 + (gain.rolling(10).mean() / loss.rolling(10).mean())))
    
    # 計算 KDJ (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    return df

# --- 3. 核心訊號邏輯 (買入邏輯 / 賣出三重共振同時成立) ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    sell_reasons = [""] * len(df)
    in_position = False 
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if not in_position:
            # 買入訊號：MA黃金交叉 (5日線穿過10日線) 且 RSI5 > 50
            if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) and (row['RSI5'] > 50):
                buy_markers[i] = row['Close']
                in_position = True
        elif in_position:
            # 賣出三條件檢查
            cond_rsi = (row['RSI5'] < row['RSI10'])
            cond_price = (row['Close'] < row['MA5'])
            cond_kdj = (row['K'] < row['D'])
            
            # 三者必須「同時成立」
            if cond_rsi and cond_price and cond_kdj:
                sell_markers[i] = row['Close']
                sell_reasons[i] = "RSI死叉 + KDJ死叉 + 跌破MA5"
                in_position = False 
    return buy_markers, sell_markers, sell_reasons

# --- 4. UI 側邊欄：【字首鎖定型 Selectbox 核心控制機制】 ---
st.sidebar.title("🚀 股票買賣時機")

# 用 session_state 後台記憶體來動態維護「被過濾後的下拉名單選項」
if "filtered_options" not in st.session_state:
    st.session_state.filtered_options = ALL_STOCKS_LIST

# 渲染出標準型態的 selectbox（可以手打字、可以往下滑，外觀型態 100% 如你所指）
selected_stock_str = st.sidebar.selectbox(
    "請選擇或輸入股票代碼：",
    options=st.session_state.filtered_options,
    index=0 if "2330 台積電" not in st.session_state.filtered_options else st.session_state.filtered_options.index("2330 台積電"),
    key="stock_selectbox_widget"
)

# 鋼鐵字首對齊的核心過濾心臟：
# 拿到目前你在選單內正選中、或是正嘗試打在框框裡的代碼字首
current_input = selected_stock_str.split(" ")[0] if selected_stock_str else ""

if current_input:
    # 進行 Python 鋼鐵字首過濾：唯有代碼「開頭」符合的股票才准留下
    # 當你手打 223，這段程式碼會徹底在後台剔除 2023、2423 這種包含型模糊雜魚，名單只剩 2231、2233
    new_options = [
        f"{k} {v}" for k, v in ALL_STOCKS.items()
        if k.startswith(current_input)
    ]
    
    # 如果過濾後的名單存在且跟當前選單狀態不一樣，即時更新網頁，讓下拉選單只顯示完美對齊的股票
    if new_options and new_options != st.session_state.filtered_options:
        st.session_state.filtered_options = new_options
        st.rerun()
else:
    # 若欄位被全部刪除清空，則恢復最初完整台股名單
    if st.session_state.filtered_options != ALL_STOCKS_LIST:
        st.session_state.filtered_options = ALL_STOCKS_LIST
        st.rerun()

# 最終提煉出要拿去 Yahoo Finance 抓真實報價的股票數字代碼
final_target_code = selected_stock_str.split(" ")[0] if selected_stock_str else "2330"


# --- 5. 主頁面：觀測天數週期切換面板 ---
if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期選擇")
p_cols = st.columns(6)
periods = {"10天": 10, "20天": 20, "30天": 30, "60天": 60, "120天": 120, "240天": 240}
for i, (lab, val) in enumerate(periods.items()):
    if p_cols[i].button(lab):
        st.session_state.view_days = val

# --- 6. 股票數據抓取（介接 yfinance 真實市場價值數據） ---
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    target_sym = f"{symbol}.TW" if "." not in symbol else symbol
    data = yf.download(target_sym, period="2y", auto_adjust=False)
    # 如果上市（.TW）沒撈到資料，自動切換至上櫃（.TWO）再試一次，雙重保險
    if data.empty and ".TW" in target_sym:
        target_sym = target_sym.replace(".TW", ".TWO")
        data = yf.download(target_sym, period="2y", auto_adjust=False)
    return data, target_sym

if final_target_code:
    data, final_symbol = fetch_stock_data(final_target_code)

    if not data.empty:
        # 顯示股票正確名稱與代號標題
        display_title = get_stock_display_name(final_symbol)
        st.subheader(f"📈 {display_title}")
        
        # 多層欄位相容性處理（封殺任何可能因 yfinance 升級造成的多重索引噴錯）
        if hasattr(data.columns, 'levels') or ('MultiIndex' in type(data.columns).__name__):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        
        # 計算各項指標與標註買賣信號點
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'], df['賣出原因'] = get_signal_markers(df)
        
        # 截取用戶選擇觀測的最後幾天區間
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期顯示'] = view_df.index.strftime('%Y-%m-%d')
        view_df['月份'] = view_df.index.strftime('%Y-%m')

        # 針對不同天數區間自動分配 x 軸刻度間隔，確保小螢幕看也不會擠成一團字
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

        # --- 7. Plotly 專業黑系 K 線與買賣標籤圖表繪製 ---
        fig = go.Figure()
        
        # 實價折線
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['Close'],
            mode='lines', name='實價',
            line=dict(color='#FFFFFF', width=2),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        # 紅色上三角型買入信號
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['買入點'],
            mode='markers', name='買入信號點',
            marker=dict(symbol='triangle-up', size=14, color='#FF3131'),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        # 綠色圓形賣出信號
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
        
        # 隱藏不必要的 plotly 浮動工具列，讓畫面最簡潔
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.warning("⚠️ 賣出核心提醒：RSI死叉 + KDJ死叉 + 跌破5日線 (三個條件須在同一天完全共振成立)")
    else:
        st.error("❌ 無法從 Yahoo Finance 獲取該股票的即時實價數據，請確認代碼是否輸入正確或市面上是否有此股票。")
