# --- 數據獲取邏輯：自動偵測上市(.TW)與上櫃(.TWO) ---
if symbol:
    with st.spinner(f"正在搜尋 {symbol} 的真實市場數據..."):
        # 1. 嘗試上市標的
        ticker_str = f"{symbol}.TW"
        data = yf.download(ticker_str, period="2y", auto_adjust=False)
        
        # 2. 如果上市沒資料，嘗試上櫃標的
        if data.empty:
            ticker_str = f"{symbol}.TWO"
            data = yf.download(ticker_str, period="2y", auto_adjust=False)
        
        if not data.empty:
            # --- 重要修正：處理 yfinance 的多重索引結構 ---
            # 當只下載一個代碼時，yfinance 有時會回傳兩層欄位（例如：('Close', '2330.TW')）
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # 清理欄位名稱（移除空格、轉為首字母大寫，確保計算公式抓得到）
            data.columns = [str(c).strip().capitalize() for c in data.columns]
            
            # 檢查關鍵欄位是否存在
            required = ['Close', 'Open', 'High', 'Low']
            if all(col in data.columns for col in required):
                # 這裡接你原本的計算與繪圖邏輯
                df_processed = calculate_indicators(data)
                df_processed['Buy_Sig'], df_processed['Sell_Sig'] = get_signal_markers(df_processed)
                
                # 顯示圖表... (其餘程式碼不變)
                st.success(f"✅ 成功讀取 {ticker_str}")
            else:
                st.error(f"資料欄位異常，請檢查資料源。欄位列表：{list(data.columns)}")
        else:
            st.error(f"❌ 找不到代碼 '{symbol}'。請確認：上市請輸 2330，上櫃請輸 8069。")
