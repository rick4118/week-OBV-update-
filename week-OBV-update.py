import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 寫入 streamlit 檔案：黃金參數版專用
with open('app_golden.py', 'w', encoding='utf-8') as f:
    f.write('''import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 網頁排版設定
st.set_page_config(
    page_title="黃金參數 ＋ 績優高波動白名單篩選器",
    page_icon="🎯",
    layout="wide"
)

# --- 密碼保護 ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "592888":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 系統安全驗證")
        st.text_input("請輸入密碼以存取黃金決策系統：", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 系統安全驗證")
        st.text_input("請輸入密碼以存取黃金決策系統：", type="password", on_change=password_entered, key="password")
        st.error("❌ 密碼錯誤，請重新輸入！")
        return False
    return True

if check_password():
    # --- 指標計算邏輯 ---
    def calculate_indicators(df, obv_ma_window=20):
        df = df.sort_index()
        close_diff = df["Close"].diff()
        volume_direction = np.where(close_diff > 0, df["Volume"], np.where(close_diff < 0, -df["Volume"], 0))
        volume_direction[0] = 0
        df["OBV"] = np.cumsum(volume_direction)
        df["OBV_MA"] = df["OBV"].rolling(window=obv_ma_window).mean()

        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["DIF"] = ema12 - ema26
        df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["DIF"] - df["DEA"]

        df["Vol_MA13"] = df["Volume"].rolling(window=13).mean()
        df["OBV_MA13"] = df["OBV"].rolling(window=13).mean()

        low_min = df["Low"].rolling(window=9).min()
        high_max = df["High"].rolling(window=9).max()
        rsv = 100 * ((df["Close"] - low_min) / (high_max - low_min))

        k_list, d_list = [], []
        current_k, current_d = 50.0, 50.0
        for r in rsv:
            if pd.isna(r):
                k_list.append(np.nan)
                d_list.append(np.nan)
            else:
                current_k = (2/3) * current_k + (1/3) * r
                current_d = (2/3) * current_d + (1/3) * current_k
                k_list.append(current_k)
                d_list.append(current_d)
        df["K9"] = k_list
        df["D9"] = d_list
        return df

    # --- 核心黃金參數回測引擎 ---
    def scan_ultimate_strategy_v4(ticker_list, start_date, end_date, break_window=13, volume_ratio_limit=10.0, obv_ratio_min=1.05, obv_ratio_max=10.0, min_k_value=63.0, max_k_value=75.0, min_dif_limit=0.05):
        all_matches = []
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        download_start = (start_dt - pd.Timedelta(days=365*1.5)).strftime("%Y-%m-%d")
        download_end = (end_dt + pd.Timedelta(days=7 + 24*7)).strftime("%Y-%m-%d")

        for ticker in ticker_list:
            try:
                df = yf.download(ticker, start=download_start, end=download_end, interval="1wk", auto_adjust=False, progress=False)
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=["Close"])
                if len(df) < 50:
                    continue

                df = calculate_indicators(df, 20)
                target_weeks = df[(df.index >= start_dt) & (df.index <= end_dt)]

                for current_date in target_weeks.index:
                    idx_loc = df.index.get_loc(current_date)
                    if idx_loc < 31:
                        continue

                    df_sub = df.iloc[:idx_loc+1]
                    latest = df_sub.iloc[-1]
                    prev = df_sub.iloc[-2]

                    # 條件 A: 13週剛突破 OBV 新高
                    last_n_weeks = df_sub["OBV"].iloc[-break_window:]
                    is_latest_nw_high = (latest["OBV"] == last_n_weeks.max())
                    prev_n_weeks = df_sub["OBV"].iloc[-(break_window+1):-1]
                    is_prev_nw_high = (prev["OBV"] == prev_n_weeks.max()) if len(prev_n_weeks) == break_window else False
                    if not (is_latest_nw_high and not is_prev_nw_high):
                        continue

                    # 條件 1: 均線多頭
                    ma_up = (latest["MA5"] > prev["MA5"]) and (latest["MA10" ] > prev["MA10"]) and (latest["MA20"] > prev["MA20"])
                    if not ma_up: 
                        continue

                    # 條件 2: MACD 柱狀體翻紅或轉強
                    hist_series = df_sub["MACD_Hist"].values
                    latest_hist = hist_series[-1]
                    prev_hist = hist_series[-2]
                    macd_ok = False
                    if latest_hist > 0:
                        red_count = sum(1 for val in reversed(hist_series) if val > 0)
                        if (red_count >= 1) and (red_count < 5) and (latest_hist > prev_hist):
                            macd_ok = True
                    if not macd_ok:
                        continue

                    # 條件 3: 成交量限制
                    prev_ma13_vol = df_sub["Vol_MA13"].iloc[-2]
                    volume_ratio = latest["Volume"] / prev_ma13_vol if prev_ma13_vol > 0 else 999
                    if volume_ratio > volume_ratio_limit:
                        continue

                    # 條件 4: 收盤價限制
                    last_30_weeks = df_sub.iloc[-30:]
                    max_obv_idx = last_30_weeks["OBV"].idxmax()
                    max_obv_week_close = df_sub.loc[max_obv_idx, "Close"]
                    if latest["Close"] > max_obv_week_close:
                        continue

                    # 條件 5: OBV 比值範圍
                    prev_ma13_obv = df_sub["OBV_MA13"].iloc[-2]
                    obv_ratio = latest["OBV"] / prev_ma13_obv if prev_ma13_obv != 0 else 0.0
                    if not (obv_ratio_min <= obv_ratio <= obv_ratio_max):
                        continue

                    # --- 核心防守濾網 ---
                    # 1. K9 必須在優勢起漲區間
                    if not (min_k_value <= latest["K9"] <= max_k_value):
                        continue
                    # 2. KD 必須為多頭交叉
                    if latest["K9"] <= latest["D9"]:
                        continue
                    # 3. 確保收盤價站在 20 週均線之上，且 20 週均線方向昂首向上
                    if latest["Close"] <= latest["MA20"] or latest["MA20"] <= prev["MA20"]:
                        continue
                    # 4. 確保長線 MACD DIF 多頭強度足夠
                    if latest["DIF"] <= min_dif_limit:
                        continue

                    # 計算 24 週獲利狀況
                    buy_price = latest["Close"]
                    future_df = df.iloc[idx_loc+1 : idx_loc+25]
                    if len(future_df) == 24:
                        max_future_close = future_df["Close"].max()
                        max_idx = future_df["Close"].idxmax()
                        max_loc = future_df.index.get_loc(max_idx) + 1
                        profit_pct = ((max_future_close - buy_price) / buy_price) * 100
                        profit_str = f"{profit_pct:+.2f}%"
                        week_str = f"第 {max_loc} 週"
                    else:
                        profit_str = "NA"
                        week_str = "NA"

                    all_matches.append({
                        "Date": current_date.strftime("%Y-%m-%d"),
                        "Ticker": ticker,
                        "Profit": profit_str,
                        "Week_Num": week_str,
                        "K9": f"{latest['K9']:.2f}",
                        "D9": f"{latest['D9']:.2f}",
                        "DIF": f"{latest['DIF']:.2f}",
                        "OBV_Ratio": f"{obv_ratio:.2f}"
                    })
            except:
                pass
        return pd.DataFrame(all_matches)

    # --- Streamlit 介面設計 ---
    st.title("🎯 黃金參數 ＋ 績優高波動白名單篩選器")
    st.markdown("本系統預設採用經過多目標網格優化 (Grid Search) 的**黃金防守參數**，專門捕捉績優高波動產業在最安全的**黃金起漲點**。")

    # 側邊欄設定
    st.sidebar.header("⚙️ 黃金參數防禦設定")
    
    # 完美參數預設值
    min_k = st.sidebar.slider("週 K9 容許下限 (起漲動能)", 50.0, 80.0, 63.0, 1.0)
    max_k = st.sidebar.slider("週 K9 容許上限 (追高防守)", 65.0, 90.0, 75.0, 1.0)
    min_obv_ratio = st.sidebar.slider("OBV 突破強度下限 (OBV_Ratio)", 1.0, 2.0, 1.05, 0.01)
    min_dif = st.sidebar.slider("週 MACD DIF 強度下限 (趨勢過濾)", 0.0, 1.0, 0.05, 0.01)
    
    st.sidebar.markdown("---之外的基本限制：")
    break_window = st.sidebar.number_input("OBV 新高計算週數", 5, 52, 13)
    vol_limit = st.sidebar.number_input("成交量限制倍數", 1.0, 30.0, 10.0, 0.5)

    # 績優高波動白名單
    white_list_dict = {
        "2603.TW": "長榮 (航運龍頭)",
        "2609.TW": "陽明 (航運主力)",
        "2615.TW": "萬海 (航運高彈性)",
        "6187.TWO": "萬潤 (半導體設備飆股)",
        "3037.TW": "欣興 (ABF 載板龍頭)",
        "3189.TW": "景碩 (載板高波動)",
        "2344.TW": "華邦電 (記憶體)",
        "1590.TW": "亞德客-KY (高階機械)",
        "3481.TW": "群創 (高波動面板)",
        "2206.TW": "三陽工業 (車用精品)"
    }
    
    st.subheader("📋 績優高波動白名單選擇 (預設全選)")
    selected_labels = st.multiselect(
        "請勾選欲納入篩選的白名單個股：",
        options=list(white_list_dict.values()),
        default=list(white_list_dict.values())
    )
    
    # 反查 Tickers
    selected_tickers = [k for k, v in white_list_dict.items() if v in selected_labels]

    # 自定義外加代號
    custom_tickers_input = st.text_input("✍️ 手動額外新增代號 (選填，如: 2330, 2454，多檔用逗號或空格隔開)：", "")
    if custom_tickers_input:
        cleaned_custom = [t.strip() for t in custom_tickers_input.replace(",", " ").replace("，", " ").split() if t.strip()]
        formatted_custom = [f"{t}.TW" if "." not in t else t for t in cleaned_custom]
        selected_tickers.extend(formatted_custom)

    # 時間範圍
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("⏰ 開始日期", datetime.now() - timedelta(days=365*2))
    with col_d2:
        end_date = st.date_input("⏰ 結束日期", datetime.now())

    run_btn = st.button("🎯 開始黃金參數篩選", use_container_width=True)

    if run_btn:
        if not selected_tickers:
            st.warning("⚠️ 請選擇至少一檔股票進行篩選！")
        else:
            with st.spinner("⏳ 正在深入分析大數據，篩選最安全的爆發買點..."):
                df_res = scan_ultimate_strategy_v4(
                    ticker_list=selected_tickers,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    break_window=break_window,
                    volume_ratio_limit=vol_limit,
                    obv_ratio_min=min_obv_ratio,
                    min_k_value=min_k,
                    max_k_value=max_k,
                    min_dif_limit=min_dif
                )
                
                st.subheader("🎉 篩選報告與回測結果")
                if not df_res.empty:
                    df_res = df_res.sort_values(by="Date")
                    unique_dates = df_res["Date"].unique()
                    st.success(f
