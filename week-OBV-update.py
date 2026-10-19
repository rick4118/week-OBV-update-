import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 設定網頁標題與排版
st.set_page_config(
    page_title="周K OBV 綜合篩選器 (受保護)",
    page_icon="📈",
    layout="centered"
)

# --- 密碼保護功能 ---
def check_password():
    """如果密碼正確則回傳 True，否則顯示輸入框並回傳 False"""
    def password_entered():
        if st.session_state["password"] == "592888":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 系統安全驗證")
        st.text_input(
            "請輸入密碼以存取周K OBV 篩選器：",
            type="password",
            on_change=password_entered,
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 系統安全驗證")
        st.text_input(
            "請輸入密碼以存取周K OBV 篩選器：",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("❌ 密碼錯誤，請重新輸入！")
        return False
    else:
        return True

# 只有密碼驗證通過，才顯示後續的 App 內容
if check_password():

    # --- OBV 與技術指標計算邏輯 ---
    def calculate_indicators(df, obv_ma_window=20):
        df = df.sort_index()

        # 1. OBV 計算
        close_diff = df["Close"].diff()
        volume_direction = np.where(close_diff > 0, df["Volume"], 
                                    np.where(close_diff < 0, -df["Volume"], 0))
        volume_direction[0] = 0
        df["OBV"] = np.cumsum(volume_direction)
        df["OBV_MA"] = df["OBV"].rolling(window=obv_ma_window).mean()

        # 2. 週 K 線 5T, 10T, 20T 均線
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()

        # 3. MACD 計算 (12, 26, 9)
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["DIF"] = ema12 - ema26
        df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["DIF" ] - df["DEA"] # 紅綠柱

        return df

    def scan_weekly_obv_stocks_v2(ticker_list, start_date, end_date, break_window=13, obv_ma_window=20, obv_ratio_limit=1.0):
        all_matches = []

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # 提前下載數據，同時也要確保突破日後續至少有 24 週（約 170 天）可供回測計算
        download_start = (start_dt - pd.Timedelta(days=365*1.5)).strftime("%Y-%m-%d")
        download_end = (end_dt + pd.Timedelta(days=7 + 24*7)).strftime("%Y-%m-%d")

        for ticker in ticker_list:
            try:
                df = yf.download(ticker, start=download_start, end=download_end, interval="1wk", auto_adjust=False, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=["Close"])

                if len(df) < 50:
                    continue

                df = calculate_indicators(df, obv_ma_window)

                # 尋找落在此時間區間內的週K棒索引
                target_weeks = df[(df.index >= start_dt) & (df.index <= end_dt)]

                for current_date in target_weeks.index:
                    idx_loc = df.index.get_loc(current_date)
                    if idx_loc < 31:
                        continue # 需要至少30週歷史資料

                    # 取得當前與歷史切片
                    df_sub = df.iloc[:idx_loc+1]
                    latest = df_sub.iloc[-1]
                    prev = df_sub.iloc[-2]

                    # 條件 A: 13週剛突破 OBV 新高
                    last_n_weeks = df_sub["OBV"].iloc[-break_window:]
                    is_latest_nw_high = (latest["OBV"] == last_n_weeks.max())

                    prev_n_weeks = df_sub["OBV"].iloc[-(break_window+1):-1]
                    is_prev_nw_high = (prev["OBV"] == prev_n_weeks.max()) if len(prev_n_weeks) == break_window else False
                    is_just_broke_nw = is_latest_nw_high and not is_prev_nw_high

                    if not is_just_broke_nw:
                        continue

                    # 條件 1: 周K線 5T, 10T, 20T 趨勢皆向上 (當周大於前周)
                    ma_up = (latest["MA5"] > prev["MA5"]) and (latest["MA10"] > prev["MA10"]) and (latest["MA20"] > prev["MA20"])
                    if not ma_up:
                        continue

                    # 條件 2: MACD 紅柱漸增且小於 5 根
                    hist_series = df_sub["MACD_Hist"]
                    red_count = 0
                    for val in reversed(hist_series.values):
                        if val > 0:
                            red_count += 1
                        else:
                            break

                    macd_ok = (red_count >= 1) and (red_count < 5) and (latest["MACD_Hist"] > prev["MACD_Hist"])
                    if not macd_ok:
                        continue

                    # 條件 3: 剛突破的 OBV 量能數值不能比前一週高出指定倍數
                    obv_ratio = latest["OBV"] / prev["OBV"] if prev["OBV"] != 0 else 999
                    if obv_ratio > obv_ratio_limit:
                        continue

                    # 條件 4: 當周收盤價不可高於往前推算 30 周內最高 OBV 那周的收盤價
                    last_30_weeks = df_sub.iloc[-30:]
                    max_obv_idx = last_30_weeks["OBV"].idxmax()
                    max_obv_week_close = df_sub.loc[max_obv_idx, "Close"]

                    if latest["Close"] > max_obv_week_close:
                        continue

                    # 通過所有篩選！計算往後推 24 週的獲利狀況
                    buy_price = latest["Close"]
                    future_df = df.iloc[idx_loc+1 : idx_loc+25] # 往後推 24 週
                    
                    # 必須確保未來剛好有 24 週的數據，不足一律顯示 NA
                    if len(future_df) == 24:
                        max_future_close = future_df["Close"].max()
                        max_idx = future_df["Close"].idxmax()
                        max_loc = future_df.index.get_loc(max_idx) + 1 # 取得第幾週
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
                        "Week_Num": week_str
                    })
            except Exception as e:
                pass

        return pd.DataFrame(all_matches)

    # --- App 主介面 ---
    st.title("📈 周K OBV 綜合決策篩選器")
    st.write("此為綜合多重指標篩選系統，需完全符合 OBV 突破、均線多頭、MACD 初升段、突破量能限制及收盤價限制。")

    # 參數設定區塊
    st.subheader("⚙️ 參數設定")
    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        break_window = st.number_input("OBV 新高計算週數 (預設 13 週)", min_value=5, max_value=52, value=13, step=1)
    with col_param2:
        ma_window = st.number_input("OBV 均線週數 (預設 20 週)", min_value=5, max_value=100, value=20, step=1)
    with col_param3:
        obv_ratio_limit = st.number_input("OBV 限制倍數 (預設 1.0 倍)", min_value=0.5, max_value=5.0, value=1.0, step=0.1, format="%.1f")

    # 時間區間選擇
    st.subheader("📅 選擇搜尋時間區間")
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("開始日期:", datetime.now() - pd.Timedelta(days=90))
    with col_date2:
        end_date = st.date_input("結束日期:", datetime.now())

    default_tickers = "2330, 1519, 2603, 2605, 1777.TWO"
    tickers_input = st.text_area(
        "輸入股票代號 (支援空格、逗號、換行分隔):",
        value=default_tickers,
        height=120
    )

    run_button = st.button("🎯 開始條件篩選", use_container_width=True)

    if run_button:
        if start_date > end_date:
            st.error("❌ 錯誤：開始日期不可晚於結束日期！")
        else:
            with st.spinner("⏳ 正在分析歷史數據，請稍候..."):
                raw_input = tickers_input
                for char in [',', ';', '\n', '\t', '，', '；']:
                    raw_input = raw_input.replace(char, ' ')
                cleaned_tickers = [t.strip() for t in raw_input.split(' ') if t.strip()]

                if not cleaned_tickers:
                    st.warning("⚠️ 請輸入至少一個股票代號！")
                else:
                    formatted_tickers = [f"{t}.TW" if "." not in t else t for t in cleaned_tickers]
                    df_res = scan_weekly_obv_stocks_v2(
                        formatted_tickers,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        break_window=break_window,
                        obv_ma_window=ma_window,
                        obv_ratio_limit=obv_ratio_limit
                    )

                    st.subheader("🎉 嚴格篩選結果")

                    if not df_res.empty:
                        df_res = df_res.sort_values(by="Date")
                        unique_dates = df_res["Date"].unique()

                        st.success(f"✨ 共有 {len(df_res)} 筆交易週紀錄符合條件：")

                        for date_str in unique_dates:
                            st.markdown(f"### 📅 週 K 對應日期: `{date_str}`")
                            matching_sub = df_res[df_res["Date"] == date_str]
                            
                            # 格式化輸出，加上獲利%與第幾週
                            result_strings = []
                            for _, row in matching_sub.iterrows():
                                result_strings.append(f"{row['Ticker']} (獲利: {row['Profit']}, {row['Week_Num']})")
                                    
                            st.code(" | ".join(result_strings), language="text")

                        st.info("💡 提示：點擊右上方『📋』按鈕可以快速複製該日期底下所有符合條件與獲利成效的股票資訊。")
                    else:
                        st.error("❌ 在此指定時間區間內，無任何個股完全符合所有篩選條件。")
