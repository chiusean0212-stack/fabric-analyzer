import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

# --- 1. 介面設定 ---
t_data = {"繁體中文": ["廣笠機械", "AI 分析系統", "📸 上傳照片", "結果", "授權登入"],
          "English": ["Goang Lih", "AI Analysis", "📸 Upload", "Result", "Login"]}
sel = st.sidebar.selectbox("Lang", ["繁體中文", "English"])
t = t_data[sel]

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    pwd = st.text_input(t[4], type="password")
    if st.button(t[4]):
        if pwd == "777": st.session_state["auth"] = True; st.rerun()
    st.stop()

st.title(f"{t[0]} Goang Lih")

# --- 2. 核心分析邏輯 ---
up = st.file_uploader(t[2], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 影像預處理：適度去噪並強化垂直邊緣 (不使用過重的模糊)
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-450 : w//2+450]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        # 投影與自相關
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # --- 3. 智慧區間鎖定 (精準解決 26 vs 53 問題) ---
        # 搜尋範圍 Lag 8 (~112 WPI) 到 Lag 60 (~15 WPI)
        s_start, s_end = 8, 60
        lags = corr[s_start:s_end]
        
        # 取得最強峰值
        p1_idx = np.argmax(lags)
        best_lag = p1_idx + s_start
        
        # --- 4. 關鍵：升頻修正邏輯 ---
        # 如果初步偵測出 26 WPI (Lag 34 附近)，我們會回頭檢查 52-53 WPI (Lag 17 附近)
        # 只要半頻位置有任何像樣的峰值，就判定為真正的紗線
        half_lag = int(best_lag / 2)
        if half_lag >= s_start:
            # 檢查半頻位置的強度，如果它達到最強峰值的 50% 以上，就強制升頻
            if corr[half_lag] > lags[p1_idx] * 0.5:
                best_lag = half_lag
        
        # --- 5. 針對透白 75 的降頻修正 ---
        # 如果結果大於 70 WPI，檢查兩倍距離
        if best_lag < 13:
            double_lag = best_lag * 2
            if double_lag < s_end:
                if corr[double_lag] > corr[best_lag] * 0.7:
                    best_lag = double_lag

        wpi = round(900 / best_lag)
        
        # --- 6. 顯示結果 ---
        st.image(up, use_container_width=True)
        st.markdown(f"""
            <div style='text-align:center; background:#f0f2f6; padding:20px; border-radius:15px; border:2px solid #1E3A8A;'>
                <h2 style='color:#1E3A8A;'>{t[3]}</h2>
                <p style='font-size:80px; font-weight:bold; color:#FF0000; margin:0;'>WPI = {wpi}</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("© 2026 Goang Lih Machinery Co., Ltd.")
