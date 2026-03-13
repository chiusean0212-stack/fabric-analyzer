import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

# --- 介面設定 ---
t = {"繁體中文": ["廣笠機械", "AI 分析系統", "📸 上傳照片", "結果", "授權登入"],
     "English": ["Goang Lih", "AI Analysis", "📸 Upload", "Result", "Login"]}[st.sidebar.selectbox("Lang", ["繁體中文", "English"])]

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    pwd = st.text_input(t[4], type="password")
    if st.button(t[4]):
        if pwd == "777": st.session_state["auth"] = True; st.rerun()
    st.stop()

st.title(f"{t[0]} Goang Lih")

up = st.file_uploader(t[2], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 精細預處理：使用中值濾波取代高斯模糊，保護 82 WPI 的邊緣
        denoise = cv2.medianBlur(gray, 3)
        enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(denoise)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 2. 定義兩個關鍵視窗：高頻區 (50-100 WPI) 與 結構區 (20-50 WPI)
        # Lag 9-18 (高頻) vs Lag 19-45 (結構)
        high_freq = corr[9:19]   # 對應 ~95 to 48 WPI
        low_freq = corr[19:46]   # 對應 ~47 to 20 WPI
        
        p_high = np.argmax(high_freq) + 9
        p_low = np.argmax(low_freq) + 19
        
        # 3. 智慧決策邏輯 (核心修正)
        # 如果高頻區的最強峰值 能量超過 結構區最強峰值的 85%
        # 代表這是一塊細密布料 (如白 82 或 灰 53)，必須取高頻值
        if corr[p_high] > corr[p_low] * 0.85:
            best_lag = p_high
            # 針對透白 75 的特別處理：
            # 如果抓到 Lag 12 (75 WPI)，檢查它的兩倍距離是否存在強峰值
            if best_lag < 14:
                check_lag = best_lag * 2
                if corr[check_lag] > corr[best_lag] * 0.75:
                    best_lag = check_lag
        else:
            # 否則，這是一塊大循環布料 (如桃紅 28)
            best_lag = p_low

        wpi = round(900 / best_lag)
        
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
