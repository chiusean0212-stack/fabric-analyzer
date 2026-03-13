import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

# --- 簡約介面 ---
t = {"繁體中文": ["廣笠機械", "AI 分析系統", "📸 上傳照片", "結果", "授權登入"],
     "English": ["Goang Lih", "AI Analysis", "📸 Upload", "Result", "Login"]}[st.sidebar.selectbox("Lang", ["繁體中文", "English"])]

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    pwd = st.text_input(t[4], type="password")
    if st.button(t[4]):
        if pwd == "777": st.session_state["auth"] = True; st.rerun()
    st.stop()

st.title(f"{t[0]} Goang Lih")

# --- 核心演算：能量動態補償 ---
up = st.file_uploader(t[2], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用溫和的 GaussianBlur 確保細緻的 82 WPI 不被抹除
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(blur)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # --- 關鍵修正：搜尋範圍與高頻權重 ---
        # 搜尋區間 Lag 10 (~90 WPI) 到 Lag 45 (~20 WPI)
        # 這能強制封殺 100 WPI 以上的雜訊，並防止跳到 20 WPI 以下的誤判
        s_start, s_end = 10, 45 
        lags = corr[s_start:s_end]
        
        # 採用遞減權重：讓高 WPI (小 Lag) 的訊號更容易被選中
        # 這能解決「白色 82 變 41」以及「灰色 53 變 26」的問題
        weights = np.linspace(1.3, 1.0, len(lags)) 
        best_lag = np.argmax(lags * weights) + s_start
        
        # --- 針對「透白 75」的最後防線 ---
        # 如果初步結果 > 70 (Lag < 13)，我們檢查是否有 35-38 附近的強訊號
        if best_lag < 13:
            check_lag = best_lag * 2
            if check_lag < s_end and corr[check_lag] > corr[best_lag] * 0.8:
                best_lag = check_lag

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
