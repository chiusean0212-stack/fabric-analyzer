import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

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
        
        # --- 1. 核心修正：移除模糊，改用高對比銳化 ---
        # 這樣 82 WPI 的細線才會比 41 WPI 的粗循環更明顯
        enhanced = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8)).apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        
        # 使用 Sobel 提取垂直細節
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # --- 2. 搜尋範圍擴大 (針對 15 - 110 WPI) ---
        # Lag 8 (~112 WPI) 到 Lag 60 (~15 WPI)
        s_start, s_end = 8, 60
        lags = corr[s_start:s_end]
        
        # --- 3. 採用「高頻優先加權」 ---
        # 對較細的間距 (Lag 小的) 給予 1.5 倍到 1.0 倍的漸進補償
        # 這是為了強迫 AI 從 41 跳回 82，從 26 跳回 53
        weights = np.linspace(1.5, 1.0, len(lags))
        best_lag = np.argmax(lags * weights) + s_start
        
        # --- 4. 針對「透白」的倍頻壓制 ---
        # 如果初步偵測到超過 90 WPI (Lag < 10)，且 38 WPI 處有強訊號，才降頻
        if best_lag < 11:
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
