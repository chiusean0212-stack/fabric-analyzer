import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

# --- 介面語言 ---
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
        
        # 1. 影像處理：使用中等強度的 GaussianBlur 壓制可能導致 112/75 的細微雜訊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blur)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 2. 搜尋範圍鎖定 (15WPI ~ 100WPI)
        s_start, s_end = 9, 60
        lags = corr[s_start:s_end]
        
        # 3. 關鍵修正：採用「結構優先加權」
        # 我們對較大的 Lag (較低的 WPI，如桃紅 28 / 灰色 53) 給予權重優勢
        # 這能防止桃紅 28 被誤判為 56 或 112
        # 權重由 Lag 60 (1.4倍) 遞減至 Lag 9 (1.0倍)
        weights = np.linspace(1.0, 1.4, len(lags)) 
        best_lag = np.argmax(lags * weights) + s_start
        
        # 4. 二階驗證 (針對白色 82 WPI)
        # 如果最強訊號在 28-53 區間，但極高頻 (Lag 9-11) 有一個超強的獨立峰值
        # 才判定為高密布料，否則維持結構優先
        if best_lag > 15: # 目前判定為中低 WPI
            high_freq_zone = corr[9:13]
            if np.max(high_freq_zone) > corr[best_lag] * 1.2: # 只有高頻訊號極強時才切換
                best_lag = np.argmax(high_freq_zone) + 9

        wpi = round(900 / best_lag)
        
        # 顯示結果
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
