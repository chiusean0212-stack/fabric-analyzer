import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

# --- 介面設定 ---
lang_data = {
    "繁體中文": ["廣笠機械", "AI 分析系統", "📸 上傳照片 (高精細版)", "結果", "授權登入"],
    "English": ["Goang Lih", "AI Analysis", "📸 Upload (High-Res)", "Result", "Login"]
}
sel = st.sidebar.selectbox("Lang", ["繁體中文", "English"])
t = lang_data[sel]

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    pwd = st.text_input(t[4], type="password")
    if st.button(t[4]):
        if pwd == "777": st.session_state["auth"] = True; st.rerun()
    st.stop()

st.title(f"{t[0]} Goang Lih")

# --- 核心演算法：高頻偵測強化 ---
up = st.file_uploader(t[2], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 針對高密 (83 WPI) 與透白 (38 WPI) 的預處理
        # 捨棄 Bilateral，改用簡單的輕微高斯去噪，避免抹除 83 WPI 的細小紗線
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 提高對比度至 5.0，強制拉出透光布料的細微輪廓
        enhanced = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4)).apply(blur)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-450 : w//2+450]
        
        # 2. 邊緣提取：混合 Sobel 與銳化
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_x = np.absolute(grad_x)
        
        # 3. 投影與自相關
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 4. 關鍵：擴大搜尋區間以涵蓋 83 WPI
        # Lag 8 (~112 WPI) 到 Lag 60 (~15 WPI)
        s_start, s_end = 8, 60 
        lags = corr[s_start:s_end]
        
        # 5. 高 WPI 權重補償 (線性補強高頻訊號)
        # 讓系統更容易抓到 Lag 10-25 之間的訊號 (對應 35-90 WPI)
        weights = np.linspace(1.2, 1.0, len(lags)) 
        weighted_lags = lags * weights
        
        best_lag = np.argmax(weighted_lags) + s_start
        
        # 6. 計算 WPI
        wpi = round(900 / best_lag)
        
        # --- 顯示結果 ---
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
