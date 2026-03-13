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
        
        # 影像預處理：強化對比，不使用模糊，保留所有細節
        enhanced = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8)).apply(gray)
        h, w = enhanced.shape
        roi = enhanced[:, w//2-450 : w//2+450]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 搜尋範圍：Lag 8 (112 WPI) 到 Lag 60 (15 WPI)
        s_start, s_end = 8, 60
        lags = corr[s_start:s_end]
        
        # --- 核心優化：權重線性補償 (解決減半問題) ---
        # 越小的 Lag (高 WPI) 越難被偵測，所以我們給予線性補強
        # 從 1.6 倍 (Lag 8) 線性降到 1.0 倍 (Lag 60)
        w_factor = np.linspace(1.6, 1.0, len(lags))
        weighted_lags = lags * w_factor
        
        # 取得加權後的最強峰值
        best_idx = np.argmax(weighted_lags)
        best_lag = best_idx + s_start
        
        # --- 智慧防跳號 (針對灰 53/26 與 透白 75/38) ---
        # 如果選中的是 20-30 WPI 區間，檢查它是否有更強的「倍頻」
        if 25 <= best_lag <= 40:
            half_lag = int(best_lag / 2)
            if half_lag >= s_start:
                # 如果一半位置的訊號有 80% 強，代表它是真正的紗線 (如灰 53)
                if corr[half_lag] > corr[best_lag] * 0.8:
                    best_lag = half_lag
        
        # 計算 WPI (使用 910 係數)
        wpi = round(910 / best_lag)
        
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
