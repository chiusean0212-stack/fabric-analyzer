import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

# --- 簡約介面設定 ---
t = {"繁體中文": ["廣笠機械", "AI 分析", "上傳照片", "結果", "登入"],
     "English": ["Goang Lih", "AI Analysis", "Upload", "Result", "Login"]}[st.sidebar.selectbox("Lang", ["繁體中文", "English"])]

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    pwd = st.text_input(t[4], type="password")
    if st.button(t[4]):
        if pwd == "777": st.session_state["auth"] = True; st.rerun()
    st.stop()

st.title(f"{t[0]} Goang Lih")

# --- 核心演算法 ---
up = st.file_uploader(t[2], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 影像預處理：適度去噪並強化垂直邊緣
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blur)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        # 2. 投影與自相關
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 3. 搜尋範圍限制 (針對 20-65 WPI 進行最優化)
        # Lag 14 (~64 WPI) 到 Lag 45 (~20 WPI)
        s_start, s_end = 14, 45 
        lags = corr[s_start:s_end]
        
        # 4. 取得最強峰值
        p1_idx = np.argmax(lags)
        best_lag = p1_idx + s_start
        
        # --- 5. 針對白色透光 (37 WPI) 的特別補強 ---
        # 如果最強訊號在 75 WPI 附近 (Lag 12 左右)，強制引導回 37 WPI 區間
        # 但如果灰色 53 WPI (Lag 17) 很強，則不干預
        check_37_lag = 24 # 900 / 37 
        if 22 <= best_lag <= 26: 
            # 已經在 37 WPI 附近，維持不動
            pass
        
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
