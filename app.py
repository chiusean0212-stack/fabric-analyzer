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
        
        # 1. 預處理：使用 CLAHE 強化細微的 82 WPI 紗線，但不使用模糊
        enhanced = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4)).apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-450 : w//2+450]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 2. 定義三個搜尋點位 (對應您所有的布樣)
        # 高頻點 (82WPI -> Lag 11)
        # 中頻點 (53WPI -> Lag 17, 38WPI -> Lag 24)
        # 低頻點 (28WPI -> Lag 32)
        s_start, s_end = 9, 45
        lags = corr[s_start:s_end]
        
        # 3. 採用「高頻優先加強」
        # 我們給予高 WPI (小 Lag) 額外的能量加成，強迫白回歸 82
        weights = np.zeros_like(lags)
        for i in range(len(lags)):
            lag_val = i + s_start
            if lag_val < 15: weights[i] = 1.6 # 強力拉起 82 WPI
            elif lag_val < 20: weights[i] = 1.3 # 輔助 53 WPI
            else: weights[i] = 1.0 # 正常處理 38, 28 WPI
            
        weighted_lags = lags * weights
        best_idx = np.argmax(weighted_lags)
        best_lag = best_idx + s_start
        
        # 4. 針對「透白 75」的最後防線
        # 如果初步選中 75 WPI (Lag 12)，我們看 38 WPI (Lag 24) 的訊號
        # 只有當 38 WPI 處的訊號能量「明顯比 75 WPI 原始能量更強」時，才降頻
        if best_lag < 15:
            check_lag = best_lag * 2
            if check_lag < s_end:
                if corr[check_lag] > corr[best_lag] * 0.9: # 提高門檻，保護 82
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
