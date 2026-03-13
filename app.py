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
        
        # 影像優化：使用中值濾波，這對去除導致 114 WPI 的細微雜訊最有效
        denoise = cv2.medianBlur(gray, 3)
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(denoise)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-450 : w//2+450]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # --- 核心優化：階梯式搜尋 ---
        # 定義高頻(>60 WPI) 與 中低頻(<60 WPI)
        # Lag 9-15 (對應 100-60 WPI) | Lag 16-50 (對應 56-18 WPI)
        high_lags = corr[9:16]
        low_lags = corr[16:51]
        
        h_best = np.argmax(high_lags) + 9
        l_best = np.argmax(low_lags) + 16
        
        # 決策邏輯：除非高頻能量「壓倒性」勝過低頻，否則不准跳高頻
        # 這能防止桃紅 28 變成 114
        if corr[h_best] > corr[l_best] * 1.3:
            best_lag = h_best
            # 針對透白布料：如果選中 80 WPI 附近，檢查兩倍距離
            check_lag = best_lag * 2
            if check_lag < 51 and corr[check_lag] > corr[best_lag] * 0.7:
                best_lag = check_lag
        else:
            best_lag = l_best

        # 計算 WPI (使用 910 穩定係數)
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
