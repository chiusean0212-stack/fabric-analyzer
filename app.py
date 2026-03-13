import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

# --- 介面文字 ---
t_data = {
    "繁體中文": ["廣笠機械", "AI 分析系統", "📸 上傳照片", "結果", "授權登入"],
    "English": ["Goang Lih", "AI Analysis", "📸 Upload", "Result", "Login"]
}
sel = st.sidebar.selectbox("Lang", ["繁體中文", "English"])
t = t_data[sel]

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
        
        # 使用輕微去噪，確保 82 WPI 不失真
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(blur)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-450 : w//2+450]
        
        # 提取垂直邊緣
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 搜尋範圍 (Lag 8=112WPI 到 Lag 65=14WPI)
        s_start, s_end = 8, 65
        lags = corr[s_start:s_end]
        
        # 取得初步最強峰值
        p1_idx = np.argmax(lags)
        best_lag = p1_idx + s_start
        
        # --- 核心修正：透光布料倍頻壓制邏輯 ---
        # 如果初步結果 > 70 WPI (即 Lag < 12.8)
        if best_lag < 13:
            # 檢查兩倍距離的 Lag (對應約 35-38 WPI)
            check_lag = best_lag * 2
            if check_lag < s_end:
                # 如果兩倍距離處的訊號強度達到最強者的 75% 以上
                # 代表原本的細訊號是「紗線+空隙」的誤判，應取較寬的兩倍距離
                if corr[check_lag] > lags[p1_idx] * 0.75:
                    best_lag = check_lag

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
