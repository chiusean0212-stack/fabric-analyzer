import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

# --- 介面文字與授權 ---
t_data = {"繁體中文": ["廣笠機械", "AI 分析系統", "📸 上傳照片", "結果", "授權登入"],
          "English": ["Goang Lih", "AI Analysis", "📸 Upload", "Result", "Login"]}
sel = st.sidebar.selectbox("Lang", ["繁體中文", "English"])
t = t_data[sel]

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    pwd = st.text_input(t[4], type="password")
    if st.button(t[4]):
        if pwd == "777": st.session_state["auth"] = True; st.rerun()
    st.stop()

st.title(f"{t[0]} Goang Lih")

# --- 核心演算：低通頻率強化 ---
up = st.file_uploader(t[2], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        raw_bytes = np.frombuffer(up.read(), np.uint8)
        img = cv2.imdecode(raw_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 關鍵修正：針對透白布料使用較大的中值濾波
        # 這能有效抹除導致 75 WPI 的細微空隙雜訊，保留 38 WPI 的主紗線
        denoise = cv2.medianBlur(gray, 5) 
        
        # 2. 強化結構對比
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoise)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-450 : w//2+450]
        
        # 3. 垂直梯度投影
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        
        # 4. 自相關分析
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 5. 搜尋範圍 (Lag 10 ~ 65)
        # 起點設為 10 (對應 90 WPI)，封殺 100 WPI 以上的超高頻錯誤
        s_start, s_end = 10, 65
        lags = corr[s_start:s_end]
        
        # 6. 加入距離加權 (越寬的間距賦予越高權重，防止倍頻錯誤)
        weights = np.linspace(1.0, 1.4, len(lags))
        best_lag = np.argmax(lags * weights) + s_start
        
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
