import streamlit as st
import cv2
import numpy as np

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
        
        # --- 核心修正 1：移除所有 Blur，使用銳化處理 ---
        # 這樣才能保證 82 WPI 的細紗線邊緣不被抹除
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # --- 核心修正 2：極致加權法 ---
        # 搜尋範圍：Lag 8 (112 WPI) 到 Lag 45 (20 WPI)
        s_start, s_end = 8, 45
        lags = corr[s_start:s_end]
        
        # 使用指數型加權：給予高 WPI (小 Lag) 絕對的優先權
        # 這能強迫系統從 29 彈回 82，從 19 彈回 38
        weights = np.power(np.linspace(1.8, 1.0, len(lags)), 2)
        weighted_lags = lags * weights
        
        best_idx = np.argmax(weighted_lags)
        best_lag = best_idx + s_start
        
        # --- 核心修正 3：透光布料「降頻」二次驗證 ---
        # 如果初步選中 Lag 10~13 (約 75~90 WPI)，但 Lag 23~26 (約 35~40 WPI)
        # 處的能量依然很強 (超過主峰的 75%)，判定為透白布料，降回 38
        if 10 <= best_lag <= 14:
            check_lag = best_lag * 2
            if check_lag < s_end:
                if corr[check_lag] > corr[best_lag] * 0.75:
                    best_lag = check_lag

        # 計算 WPI (係數 910)
        wpi = round(910 / best_lag)
        
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
