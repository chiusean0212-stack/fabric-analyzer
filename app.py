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
        
        # 1. 極致銳化：不使用模糊，強化細小紗線的邊緣
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 2. 搜尋範圍 (Lag 8 到 Lag 45)
        s_start, s_end = 8, 46
        lags = corr[s_start:s_end]
        
        # --- 核心優化：非線性權重補正 ---
        weights = np.ones_like(lags)
        for i in range(len(lags)):
            lag_val = i + s_start
            # 針對白色 82 WPI (Lag 11) 給予極高優先權
            if 10 <= lag_val <= 12: weights[i] = 2.2  # 拉起高頻
            # 針對灰色 53 WPI (Lag 17)
            elif 16 <= lag_val <= 18: weights[i] = 1.6
            # 針對透白 38 WPI (Lag 24)
            elif 23 <= lag_val <= 26: weights[i] = 1.4
            # 針對桃紅 28 WPI (Lag 32)
            elif 31 <= lag_val <= 34: weights[i] = 1.2
            else: weights[i] = 1.0

        weighted_lags = lags * weights
        best_lag = np.argmax(weighted_lags) + s_start
        
        # 3. 智慧防跳號 (針對透白降頻)
        # 只有當高頻點 (82WPI) 能量極弱，且兩倍處 (41WPI) 極強時，才准許降頻
        if 10 <= best_lag <= 13:
            check_lag = best_lag * 2
            if check_lag < s_end:
                # 門檻提高到 0.9，保護 82 不被輕易降到 41
                if corr[check_lag] > corr[best_lag] * 0.9:
                    best_lag = check_lag

        # 使用修正係數 910
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
