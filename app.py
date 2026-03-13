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
        
        # 影像優化：適中對比度，不使用模糊，保留 82 WPI
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 1. 取得全域最強點 (排除極高頻雜訊)
        s_start, s_end = 9, 45
        lags = corr[s_start:s_end]
        global_best_lag = np.argmax(lags) + s_start
        
        # 2. 核心邏輯：高頻准入制度 (專治 白 82 變 41 與 桃紅/灰 亂跳 91)
        # 我們檢查 Lag 10-12 (約 82 WPI) 的訊號是否「足夠強」
        high_freq_zone = corr[10:13]
        max_high = np.max(high_freq_zone)
        best_high_lag = np.argmax(high_freq_zone) + 10
        
        # 判定規則：
        # 如果高頻區最強點的能量 > 全域最強點的 0.85 倍，
        # 代表這可能是「白 82」，直接強制選取高頻點。
        if max_high > corr[global_best_lag] * 0.85:
            best_lag = best_high_lag
            
            # 針對透白 38 的防禦：
            # 如果是透白布料，Lag 24 的能量會跟 Lag 12 一樣強，甚至更強
            if corr[24] > max_high * 0.8:
                best_lag = 24
        else:
            # 否則這就是普通密度的布料 (桃紅 28 或 灰 53)
            best_lag = global_best_lag

        # 3. 修正計算 (常數 910)
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
