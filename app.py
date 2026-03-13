import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(page_title="Goang Lih AI", layout="centered")

texts = {
    "繁體中文": {"t": "廣笠機械", "s": "AI 分析系統", "u": "📸 上傳布樣照片", "r": "偵測結果", "l": "授權登入"},
    "English": {"t": "Goang Lih", "s": "AI Analysis", "u": "📸 Upload Photo", "r": "Result", "l": "Login"}
}

with st.sidebar:
    lang = st.selectbox("Language", ["繁體中文", "English"])
    t = texts[lang]

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.title(t["t"])
        pwd = st.text_input(t["l"], type="password")
        if st.button(t["l"]):
            if pwd == "777": st.session_state["auth"] = True; st.rerun()
    st.stop()

col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("LOGO.png"): st.image("LOGO.png", width=80)
with col2:
    st.markdown(f"<h1><span style='color:#1E3A8A;'>{t['t']}</span> <span style='color:#FF0000;'>Goang Lih</span></h1>", unsafe_allow_html=True)
    st.write(f"### {t['s']}")

up = st.file_uploader(t["u"], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        raw = np.frombuffer(up.read(), np.uint8)
        img = cv2.imdecode(raw, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 針對白色透光調整：降低濾波強度避免抹平訊號，提高對比強化細線
        clean = cv2.bilateralFilter(gray, 5, 40, 40)
        enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(clean)
        
        h, w = enhanced.shape
        roi = enhanced[:, max(0, w//2-450) : min(w, w//2+450)]
        
        # 提取垂直邊緣
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # --- 重新校準搜尋範圍 (10 - 80 WPI) ---
        # 11 對應 81 WPI, 65 對應 14 WPI
        s_start, s_end = 11, 65
        lags = corr[s_start:s_end]
        
        # 訊號增強：對較細的間距（高WPI區段）給予輕微權重補償
        weights = np.linspace(1.1, 1.0, len(lags))
        weighted_lags = lags * weights
        
        best_lag = np.argmax(weighted_lags) + s_start
        wpi = round(900 / best_lag)
        
        st.image(up, use_container_width=True)
        st.markdown(f"""
            <div style='text-align:center; background:#f0f2f6; padding:20px; border-radius:15px; border:2px solid #1E3A8A;'>
                <h2 style='color:#1E3A8A;'>{t['r']}</h2>
                <p style='font-size:80px; font-weight:bold; color:#FF0000; margin:0;'>WPI = {wpi}</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("© 2026 Goang Lih Machinery Co., Ltd.")
