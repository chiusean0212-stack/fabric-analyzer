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
    with col2:
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
        
        # 混合濾波：先用中值濾波去噪，再用雙邊濾波保留邊緣
        denoise = cv2.medianBlur(gray, 3)
        clean = cv2.bilateralFilter(denoise, 5, 50, 50)
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(clean)
        
        h, w = enhanced.shape
        roi = enhanced[:, max(0, w//2-450) : min(w, w//2+450)]
        
        # 提取垂直邊緣
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 搜尋區間 (對應約 13 - 90 WPI)
        s_start, s_end = 10, 70
        lags = corr[s_start:s_end]
        
        # 基礎最強峰值
        p1_idx = np.argmax(lags)
        best_lag = p1_idx + s_start
        
        # --- 核心：防跳號驗證邏輯 ---
        # 如果抓到的 Lag 太小 (WPI 太大，例如 75)，檢查它的 2 倍 Lag (也就是 1/2 WPI，例如 37)
        check_lag = best_lag * 2
        if check_lag < s_end:
            # 如果兩倍距離的地方也有一個夠強的峰值 (強度達到最強者的 60% 以上)
            # 說明兩倍距離的才是真正的紗線間距
            if corr[check_lag] > lags[p1_idx] * 0.6:
                best_lag = check_lag

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
