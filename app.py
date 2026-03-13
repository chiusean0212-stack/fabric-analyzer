import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. 基本設定與語系 ---
st.set_page_config(page_title="Goang Lih AI", layout="centered")

lang = {
    "繁體中文": {
        "title": "廣笠機械", "sub": "AI 分析系統", "upload": "📸 上傳照片 (全布樣優化版)",
        "res": "偵測結果", "login": "授權登入", "err": "驗證失敗"
    },
    "English": {
        "title": "Goang Lih", "sub": "AI Analysis", "upload": "📸 Upload Photo (All-Fabric Mode)",
        "res": "Result", "login": "Login", "err": "Failed"
    }
}

with st.sidebar:
    sel = st.selectbox("Language", ["繁體中文", "English"])
    t = lang[sel]

# --- 2. 登入邏輯 ---
if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title(t["title"])
        with st.form("l"):
            pwd = st.text_input(t["login"], type="password")
            if st.form_submit_button(t["login"]):
                if pwd == "777": st.session_state["auth"] = True; st.rerun()
                else: st.error(t["err"])
    st.stop()

# --- 3. 主介面標題 ---
c1, c2 = st.columns([1, 4])
with c1:
    if os.path.exists("LOGO.png"): st.image("LOGO.png", width=80)
with c2:
    st.markdown(f"<h1><span style='color:#1E3A8A;'>{t['title']}</span> <span style='color:#FF0000;'>Goang Lih</span></h1>", unsafe_allow_html=True)
    st.write(f"### {t['sub']}")

# --- 4. 核心演算法 (平衡版：紅28、灰53、白37) ---
up = st.file_uploader(t["upload"], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        with st.spinner("..."):
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 平滑處理與對比增強
            clean = cv2.bilateralFilter(gray, 7, 50, 50)
            enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(clean)
            
            h, w = enhanced.shape
            roi = enhanced[:, max(0, w//2-450) : min(w, w//2+450)]
            grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
            proj = np.mean(grad_x, axis=0).astype(np.float32)
            proj -= np.mean(proj)
            
            n = len(proj)
            corr = np.correlate(proj, proj, mode='full')[n-1:]
            
            # 搜尋區間 (8=112WPI, 90=10WPI)
            lags = corr[8:90]
            p1_idx = np.argmax(lags)
            best_lag = p1_idx + 8
            
            # --- 關鍵：動態門檻判別 ---
            # 只有當細訊號極度明顯時才切換，防止紅/灰布樣數值加倍
            half_lag = int(best_lag / 2)
            if half_lag >= 8:
                if lags[half_lag - 8] > lags[p1_idx] * 0.95: # 提高到 0.95 確保穩定性
                    best_lag = half_lag

            wpi = round(900 / best_lag)
            
            st.image(up, use_container_width=True)
            st.markdown(f"""
                <div style='text-align:center; background:#f0f2f6; padding:20px; border-radius:15px; border:2px solid #1E3A8A;'>
                    <h2 style='color:#1E3A8A;'>{t['res']}</h2>
                    <p style='font-size:80px; font-weight:bold; color:#FF0000; margin:0;'>WPI = {wpi}</p>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("© 2026 Goang Lih Machinery Co., Ltd. All rights reserved.")
