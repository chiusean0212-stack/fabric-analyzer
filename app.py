import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="廣笠機械 Goang Lih - AI Analysis", layout="centered")

# --- 2. 語系字典設定 ---
lang_dict = {
    "繁體中文": {
        "login_title": "系統授權登入", "pwd_placeholder": "請輸入密碼", "login_btn": "進入系統",
        "login_err": "密碼驗證失敗", "app_subtitle": "AI 分析系統",
        "upload_label": "📸 點擊或拖曳上傳照片 (支援白色/透明/精細布樣優化)",
        "result_title": "偵測結果", "footer_motto": "智慧針織 · 領新未來",
        "footer_sub": "專業針織機械製造與 AI 數位化解決方案", "analyzing": "掃描中...", "err_msg": "分析發生錯誤"
    },
    "English": {
        "login_title": "System Authorization", "pwd_placeholder": "Enter Password", "login_btn": "Login",
        "login_err": "Authentication failed", "app_subtitle": "AI Analysis System",
        "upload_label": "📸 Click or Drag to Upload (Optimized for White/Transparent Fabric)",
        "result_title": "Analysis Result", "footer_motto": "Smart Knitting · Leading Future",
        "footer_sub": "Professional Machinery & AI Digital Solutions", "analyzing": "Analyzing...", "err_msg": "Analysis Error"
    }
}

# --- 3. 語言切換 (Sidebar) ---
with st.sidebar:
    st.markdown("### 🌐 Language")
    selected_lang = st.selectbox("Select Language", ["繁體中文", "English"])
    texts = lang_dict[selected_lang]

# --- 4. CSS 美化 ---
st.markdown("""<style>
    .stFileUploader label p { font-size: 24px !important; font-weight: bold !important; color: #1E3A8A !important; }
    .stFileUploader section { border: 2px dashed #1E3A8A !important; border-radius: 15px !important; }
    div.stButton > button:first-child { background-color: #1E3A8A; color: white; border-radius: 10px; }
</style>""", unsafe_allow_html=True)

# --- 5. 登入介面 ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, login_col, _ = st.columns([1, 2, 1])
    with login_col:
        st.markdown(f"<div style='text-align:center;'><h1>廣笠機械</h1><h2>Goang Lih</h2><p>{texts['login_title']}</p></div>", unsafe_allow_html=True)
        with st.form("login_form"):
            pwd = st.text_input("PWD", type="password", placeholder=texts['pwd_placeholder'], label_visibility="collapsed")
            if st.form_submit_button(texts['login_btn'], use_container_width=True):
                if pwd == "777":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error(texts['login_err'])
    st.stop()

# --- 6. 主介面標題 ---
c1, c2 = st.columns([1, 4])
with c1:
    if os.path.exists("LOGO.png"): st.image("LOGO.png", width=100)
with c2:
    st.markdown(f"<h1><span style='color:#1E3A8A;'>廣笠機械</span> <span style='color:#FF0000;'>Goang Lih</span></h1><h3>{texts['app_subtitle']}</h3>", unsafe_allow_html=True)

st.divider()

# --- 7. 分析核心 ---
uploaded_file = st.file_uploader(texts['upload_label'], type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        with st.spinner(texts['analyzing']):
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clean = cv2.bilateralFilter(gray, 9, 75, 75)
            enhanced = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8)).apply(clean)
            
            h, w = enhanced.shape
            roi = enhanced[:, max(0, w//2-450) : min(w, w//2+450)]
            grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=5))
            proj = np.mean(grad_x, axis=0).astype(np.float32)
            proj -= np.mean(proj)
            
            n = len(proj)
            corr = np.correlate(proj, proj, mode='full')[n-1:]
            lags = corr[8:90]
            
            p1_idx = np.argmax(lags)
            best_lag = p1_idx + 8
            
            # 37 WPI vs 19 WPI 修正邏輯
            half_lag = int(best_lag / 2)
            if half_lag >= 8:
                if lags[half_lag - 8] > lags[p1_idx] * 0.7:
                    best_lag = half_lag

            wpi_res = round(900 / best_lag)
            
            st.image(uploaded_file, use_container_width=True)
            st.markdown(f"<div style='text-align:center;background:#f0f2f6;padding:20px;border-radius:15px;border:2px solid #1E3A8A;margin-top:20px;'><h2 style='color:#1E3A8A;'>{texts['result_title']}</h2><span style='font-size:80px;font-weight:bold;color:#FF0000;'>WPI = {wpi_res}</span></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"{texts['err_msg']}: {e}")

st.markdown("<br><br><hr><div style='text-align:center;color:#666;'><p style='font-weight:bold;'>"+texts['footer_motto']+"</p><p>© 2026 Goang Lih Machinery Co., Ltd.</p></div>", unsafe_allow_html=True)
