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
        "login_title": "系統授權登入",
        "pwd_placeholder": "請輸入密碼",
        "login_btn": "進入系統",
        "login_err": "密碼驗證失敗，請重新輸入",
        "app_subtitle": "AI 分析系統",
        "upload_label": "📸 點擊或拖曳上傳照片 (支援白色/透明/精細布樣優化)",
        "result_title": "偵測結果",
        "footer_motto": "智慧針織 · 領先未來",
        "footer_sub": "專業針織機械製造與 AI 數位化解決方案",
        "analyzing": "深度精細掃描中...",
        "err_msg": "分析發生錯誤"
    },
    "English": {
        "login_title": "System Authorization",
        "pwd_placeholder": "Enter Password",
        "login_btn": "Login",
        "login_err": "Authentication failed. Please try again.",
        "app_subtitle": "AI Analysis System",
        "upload_label": "📸 Click or Drag to Upload (Optimized for White/Transparent Fabric)",
        "result_title": "Analysis Result",
        "footer_motto": "Smart Knitting · Leading Future",
        "footer_sub": "Professional Machinery & AI Digital Solutions",
        "analyzing": "Analyzing...",
        "err_msg": "Analysis Error"
    }
}

# --- 3. 語言切換選擇器 (Sidebar) ---
with st.sidebar:
    st.markdown("### 🌐 Language / 語言")
    selected_lang = st.selectbox("Select Language", ["繁體中文", "English"])
    texts = lang_dict[selected_lang]

# --- 4. 自定義 CSS (美化介面與放大字體) ---
st.markdown(f"""
    <style>
    .stFileUploader label p {{
        font-size: 24px !important;
        font-weight: bold !important;
        color: #1E3A8A !important;
        margin-bottom: 10px;
    }}
    .stFileUploader section {{
        border: 2px dashed #1E3A8A !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }}
    div.stButton > button:first-child {{
        background-color: #1E3A8A;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# --- 5. 專業版登入介面 ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, login_col, _ = st.columns([1, 2, 1])
    with login_col:
        st.markdown(f"""
            <div style="text-align: center; background-color: white; padding: 30px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: 1px solid #eee;">
                <h1 style="color: #1E3A8A; margin-bottom: 5px;">廣笠機械</h1>
                <h2 style="color: #FF0000; margin-top: 0; font-size: 24px;">Goang Lih</h2>
                <hr style="border: 0.5px solid #eee;">
                <p style="color: #666; font-size: 16px;">{texts['login_title']}</p>
            </div>
        """, unsafe_allow_html=True)
        with st.form("login_form"):
            pwd = st.text_input(texts['login_title'], type="password", placeholder=texts['pwd_placeholder'], label_visibility="collapsed")
            submit = st.form_submit_button(texts['login_btn'], use_container_width=True)
            if submit:
                if pwd == "777":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error(texts['login_err'])
    st.stop()

# --- 6. 主介面標題與 Logo ---
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=100)
with col2:
    st.markdown(f"""
        <h1 style='margin-bottom: 0;'>
            <span style='color: #1E3A8A;'>廣笠機械</span> 
            <span style='color: #FF0000;'>Goang Lih</span>
        </h1>
        <h3 style='margin-top: 0; color: #333;'>{texts['app_subtitle']}</h3>
    """, unsafe_allow_html=True)

st.divider()

# --- 7. 檔案上傳與核心分析 ---
uploaded_file = st.file_uploader(texts['upload_label'], type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        with st.spinner(texts['analyzing']):
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # --- 核心優化演算法 ---
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 雙邊濾波保留邊緣並減少透明背景雜訊
            img_clean = cv2.bilateralFilter(img_gray, 9, 75, 75)
            
            # 強力對比
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img_clean)
            
            h, w = enhanced.shape
            x_start = max(0, w // 2 - 450)
            x_end = min(w, x_start + 900)
            roi = enhanced[:, x_start:x_end]

            # 專注垂直邊緣，減少橫向干擾
            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=5)
            grad_x = np.absolute(grad_x)
            
            projection = np.mean(grad_x, axis=0).astype(np.float32)
            projection -= np.mean(projection)
            
            n = len(projection)
            corr = np.correlate(projection, projection, mode='full')[n-1:]
            
            # 搜尋區間 (10 - 112 WPI)
            search_start, search_end = 8, 90 
            lags = corr[search_start:search_end]
            
            # 取得主要峰值
            p1_idx = np.argmax(lags)
            best_lag = p1_idx + search_start
            
            # 倍頻檢查邏輯 (解決 19 變 37 問題)
            half_lag = int(best_lag / 2)
            if half_lag >= search_start:
                half_idx = half_lag - search_start
                # 若半頻位置訊號強度夠，則選取更細的間距
                if lags[half_idx] > lags[p1_idx] * 0.7:
                    best_lag = half_lag

            wpi_result = round(900 / best_lag)
            
            # --- 顯示結果 ---
            st.image(uploaded_file, caption=selected_lang, use_container_width=True)
            st.markdown(f"""
                <div style="text-align: center;
