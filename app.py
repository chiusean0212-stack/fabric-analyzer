import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- 頁面設定 ---
st.set_page_config(page_title="廣笠機械 Goang Lih - AI Analysis", layout="centered")

# --- 語系字典 (Dictionary) ---
lang_dict = {
    "繁體中文": {
        "login_title": "系統授權登入",
        "pwd_placeholder": "請輸入密碼",
        "login_btn": "進入系統",
        "login_err": "密碼驗證失敗，請重新輸入",
        "app_subtitle": "AI 分析系統",
        "upload_label": "📸 點擊或拖曳上傳 FiberCatch 照片",
        "result_title": "偵測結果",
        "footer_motto": "智慧針織 · 領先未來",
        "footer_sub": "專業針織機械製造與 AI 數位化解決方案",
        "analyzing": "分析中...",
        "err_msg": "分析發生錯誤"
    },
    "English": {
        "login_title": "System Authorization",
        "pwd_placeholder": "Enter Password",
        "login_btn": "Login",
        "login_err": "Authentication failed. Please try again.",
        "app_subtitle": "AI Analysis System",
        "upload_label": "📸 Click or Drag to Upload FiberCatch Photo",
        "result_title": "Analysis Result",
        "footer_motto": "Smart Knitting · Leading Future",
        "footer_sub": "Professional Machinery & AI Digital Solutions",
        "analyzing": "Analyzing...",
        "err_msg": "Analysis Error"
    }
}

# --- 語言切換選擇器 (放置於側邊欄) ---
with st.sidebar:
    st.markdown("### 🌐 Language / 語言")
    selected_lang = st.selectbox("Select Language", ["繁體中文", "English"])
    texts = lang_dict[selected_lang]

# --- 自定義 CSS ---
st.markdown(f"""
    <style>
    .stFileUploader label p {{
        font-size: 24px !important;
        font-weight: bold !important;
        color: #1E3A8A !important;
    }}
    .stFileUploader section {{
        border: 2px dashed #1E3A8A !important;
        border-radius: 15px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- 專業版登入介面 ---
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

# --- 主介面標題 (登入後) ---
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

# --- 檔案上傳 ---
uploaded_file = st.file_uploader(texts['upload_label'], type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        with st.spinner(texts['analyzing']):
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # 核心演算法
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img_blurred)
            
            h, w = enhanced.shape
            x_start = max(0, w // 2 - 450)
            x_end = min(w, x_start + 900)
            roi = enhanced[:, x_start:x_end]

            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            
            projection = np.mean(grad_mag, axis=0).astype(np.float32)
            projection -= np.mean(projection)
            
            n = len(projection)
            corr = np.correlate(projection, projection, mode='full')[n-1:]
            
            search_start, search_end = 15, 65 
            lags = corr[search_start:search_end]
            best_lag = np.argmax(lags) + search_start
            wpi_result = round(900 / best_lag)
            
            # 顯示結果
            st.image(uploaded_file, caption=selected_lang, use_container_width=True)
            
            st.markdown(f"""
                <div style="text-align: center; background-color: #f0f2f6; padding: 25px; border-radius: 15px; border: 2px solid #1E3A8A; margin-top: 20px;">
                    <h2 style="color: #1E3A8A; margin-top: 0;">{texts['result_title']}</h2>
                    <span style="font-size: 85px; font-weight: bold; color: #FF0000;">WPI = {wpi_result}</span>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"{texts['err_msg']}: {e}")

# --- 頁尾資訊 ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(f"""
    <div style="text-align: center; color: #666;">
        <p style="font-size: 18px; font-weight: bold; color: #1E3A8A;">{texts['footer_motto']}</p>
        <p style="font-size: 14px;">{texts['footer_sub']}</p>
        <p style="font-size: 12px; margin-top: 10px;">© 2026 Goang Lih Machinery Co., Ltd. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
