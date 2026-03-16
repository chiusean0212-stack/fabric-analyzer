import streamlit as st
import cv2
import numpy as np
import os

# 頁面配置
st.set_page_config(page_title="廣笠機械 Goang Lih", layout="centered", page_icon="⚙️")

# --- 語言包設定 ---
languages = {
    "繁體中文": {
        "company": "廣笠機械",
        "system": "數位布鏡 - 自動 WPI 分析系統",
        "login_title": "系統授權登入",
        "password": "授權密碼 (Password)",
        "login_btn": "系統登入",
        "upload_title": "📥 上傳布樣",
        "upload_hint": "選擇照片 (建議解析度：1英吋 = 900像素)",
        "result_title": "分析結果",
        "calc_ref": "精確計算參考",
        "error_pwd": "密碼錯誤，請重新輸入",
        "footer": "© 2026 廣笠機械 Goang Lih | 專業針織機械製造 | AI 數位轉型專案"
    },
    "English": {
        "company": "Goang Lih",
        "system": "Digital Fabric Scope - Auto WPI System",
        "login_title": "System Authorization",
        "password": "Password",
        "login_btn": "Login",
        "upload_title": "📥 Upload Fabric",
        "upload_hint": "Select Photo (Recommended: 1 inch = 900px)",
        "result_title": "Analysis Result",
        "calc_ref": "Raw Calculation",
        "error_pwd": "Wrong password, please try again",
        "footer": "© 2026 Goang Lih Machinery | Professional Knitting Machinery | AI Transformation Project"
    }
}

# --- 頁面頂部：中英文切換按鈕 ---
# 使用 st.radio 並橫向排列，放在最上方
lang_choice = st.radio("Language / 語言選擇", ["繁體中文", "English"], horizontal=True, label_visibility="visible")
txt = languages[lang_choice]

# 自定義 CSS
st.markdown(f"""
    <style>
    .title-text {{
        color: #1e3a8a;
        font-family: "Microsoft JhengHei", sans-serif;
        font-weight: bold;
        margin-bottom: 0px;
    }}
    .subtitle-text {{
        color: #64748b;
        font-size: 1.1em;
        margin-top: 0px;
    }}
    .login-box {{
        padding: 30px;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background-color: #f8fafc;
        text-align: center;
    }}
    </style>
""", unsafe_allow_html=True)

# 登入邏輯
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    # 標題排版 (仿照片)
    col1, col2 = st.columns([1, 3])
    with col1:
        if os.path.exists("LOGO.png"):
            st.image("LOGO.png", width=120)
        else:
            st.write("⚙️ LOGO")
    with col2:
        st.markdown(f"<h1 class='title-text'>{txt['company']} <span style='color:#dc2626;'>Goang Lih</span></h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='subtitle-text'>{txt['system']}</p>", unsafe_allow_html=True)

    st.write("---")
    
    # 登入框
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(f"<div class='login-box'><h3>{txt['login_title']}</h3>", unsafe_allow_html=True)
        pwd = st.text_input(txt['password'], type="password")
        if st.button(txt['login_btn']):
            if pwd == "777":
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error(txt['error_pwd'])
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- 登入後的介面 ---
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=100)
with col2:
    st.markdown(f"<h2 style='margin-bottom:0;'>{txt['company']} <span style='color:#dc2626;'>Goang Lih</span></h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:gray;'>{txt['system']}</p>", unsafe_allow_html=True)

st.write("---")

st.markdown(f"### {txt['upload_title']}")
st.markdown(f"<p style='font-size:0.8em; color:gray;'>{txt['upload_hint']}</p>", unsafe_allow_html=True)

up = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if up:
    try:
        # 分析核心邏輯 (延續穩定版)
        img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 判定
        high_zone, low_zone = corr[10:15], corr[15:51]
        max_high, max_low = np.max(high_zone), np.max(low_zone)
        is_high_pure = (max_high > np.mean(high_zone) * 1.5)
        
        if max_high > max_low * 0.75 and is_high_pure:
            best_lag = np.argmax(high_zone) + 10
        else:
            s_start = 15
            lags = corr[s_start:51]
            weights = np.ones_like(lags)
            for i in range(len(lags)):
                l_val = i + s_start
                if 16 <= l_val <= 19: weights[i] = 1.3
                if 31 <= l_val <= 45: weights[i] = 1.2
            best_lag = np.argmax(lags * weights) + s_start
        
        raw_wpi = 925 / best_lag
        
        # 歸位矩陣
        if 78 <= raw_wpi <= 95: final_wpi = 83
        elif 50 <= raw_wpi <= 58:
