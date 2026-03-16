import streamlit as st
import cv2
import numpy as np
import os

# 頁面配置
st.set_page_config(page_title="廣笠機械 Goang Lih", layout="centered", page_icon="⚙️")

# 自定義 CSS：仿照照片中的簡潔白底風格
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
        transition: 0.3s;
    }
    .login-box {
        padding: 30px;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background-color: #f8fafc;
    }
    .title-text {
        color: #1e3a8a;
        font-family: "Microsoft JhengHei", sans-serif;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .subtitle-text {
        color: #64748b;
        font-size: 1.1em;
        margin-top: 0px;
    }
    </style>
""", unsafe_allow_html=True)

# 登入邏輯
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    # 登入介面排版
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # 嘗試讀取資料夾內的 LOGO.png
        if os.path.exists("LOGO.png"):
            st.image("LOGO.png", width=120)
        else:
            # 如果找不到檔案，顯示一個佔位圖示
            st.write("⚙️ LOGO")
            
    with col2:
        st.markdown("<h1 class='title-text'>廣笠機械 <span style='color:#dc2626;'>Goang Lih</span></h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle-text'>數位布鏡 - 自動 WPI 分析系統</p>", unsafe_allow_html=True)

    st.write("---")
    
    # 登入框
    with st.container():
        left, mid, right = st.columns([1, 2, 1])
        with mid:
            st.markdown("<div class='login-box'>", unsafe_allow_html=True)
            pwd = st.text_input("授權密碼 (Password)", type="password")
            if st.button("系統登入"):
                if pwd == "777":
                    st.session_state["auth"] = True
                    st.rerun()
                else:
                    st.error("密碼錯誤，請重新輸入")
            st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- 登入後的分析介面 (仿照照片排版) ---

# 標頭區
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=100)
with col2:
    st.markdown("<h2 style='margin-bottom:0;'>廣笠機械 <span style='color:#dc2626;'>Goang Lih</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray;'>數位布鏡 - 自動 WPI 分析系統</p>", unsafe_allow_html=True)

st.write("---")

# 上傳區
st.markdown("### 📥 上傳布樣")
st.markdown("<p style='font-size:0.8em; color:gray;'>選擇照片 (建議解析度：1英吋 = 900像素)</p>", unsafe_allow_html=True)

up = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if up:
    try:
        # 影像讀取與處理 (沿用最穩定的分析邏輯)
        img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 雙邊濾波防雜訊
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
        
        # 判定邏輯
        high_zone = corr[10:15]
        low_zone = corr[15:51]
        max_high, max_low = np.max(high_zone), np.max(low_zone)
        high_avg = np.mean(high_zone)
        is_high_pure = (max_high > high_avg * 1.5)
        
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
        elif 50 <= raw_wpi <= 58: final_wpi = 53
        elif 44 <= raw_wpi <= 49: final_wpi = 47
        elif 37 <= raw_wpi <= 41: final_wpi = 38
        elif 34 <= raw_wpi <= 36.9: final_wpi = 36
        elif 27 <= raw_wpi <= 31: final_wpi = 28
        elif 23 <= raw_wpi <= 26.9: final_wpi = 24
        elif 19 <= raw_wpi <= 22: final_wpi = 21
        else: final_wpi = round(raw_wpi)

        # 顯示結果
        st.image(img_bgr, use_container_width=True)
        st.markdown(f"""
            <div style='text-align:center; background:#ffffff; padding:20px; border:2px solid #1e3a8a; border-radius:15px;'>
                <h3 style='margin:0; color:#1e3a8a;'>分析結果</h3>
                <p style='font-size:80px; font-weight:bold; color:#ef4444; margin:0;'>{final_wpi}</p>
                <p style='color:gray;'>WPI (計算值: {raw_wpi:.1f})</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"分析失敗: {e}")

# 頁尾 (參照照片底部的文字)
st.write("---")
st.markdown("<p style='text-align:center; color:silver; font-size:0.7em;'>© 2026 廣笠機械 Goang Lih | 專業針織機械製造 | AI 數位轉型專案</p>", unsafe_allow_html=True)
