import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- 頁面設定 ---
st.set_page_config(page_title="廣笠機械 Goang Lih - AI Analysis", layout="centered")

# --- 專業版登入介面 ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    # 建立一個置中的容器
    st.markdown("<br><br>", unsafe_allow_html=True)
    login_col_1, login_col_2, login_col_3 = st.columns([1, 2, 1])
    
    with login_col_2:
        st.markdown("""
            <div style="text-align: center; background-color: white; padding: 30px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: 1px solid #eee;">
                <h1 style="color: #1E3A8A; margin-bottom: 5px;">廣笠機械</h1>
                <h2 style="color: #FF0000; margin-top: 0; font-size: 24px;">Goang Lih</h2>
                <hr style="border: 0.5px solid #eee;">
                <p style="color: #666; font-size: 16px;">AI 分析系統 - 授權登入</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 登入表單容器
        with st.form("login_form"):
            pwd = st.text_input("請輸入系統授權密碼", type="password", placeholder="Password")
            submit = st.form_submit_button("進入系統", use_container_width=True)
            if submit:
                if pwd == "777":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("密碼驗證失敗，請重新輸入")
                    
        st.markdown("<p style='text-align: center; color: #999; font-size: 12px;'>© 2026 Goang Lih Machinery Co., Ltd.</p>", unsafe_allow_html=True)
    st.stop()

# --- 主介面標題與 Logo (登入後顯示) ---
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=100)
with col2:
    st.markdown("""
        <h1 style='margin-bottom: 0;'>
            <span style='color: #1E3A8A;'>廣笠機械</span> 
            <span style='color: #FF0000;'>Goang Lih</span>
        </h1>
        <h3 style='margin-top: 0; color: #333;'>AI 分析系統</h3>
    """, unsafe_allow_html=True)

st.divider()

# --- 檔案上傳 ---
uploaded_file = st.file_uploader("📁 請上傳 FiberCatch 照片 (Upload Photo)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        # 讀取圖片
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 1. 預處理：維持溫和降噪與高對比
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_blurred)
        
        h, w = enhanced.shape
        x_start = max(0, w // 2 - 450)
        x_end = min(w, x_start + 900)
        roi = enhanced[:, x_start:x_end]

        # 2. 梯度分析
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # 3. 垂直投影
        projection = np.mean(grad_mag, axis=0).astype(np.float32)
        projection -= np.mean(projection)
        
        # 4. 自相關分析
        n = len(projection)
        corr = np.correlate(projection, projection, mode='full')[n-1:]
        
        # 5. 倍頻鎖定與範圍優化
        search_start, search_end = 15, 65 
        lags = corr[search_start:search_end]
        best_lag = np.argmax(lags) + search_start
        wpi_result = round(900 / best_lag)
        
        # --- 顯示圖片與結果 ---
        st.image(uploaded_file, caption="分析目標照片", use_container_width=True)
        
        st.markdown(f"""
            <div style="text-align: center; background-color: #f0f2f6; padding: 25px; border-radius: 15px; border: 2px solid #1E3A8A;">
                <h2 style="color: #1E3A8A; margin-top: 0;">偵測結果 (Result)</h2>
                <span style="font-size: 80px; font-weight: bold; color: #FF0000;">WPI = {wpi_result}</span>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"分析發生錯誤: {e}")

# --- 頁尾資訊 ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p style="font-size: 18px; font-weight: bold; color: #1E3A8A;">智慧針織 · 領先未來</p>
        <p style="font-size: 14px;">專業針織機械製造與 AI 數位化解決方案</p>
        <p style="font-size: 12px; margin-top: 10px;">© 2026 廣笠機械 Goang Lih Machinery Co., Ltd. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
