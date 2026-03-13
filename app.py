import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- 頁面設定 ---
st.set_page_config(page_title="廣笠機械 Goang Lih - AI Analysis", layout="centered")

# --- 密碼檢查 (維持 777) ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center;'>系統授權登入</h2>", unsafe_allow_html=True)
    pwd = st.text_input("請輸入密碼 (Password)", type="password")
    if st.button("登入"):
        if pwd == "777":
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("密碼錯誤")
    st.stop()

# --- 主介面標題與 Logo ---
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=100)
with col2:
    # 廣笠機械為藍色 (#1E3A8A)，Goang Lih 為紅色 (#FF0000)
    st.markdown("""
        <h1 style='margin-bottom: 0;'>
            <span style='color: #1E3A8A;'>廣笠機械</span> 
            <span style='color: #FF0000;'>Goang Lih</span>
        </h1>
        <p style='color: #666;'>AI Fabric Analysis System v2.0</p>
    """, unsafe_allow_html=True)

st.divider()

# --- 檔案上傳 ---
uploaded_file = st.file_uploader("📁 請上傳 FiberCatch 照片 (Upload Photo)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        # 讀取圖片
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 1. 預處理：溫和降噪與高對比 (解決黑色與米色平衡)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_blurred)
        
        h, w = enhanced.shape
        x_start = max(0, w // 2 - 450)
        x_end = min(w, x_start + 900)
        roi = enhanced[:, x_start:x_end]

        # 2. 梯度分析：強化三角形線圈特徵並減少直線雜訊
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # 3. 垂直投影
        projection = np.mean(grad_mag, axis=0).astype(np.float32)
        projection -= np.mean(projection)
        
        # 4. 自相關分析 (Autocorrelation)
        n = len(projection)
        corr = np.correlate(projection, projection, mode='full')[n-1:]
        
        # 5. 【倍頻鎖定與範圍優化】
        # 鎖定搜尋區間 Lag 15-65 (解決米色 22 變 12 的問題)
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

# 頁尾資訊
st.markdown("---")
st.caption("© 2026 廣笠機械 Goang Lih Machinery Co., Ltd. All rights reserved.")




