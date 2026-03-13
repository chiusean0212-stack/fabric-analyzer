import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 頁面設定 ---
st.set_page_config(page_title="廣笠機械 Goang Lih - AI Analysis", layout="centered")

st.title("廣笠機械 Goang Lih")
st.subheader("AI 布料密度分析系統 (Cloud Version)")

# --- 密碼檢查 ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    pwd = st.text_input("請輸入授權密碼 (Password)", type="password")
    if st.button("登入"):
        if pwd == "777":
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("密碼錯誤")
    st.stop()

# --- 主程式介面 ---
uploaded_file = st.file_uploader("請上傳 FiberCatch 照片", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        # 讀取圖片
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 1. 預處理：溫和降噪與高對比 (維持黑色與米色的平衡)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_blurred)
        
        h, w = enhanced.shape
        # 固定抓取中央 900 像素區域
        x_start = max(0, w // 2 - 450)
        x_end = min(w, x_start + 900)
        roi = enhanced[:, x_start:x_end]

        # 2. 梯度分析：支援三角形線圈與消除直線雜訊
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # 3. 垂直投影
        projection = np.mean(grad_mag, axis=0).astype(np.float32)
        projection -= np.mean(projection)
        
        # 4. 自相關分析 (Autocorrelation)
        n = len(projection)
        corr = np.correlate(projection, projection, mode='full')[n-1:]
        
        # --- 5. 【核心鎖定邏輯：防止 22 變 12】 ---
        # 限制搜尋範圍在 Lag 15-65 像素 (對應 WPI 14-60)
        # 這樣就不會去抓到兩倍大的間隔
        search_start, search_end = 15, 65 
        lags = corr[search_start:search_end]
        
        best_lag = np.argmax(lags) + search_start
        wpi_result = round(900 / best_lag)
        
        # --- 顯示結果 ---
        st.image(uploaded_file, caption="已上傳的照片", use_container_width=True)
        st.markdown(f"""
            <div style="text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <h1 style="color: #1E3A8A; margin-bottom: 0;">偵測結果 (Result)</h1>
                <p style="font-size: 64px; font-weight: bold; color: #FF0000; margin-top: 10px;">WPI = {wpi_result}</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"分析發生錯誤: {e}")




