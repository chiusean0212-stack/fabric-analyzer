import streamlit as st
import cv2
import numpy as np

# 1. 網頁頁籤標題
st.set_page_config(page_title="廣笠機械 Goang Lih - 數位布鏡", layout="centered")

# 2. 品牌字體大強化
# 我們使用 HTML 語法來精確控制字體大小 (size) 與顏色 (color)
st.markdown("""
    <style>
    .brand-title {
        font-size: 50px !important;
        font-weight: 800 !important;
        color: #1E3A8A; /* 深藍色，顯得專業穩定 */
        margin-bottom: -10px;
    }
    .brand-sub {
        font-size: 24px !important;
        color: #666666;
        margin-bottom: 20px;
    }
    </style>
    <p class="brand-title">廣笠機械 Goang Lih</p>
    <p class="brand-sub">數位布鏡 - 自動 WPI 分析系統</p>
    """, unsafe_allow_html=True)

st.write("---") # 分隔線
st.write("請上傳布樣照片，系統將利用 AI 頻譜分析技術自動計算 WPI。")

# --- 以下維持分析邏輯 ---

# 上傳檔案按鈕
uploaded_file = st.file_uploader("選擇布樣照片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取影像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 顯示上傳的照片
    st.image(img_bgr, caption='已上傳照片', use_container_width=True)

    # 執行分析邏輯
    with st.spinner('正在分析布料結構...'):
        h, w = img_gray.shape
        x_start = max(0, w // 2 - 450)
        x_end = min(w, x_start + 900)
        roi = img_gray[:, x_start:x_end]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.absolute(sobelx)
        
        projection = np.mean(sobelx, axis=0)
        projection = projection - np.mean(projection)
        fft_result = np.fft.rfft(projection)
        magnitudes = np.abs(fft_result)
        
        search_range = magnitudes[15:45]
        wpi_result = np.argmax(search_range) + 15

    # 顯示結果 (結果的字也稍微放大)
    st.success(f"### 偵測結果：WPI = {wpi_result}")
    
    st.line_chart(projection[:200], use_container_width=True)

st.divider()
st.info("© 2026 廣笠機械 Goang Lih - 致力於傳統紡織業 AI 數位轉型")
