import streamlit as st
import cv2
import numpy as np

# 網頁標題
st.markdown("### 廣笠機械 Goang Lih")
st.set_page_config(page_title="數位布鏡分析儀", layout="centered")
st.title("🔬 數位布鏡 - 自動 WPI 分析")
st.write("請上傳布樣照片，系統將自動計算 1 英吋內的線圈數量。")

# 上傳檔案按鈕
uploaded_file = st.file_uploader("選擇布樣照片...", type=["jpg", "jpeg", "png"])

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
        # 取中心 900 像素
        x_start = max(0, w // 2 - 450)
        x_end = min(w, x_start + 900)
        roi = img_gray[:, x_start:x_end]

        # 影像增強
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.absolute(sobelx)
        
        # 頻譜分析 (FFT)
        projection = np.mean(sobelx, axis=0)
        projection = projection - np.mean(projection)
        fft_result = np.fft.rfft(projection)
        magnitudes = np.abs(fft_result)
        
        # 搜尋 WPI (15-45 範圍)
        search_range = magnitudes[15:45]
        wpi_result = np.argmax(search_range) + 15

    # 顯示結果
    st.success(f"### 偵測結果：WPI = {wpi_result}")
    
    # 增加裝飾性圖表 (讓產品看起來更專業)
    st.line_chart(projection[:200], use_container_width=True)
    st.caption("線圈波紋訊號圖 (前 200 像素)")

st.divider()

st.info("提示：請確保拍照時，1 英吋範圍剛好填滿 900 像素寬度以獲得最佳精度。")
