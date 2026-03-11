import streamlit as st
import cv2
import numpy as np
import os

# 1. 網頁頁籤設定
st.set_page_config(page_title="廣笠機械 Goang Lih - 數位布鏡", layout="wide")

# 2. 依照模擬圖排版：LOGO 與 標題橫向並列
# 使用自定義 HTML 和 CSS 來達成精確的排版
st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .logo-img {
        margin-right: 25px;
    }
    .title-text {
        line-height: 1.1;
    }
    .company-name-zh {
        font-size: 52px !important;
        font-weight: 800 !important;
        color: #FF0000; /* 紅色字體 */
        margin: 0;
        font-family: "Microsoft JhengHei", sans-serif;
    }
    .company-name-en {
        font-size: 36px !important;
        font-weight: 600 !important;
        color: #1E3A8A; /* 深藍色英文，增加層次感 */
        margin: 0;
    }
    .system-subtitle {
        font-size: 26px !important;
        color: #555555;
        font-weight: 400;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# 建立 Header 區塊
header_html = f"""
<div class="header-container">
    <div class="logo-img">
        <img src="https://raw.githubusercontent.com/{st.get_option("server.address") if st.get_option("server.address") else "your-username"}/fabric-analyzer/main/LOGO.png" width="120">
    </div>
    <div class="title-text">
        <p class="company-name-zh">廣笠機械 <span class="company-name-en">Goang Lih</span></p>
        <p class="system-subtitle">數位布鏡 - 自動 WPI 分析系統</p>
    </div>
</div>
"""

# 如果 GitHub 上的圖片路徑不好抓，我們改用 Streamlit 標準方式結合 HTML
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=130)
with col2:
    st.markdown('<p class="company-name-zh">廣笠機械 <span class="company-name-en">Goang Lih</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="system-subtitle">數位布鏡 - 自動 WPI 分析系統</p>', unsafe_allow_html=True)

st.write("---")

# --- 核心分析邏輯 ---

st.write("### 📥 上傳布樣")
uploaded_file = st.file_uploader("選擇照片 (建議解析度：1英吋 = 900像素)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取與處理
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 顯示上傳照片
    st.image(img_bgr, caption='原始布樣照片', use_container_width=True)

    with st.spinner('AI 正在計算線圈密度...'):
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

    # 顯眼的結果顯示
    st.markdown(f"""
        <div style="background-color:#F0F2F6; padding:20px; border-radius:10px; border-left: 10px solid #FF0000;">
            <h2 style="color:#333; margin:0;">偵測結果：WPI = <span style="color:#FF0000; font-size:48px;">{wpi_result}</span></h2>
        </div>
    """, unsafe_allow_html=True)
    
    # 輔助圖表
    with st.expander("查看波紋分析數據"):
        st.line_chart(projection[:300])
        st.caption("線圈波紋訊號圖 (前 300 像素)")

st.divider()
st.caption("© 2026 廣笠機械 Goang Lih | 專業針織機械製造 | AI 數位轉型專案")
