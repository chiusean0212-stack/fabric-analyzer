import streamlit as st
import cv2
import numpy as np
import os

# --- 1. 設定驗證碼 (您可以隨時修改這裡的字串) ---
CORRECT_PASSWORD = "777" # 這是您的驗證碼，您可以改成您喜歡的數字或英文

# 網頁頁籤設定
st.set_page_config(page_title="廣笠機械 Goang Lih - 數位布鏡", layout="wide")

# --- 2. 密碼驗證邏輯 ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# 如果還沒驗證成功，顯示登入畫面
if not st.session_state["authenticated"]:
    st.title("🔒 廣笠機械 - 系統訪問授權")
    pwd_input = st.text_input("請輸入授權驗證碼以開始使用：", type="password")
    
    if st.button("確認登入"):
        if pwd_input == CORRECT_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun() # 驗證成功，重新整理頁面顯示功能
        else:
            st.error("驗證碼錯誤，請重新輸入或聯繫廣笠機械。")
    st.stop() # 沒驗證過，後面的程式碼都不會執行

# --- 3. 驗證成功後才顯示的內容 (原本的排版與分析邏輯) ---

# 自定義樣式
st.markdown("""
    <style>
    .company-name-zh { font-size: 52px !important; font-weight: 800 !important; color: #1E3A8A; margin: 0; }
    .company-name-en { font-size: 38px !important; font-weight: 700 !important; color: #FF0000; margin-left: 10px; }
    .system-subtitle { font-size: 26px !important; color: #555555; margin-top: 5px; }
    .result-box { background-color: #F0F2F6; padding: 20px; border-radius: 10px; border-left: 10px solid #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=130)
with col2:
    st.markdown('<p class="company-name-zh">廣笠機械 <span class="company-name-en">Goang Lih</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="system-subtitle">數位布鏡 - 自動 WPI 分析系統</p>', unsafe_allow_html=True)

st.write("---")

# 登出按鈕 (放在角落)
if st.sidebar.button("登出系統"):
    st.session_state["authenticated"] = False
    st.rerun()

# --- 核心分析邏輯 (與之前相同) ---
st.write("### 📥 上傳布樣")
uploaded_file = st.file_uploader("選擇照片 (建議解析度：1英吋 = 900像素)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
        projection = np.mean(sobelx, axis=0) - np.mean(np.mean(sobelx, axis=0))
        fft_result = np.fft.rfft(projection)
        magnitudes = np.abs(fft_result)
        search_range = magnitudes[15:45]
        wpi_result = np.argmax(search_range) + 15

    st.markdown(f"""
        <div class="result-box">
            <h2 style="color:#333; margin:0;">偵測結果：WPI = <span style="color:#1E3A8A; font-size:48px;">{wpi_result}</span></h2>
        </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("© 2026 廣笠機械 Goang Lih | 專業針織機械製造 | AI 數位轉型專案")

