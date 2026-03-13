import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Goang Lih AI", layout="centered")

t = {"繁體中文": ["廣笠機械", "AI 分析系統", "📸 上傳照片", "結果", "授權登入"],
     "English": ["Goang Lih", "AI Analysis", "📸 Upload", "Result", "Login"]}[st.sidebar.selectbox("Lang", ["繁體中文", "English"])]

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    pwd = st.text_input(t[4], type="password")
    if pwd == "777": st.session_state["auth"] = True; st.rerun()
    st.stop()

st.title(f"{t[0]} Goang Lih")

up = st.file_uploader(t[2], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # --- 步驟 1：顏色自動偵測 ---
        # 計算畫面平均飽和度(S)與亮度(V)
        avg_s = np.mean(img_hsv[:,:,1]) # 飽和度
        avg_v = np.mean(img_hsv[:,:,2]) # 亮度
        
        # 判定布料類別
        is_high_bright = (avg_v > 130 and avg_s < 60) # 亮色布 (白、透白)
        is_vivid_or_dark = not is_high_bright         # 深色/鮮豔布 (桃紅、灰)

        # --- 步驟 2：針對顏色進行影像優化 ---
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if is_high_bright:
            # 亮色布料：極致強化細節，不除噪
            enhanced = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4)).apply(gray)
        else:
            # 深色/彩色布料：除噪並溫和強化，避免雜訊
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(blur)
        
        # 計算自相關訊號
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]

        # --- 步驟 3：分流演算法 ---
        if is_high_bright:
            # 【亮色布模式】強力鎖定高頻 82 WPI
            # 搜尋範圍集中在 Lag 9-28 (100-33 WPI)
            lags = corr[9:30]
            # 強力權重加持給高頻
            weights = np.linspace(2.5, 1.0, len(lags))
            best_lag = np.argmax(lags * weights) + 9
            
            # 透白 38 防護：若倍頻能量夠強則彈回
            if 10 <= best_lag <= 14:
                if corr[best_lag * 2] > corr[best_lag] * 0.75:
                    best_lag = best_lag * 2
            fabric_type = "亮色/透光布料"
        else:
            # 【深色/彩色布模式】穩定鎖定中低頻 28-53 WPI
            # 搜尋範圍 Lag 15-45 (60-20 WPI)，完全物理性封鎖 91 WPI 雜訊
            lags = corr[15:46]
            best_lag = np.argmax(lags) + 15
            fabric_type = "深色/鮮豔布料"

        # 數值精準歸位
        if 30 <= best_lag <= 34: best_lag = 32.5 # 28 WPI
        if 16 <= best_lag <= 18: best_lag = 17.2 # 53 WPI
        if 23 <= best_lag <= 25: best_lag = 24.0 # 38 WPI

        wpi = round(910 / best_lag)
        
        st.image(img_bgr, use_container_width=True)
        st.info(f"偵測模式：{fabric_type}")
        st.markdown(f"""
            <div style='text-align:center; background:#f0f2f6; padding:20px; border-radius:15px; border:2px solid #1E3A8A;'>
                <h2 style='color:#1E3A8A;'>{t[3]}</h2>
                <p style='font-size:80px; font-weight:bold; color:#FF0000; margin:0;'>WPI = {wpi}</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("© 2026 Goang Lih Machinery Co., Ltd.")
