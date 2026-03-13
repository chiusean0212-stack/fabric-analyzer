import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Goang Lih AI", layout="centered")

t = {"繁體中文": ["廣笠機械", "AI 分析系統", "📸 上傳照片", "結果", "授權登入"],
     "English": ["Goang Lih", "AI Analysis", "📸 Upload", "Result", "Login"]}[st.sidebar.selectbox("Lang", ["繁體中文", "English"])]

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    pwd = st.text_input(t[4], type="password")
    if st.button(t[4]):
        if pwd == "777": st.session_state["auth"] = True; st.rerun()
    st.stop()

st.title(f"{t[0]} Goang Lih")

up = st.file_uploader(t[2], type=['jpg', 'jpeg', 'png'])
if up:
    try:
        img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 影像優化：使用中度 CLAHE，保留細節但不產生過多雜訊
        enhanced = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8)).apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]

        # --- 核心邏輯：多頻段競爭 (不分顏色) ---
        # 抓取四個關鍵 WPI 區間的能量
        # 82 WPI (Lag 10-12) | 53 WPI (Lag 16-18) | 38 WPI (Lag 23-25) | 28 WPI (Lag 31-34)
        e_82 = np.max(corr[10:13])
        e_53 = np.max(corr[16:19])
        e_38 = np.max(corr[23:26])
        e_28 = np.max(corr[31:35])
        
        # 判定順序 (權重微調)
        # 1. 如果 82 WPI 能量非常集中且強大 (針對白布)
        if e_82 > e_53 * 1.2 and e_82 > e_38 * 1.2:
            final_wpi = 82
        # 2. 如果 53 WPI 能量勝出 (針對灰色)
        elif e_53 > e_82 * 1.0 and e_53 > e_38 * 1.1:
            final_wpi = 53
        # 3. 如果 38 WPI 與 82 WPI 能量接近 (針對透白，透光會產生兩倍頻)
        elif e_38 > e_82 * 0.7 and e_38 > e_28 * 1.1:
            final_wpi = 38
        # 4. 其他情況歸類為 28 (針對桃紅)
        else:
            final_wpi = 28

        st.image(img_bgr, use_container_width=True)
        st.markdown(f"""
            <div style='text-align:center; background:#f0f2f6; padding:20px; border-radius:15px; border:2px solid #1E3A8A;'>
                <h2 style='color:#1E3A8A;'>分析結果</h2>
                <p style='font-size:80px; font-weight:bold; color:#FF0000; margin:0;'>WPI = {final_wpi}</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("© 2026 Goang Lih Machinery Co., Ltd.")
