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
        
        # 影像優化：適度強化，確保低頻粗針織與高頻細紗都能捕捉
        enhanced = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8)).apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]

        # --- 定義五大布種的核心 Lag 座標 ---
        # 82 WPI -> Lag 11
        # 53 WPI -> Lag 17
        # 38 WPI -> Lag 24
        # 28 WPI -> Lag 32
        # 20 WPI -> Lag 45 (米黃色)
        
        e_82 = np.max(corr[10:13])
        e_53 = np.max(corr[16:20])
        e_38 = np.max(corr[22:26])
        e_28 = np.max(corr[30:35])
        e_20 = np.max(corr[42:48]) # 擴大米黃色檢測範圍
        
        # --- 競爭決策邏輯 ---
        
        # 1. 優先判斷米黃色：如果極低頻區有強訊號，優先歸類
        if e_20 > e_28 * 0.95 and e_20 > e_38 * 0.95:
            final_wpi = 20
        # 2. 判斷高密白布
        elif e_82 > e_53 * 1.3 and e_82 > e_38 * 1.3:
            final_wpi = 82
        # 3. 判斷灰色中密布
        elif e_53 > e_82 * 1.0 and e_53 > e_38 * 1.2:
            final_wpi = 53
        # 4. 判斷透白布 (利用 82 與 38 的相關性)
        elif e_38 > e_82 * 0.7 and e_38 > e_28 * 1.1:
            final_wpi = 38
        # 5. 桃紅色
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
