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
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 影像優化
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8)).apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]

        # --- 核心數據鎖定：六大布種座標 ---
        e_82 = np.max(corr[10:13]) # 白
        e_53 = np.max(corr[16:20]) # 灰
        e_38 = np.max(corr[23:26]) # 透白 / 綠 (偏誤區)
        e_35 = np.max(corr[26:29]) # 綠色 35 (新增)
        e_28 = np.max(corr[31:35]) # 桃紅
        e_24 = np.max(corr[37:41]) # 米白 24 (新增)
        e_20 = np.max(corr[43:48]) # 米黃 20

        # --- 色彩輔助判斷 ---
        avg_s = np.mean(img_hsv[:,:,1]) # 飽和度
        avg_v = np.mean(img_hsv[:,:,2]) # 亮度
        
        # --- 決策邏輯：排除法與競爭法 ---
        
        # 1. 米白 24 專屬保護 (防止跳 82)
        # 如果亮度高但飽和度有一點點(米白)，且低頻能量大於高頻一半，鎖定低頻
        if 20 < avg_s < 60 and e_24 > e_82 * 0.5:
            final_wpi = 24
        
        # 2. 米黃 20 優先 (最粗針織)
        elif e_20 > e_28 * 0.9 and e_20 > e_35 * 0.9:
            final_wpi = 20
            
        # 3. 白色 82 (必須是極低飽和度且高頻具備壓倒性)
        elif avg_s < 30 and e_82 > e_53 * 1.3 and e_82 > e_24 * 1.5:
            final_wpi = 82
            
        # 4. 灰色 53
        elif e_53 > e_82 * 1.0 and e_53 > e_38 * 1.1:
            final_wpi = 53
            
        # 5. 綠色 35 與 透白 38 的微細競爭
        elif e_35 > e_38 * 0.95 and e_35 > e_28 * 1.1:
            final_wpi = 35
            
        # 6. 透白 38
        elif e_38 > e_82 * 0.7 and e_38 > e_28 * 1.1:
            final_wpi = 38
            
        # 7. 桃紅 28
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
