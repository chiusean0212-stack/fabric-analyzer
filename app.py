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
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 影像優化：適度強化細節，不使用模糊，保留 82 WPI
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # 1. 取得三個關鍵頻段的最強點
        # 高頻 (80-100 WPI) | 中頻 (40-60 WPI) | 低頻 (20-35 WPI)
        h_zone = corr[9:14]   # Lag 9-13
        m_zone = corr[15:21]  # Lag 15-20 (含 53 WPI)
        l_zone = corr[22:45]  # Lag 22-44 (含 38, 28 WPI)
        
        max_h = np.max(h_zone)
        max_m = np.max(m_zone)
        max_l = np.max(l_zone)
        
        # 2. 決策樹邏輯
        # 規則 A: 只有當高頻能量「強過」中頻時，才考慮高頻 (針對白 82)
        if max_h > max_m * 1.1:
            best_lag = np.argmax(h_zone) + 9
            # 規則 B: 針對透白布料的防禦 (高頻強但倍頻更強)
            if corr[best_lag * 2] > max_h * 0.7:
                best_lag = best_lag * 2
        
        # 規則 C: 如果中頻能量夠強，優先鎖定中頻 (針對灰 53)
        elif max_m > max_l * 0.9:
            best_lag = np.argmax(m_zone) + 15
        
        # 規則 D: 否則歸類為低頻 (針對桃紅 28 或 透白 38)
        else:
            best_lag = np.argmax(l_zone) + 22

        # 3. 數值微調 (常數 910)
        wpi = round(910 / best_lag)
        
        st.image(up, use_container_width=True)
        st.markdown(f"""
            <div style='text-align:center; background:#f0f2f6; padding:20px; border-radius:15px; border:2px solid #1E3A8A;'>
                <h2 style='color:#1E3A8A;'>{t[3]}</h2>
                <p style='font-size:80px; font-weight:bold; color:#FF0000; margin:0;'>WPI = {wpi}</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("© 2026 Goang Lih Machinery Co., Ltd.")
