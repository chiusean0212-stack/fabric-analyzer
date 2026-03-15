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
        
        # 影像優化：針對高頻紗線強化邊緣，不使用過度模糊
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # --- 核心邏輯：高頻優先准入制 ---
        # 取得高頻區 (Lag 10-13, 約 70-90 WPI) 與 低頻區 (Lag 15-50) 的最強能量
        high_zone = corr[10:14]
        low_zone = corr[15:51]
        
        max_high = np.max(high_zone)
        max_low = np.max(low_zone)
        
        # 判斷門檻：如果高頻能量具備一定規模 (達到低頻的 65% 以上)
        # 代表這極可能是高密度的白布，強制選擇高頻訊號
        if max_high > max_low * 0.65:
            best_lag = np.argmax(high_zone) + 10
        else:
            # 否則在低頻區套用加權競爭 (保護 53, 36, 28, 24)
            s_start = 15
            lags = corr[s_start:51]
            weights = np.ones_like(lags)
            for i in range(len(lags)):
                l_val = i + s_start
                if 16 <= l_val <= 19: weights[i] = 1.3  # 53
                if 31 <= l_val <= 42: weights[i] = 1.2  # 28, 24
            best_lag = np.argmax(lags * weights) + s_start
        
        # 物理計算
        raw_wpi = 925 / best_lag
        
        # 硬性歸位矩陣
        if 78 <= raw_wpi <= 92: final_wpi = 83
        elif 50 <= raw_wpi <= 58: final_wpi = 53
        elif 40 <= raw_wpi <= 49: final_wpi = 47
        elif 37 <= raw_wpi <= 39.5: final_wpi = 38
        elif 34 <= raw_wpi <= 36.9: final_wpi = 36
        elif 27 <= raw_wpi <= 31: final_wpi = 28
        elif 23 <= raw_wpi <= 26.9: final_wpi = 24
        elif 19 <= raw_wpi <= 22: final_wpi = 21
        else: final_wpi = round(raw_wpi)

        st.image(img_bgr, use_container_width=True)
        st.markdown(f"""
            <div style='text-align:center; background:#f0f2f6; padding:20px; border-radius:15px; border:2px solid #1E3A8A;'>
                <h2 style='color:#1E3A8A;'>分析結果</h2>
                <p style='font-size:80px; font-weight:bold; color:#FF0000; margin:0;'>WPI = {final_wpi}</p>
                <p style='color:gray;'>計算值: {raw_wpi:.1f} | 頻段: {"高頻" if max_high > max_low * 0.65 else "低頻"}</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("© 2026 Goang Lih Machinery Co., Ltd.")
