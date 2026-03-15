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
        
        # 影像優化：使用雙邊濾波 (Bilateral Filter)
        # 這種濾波器能在保持邊緣（紗線）的同時，抹平表面的毛羽雜訊
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        h, w = enhanced.shape
        roi = enhanced[:, w//2-400 : w//2+400]
        grad_x = np.absolute(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3))
        
        proj = np.mean(grad_x, axis=0).astype(np.float32)
        proj -= np.mean(proj)
        n = len(proj)
        corr = np.correlate(proj, proj, mode='full')[n-1:]
        
        # --- 核心邏輯：動態頻段准入 ---
        # 搜尋範圍從 Lag 10 到 Lag 50
        # 定義高頻 (Lag 10-14) 與 中低頻 (Lag 15-50)
        high_zone = corr[10:15]
        low_zone = corr[15:51]
        
        max_high = np.max(high_zone)
        max_low = np.max(low_zone)
        
        # 計算訊號比 (SNR)
        # 如果高頻最強點比周圍平均能量高出很多，才認定是真實的 83 WPI
        high_avg = np.mean(high_zone)
        is_high_pure = (max_high > high_avg * 1.5) # 純度檢查
        
        # 判定規則：
        # 只有當高頻能量夠強，且具備足夠純度時，才判定為高頻 (白布 83)
        if max_high > max_low * 0.75 and is_high_pure:
            best_lag = np.argmax(high_zone) + 10
        else:
            # 否則一律在低頻區競爭 (保護 53, 36, 28, 24, 21)
            s_start = 15
            lags = corr[s_start:51]
            # 給予特定布種座標一點點引力
            weights = np.ones_like(lags)
            for i in range(len(lags)):
                l_val = i + s_start
                if 16 <= l_val <= 19: weights[i] = 1.3 # 53
                if 31 <= l_val <= 45: weights[i] = 1.2 # 28, 24, 21
            best_lag = np.argmax(lags * weights) + s_start
        
        # 物理計算 (採用 925 常數)
        raw_wpi = 925 / best_lag
        
        # 硬性歸位矩陣
        if 78 <= raw_wpi <= 95: final_wpi = 83
        elif 50 <= raw_wpi <= 58: final_wpi = 53
        elif 44 <= raw_wpi <= 49: final_wpi = 47
        elif 37 <= raw_wpi <= 41: final_wpi = 38
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
                <p style='color:gray;'>物理計算值: {raw_wpi:.1f} | 模式: {"高頻(白布)" if max_high > max_low * 0.75 and is_high_pure else "中低頻(彩色/粗針)"}</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("© 2026 Goang Lih Machinery Co., Ltd.")
