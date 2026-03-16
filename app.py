import streamlit as st
import cv2
import numpy as np

# 頁面配置
st.set_page_config(page_title="Goang Lih AI Analysis", layout="centered", page_icon="⚙️")

# 自定義 CSS 提升介面質感
st.markdown("""
    <style>
    .login-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        padding: 50px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 語言包
t = {
    "繁體中文": ["廣笠機械", "AI 智能織物分析系統", "📸 上傳布料照片", "分析結果", "授權登入", "請輸入系統授權密碼", "密碼錯誤", "專業 · 精準 · 智能"],
    "English": ["Goang Lih", "AI Fabric Analysis System", "📸 Upload Photo", "Result", "Login", "Enter System Password", "Wrong Password", "Professional · Precise · Smart"]
}

# 側邊欄語言切換
lang = st.sidebar.selectbox("Language / 語言", ["繁體中文", "English"])
txt = t[lang]

# 登入邏輯
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    # 登入視覺容器
    st.markdown(f"""
        <div class="login-container">
            <h1 style='margin-bottom:0;'>{txt[0]}</h1>
            <p style='font-size:1.2em; opacity:0.8;'>{txt[1]}</p>
            <hr style='border-color: rgba(255,255,255,0.2);'>
            <p style='font-size:0.9em; margin-bottom:20px;'>{txt[7]}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 密碼輸入
    with st.container():
        st.write(" ")
        pwd = st.text_input(txt[5], type="password", help="請輸入 3 位數授權碼")
        if st.button(txt[4]):
            if pwd == "777":
                st.session_state["auth"] = True
                st.success("驗證成功！正在啟動系統...")
                st.rerun()
            else:
                st.error(txt[6])
    st.stop()

# --- 進入分析主程式 ---
st.title(f"⚙️ {txt[0]} {txt[1]}")

up = st.file_uploader(txt[2], type=['jpg', 'jpeg', 'png'])

if up:
    try:
        img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 影像優化：雙邊濾波 + CLAHE
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
        
        # 核心判定邏輯 (延續之前的穩定版)
        high_zone = corr[10:15]
        low_zone = corr[15:51]
        max_high = np.max(high_zone)
        max_low = np.max(low_zone)
        high_avg = np.mean(high_zone)
        is_high_pure = (max_high > high_avg * 1.5)
        
        if max_high > max_low * 0.75 and is_high_pure:
            best_lag = np.argmax(high_zone) + 10
        else:
            s_start = 15
            lags = corr[s_start:51]
            weights = np.ones_like(lags)
            for i in range(len(lags)):
                l_val = i + s_start
                if 16 <= l_val <= 19: weights[i] = 1.3
                if 31 <= l_val <= 45: weights[i] = 1.2
            best_lag = np.argmax(lags * weights) + s_start
        
        raw_wpi = 925 / best_lag
        
        # 歸位矩陣
        if 78 <= raw_wpi <= 95: final_wpi = 83
        elif 50 <= raw_wpi <= 58: final_wpi = 53
        elif 44 <= raw_wpi <= 49: final_wpi = 47
        elif 37 <= raw_wpi <= 41: final_wpi = 38
        elif 34 <= raw_wpi <= 36.9: final_wpi = 36
        elif 27 <= raw_wpi <= 31: final_wpi = 28
        elif 23 <= raw_wpi <= 26.9: final_wpi = 24
        elif 19 <= raw_wpi <= 22: final_wpi = 21
        else: final_wpi = round(raw_wpi)

        # 結果呈現
        st.image(img_bgr, use_container_width=True)
        st.markdown(f"""
            <div style='text-align:center; background:#f8fafc; padding:30px; border-radius:20px; border:3px solid #1e3a8a; margin-top:20px;'>
                <h2 style='color:#1e3a8a; margin-bottom:10px;'>{txt[3]}</h2>
                <div style='background:#1e3a8a; color:white; display:inline-block; padding:10px 40px; border-radius:50px; margin-bottom:15px;'>
                    <span style='font-size:1.2em; letter-spacing:2px;'>WPI</span>
                </div>
                <p style='font-size:100px; font-weight:bold; color:#ef4444; margin:0; line-height:1;'>{final_wpi}</p>
                <p style='color:gray; margin-top:10px;'>精確計算參考: {raw_wpi:.1f}</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"系統分析錯誤: {e}")

# 頁尾
st.markdown("---")
st.caption(f"© 2026 Goang Lih Machinery Co., Ltd. All Rights Reserved. | 廣笠機械工業 版權所有")
