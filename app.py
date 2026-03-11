import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# --- 1. 客戶訂閱管理資料庫 (您可以隨時在這裡新增客戶) ---
# 格式: "帳號": {"password": "密碼", "expiry": "到期日期 YYYY-MM-DD"}
USER_DATABASE = {
    "admin": {"password": "gl-master-777", "expiry": "2099-12-31"},
    "egypt_factory01": {"password": "eg-888", "expiry": "2026-12-31"},
    "bangladesh_test": {"password": "bd-999", "expiry": "2026-04-01"}, # 試用客戶
}

def check_subscription(username, password):
    if username in USER_DATABASE:
        user_info = USER_DATABASE[username]
        if user_info["password"] == password:
            # 檢查是否到期
            expiry_date = datetime.strptime(user_info["expiry"], "%Y-%m-%d")
            if datetime.now() <= expiry_date:
                return True, "驗證成功"
            else:
                return False, "您的訂閱已到期，請聯繫廣笠機械進行續約。"
    return False, "帳號或密碼錯誤"

# --- 2. 網頁基本設定 ---
st.set_page_config(page_title="廣笠機械 Goang Lih - AI 數位認證系統", layout="wide")

# --- 3. 登入介面 ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 廣笠機械 - AI 數位認證系統登入")
    
    col_l, col_r = st.columns(2)
    with col_l:
        user_input = st.text_input("客戶帳號")
        pwd_input = st.text_input("授權密碼", type="password")
    
    if st.button("確認授權並進入"):
        success, message = check_subscription(user_input, pwd_input)
        if success:
            st.session_state["authenticated"] = True
            st.session_state["current_user"] = user_input
            st.rerun()
        else:
            st.error(message)
    
    st.info("💡 尚未取得授權？請連繫廣笠機械辦事處。")
    st.stop()

# --- 4. 登入成功後的功能介面 (原本的分析邏輯) ---

# 側邊欄顯示狀態
st.sidebar.title("👤 客戶中心")
st.sidebar.write(f"目前登入：{st.session_state['current_user']}")
st.sidebar.write(f"到期日期：{USER_DATABASE[st.session_state['current_user']]['expiry']}")
if st.sidebar.button("登出系統"):
    st.session_state["authenticated"] = False
    st.rerun()

# 品牌標題排版 (與之前相同)
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=130)
with col_title:
    st.markdown('<p style="font-size:52px; font-weight:800; color:#1E3A8A; margin:0;">廣笠機械 <span style="color:#FF0000; font-size:38px;">Goang Lih</span></p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:26px; color:#555555;">FiberCatch 專用 - AI 線圈密度分析系統</p>', unsafe_allow_html=True)

# ... [保留原本的分析與校正邏輯] ...
