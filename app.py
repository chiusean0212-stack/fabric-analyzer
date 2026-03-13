import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import numpy as np
import os
import sys

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

OFFLINE_PASSWORD = "777"

class GoangLihApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("廣笠機械 Goang Lih - AI Analysis System")
        self.geometry("900x800")
        ctk.set_appearance_mode("light")
        self.show_login_screen()

    def show_login_screen(self):
        self.login_frame = ctk.CTkFrame(self, fg_color="white")
        self.login_frame.place(relx=0.5, rely=0.5, anchor="center")
        try:
            logo_path = get_resource_path("LOGO.png")
            logo_img = Image.open(logo_path)
            ctk.CTkLabel(self.login_frame, image=ctk.CTkImage(logo_img, size=(120, 120)), text="").pack(pady=10)
        except: pass
        ctk.CTkLabel(self.login_frame, text="廣笠機械 - 系統授權", font=("Microsoft JhengHei", 24, "bold"), text_color="#1E3A8A").pack(pady=10)
        self.pwd_entry = ctk.CTkEntry(self.login_frame, placeholder_text="請輸入密碼", show="*", width=200)
        self.pwd_entry.pack(pady=10)
        ctk.CTkButton(self.login_frame, text="登入系統", command=self.check_password, fg_color="#FF0000").pack(pady=20)

    def check_password(self):
        if self.pwd_entry.get() == OFFLINE_PASSWORD:
            self.login_frame.destroy()
            self.show_main_screen()
        else:
            messagebox.showerror("Error", "Incorrect Password")

    def show_main_screen(self):
        header = ctk.CTkFrame(self, fg_color="#F0F2F6", corner_radius=0)
        header.pack(side="top", fill="x")
        ctk.CTkLabel(header, text="廣笠機械 ", font=("Microsoft JhengHei", 36, "bold"), text_color="#1E3A8A").pack(side="left", padx=(20, 0), pady=20)
        ctk.CTkLabel(header, text="Goang Lih", font=("Microsoft JhengHei", 28, "bold"), text_color="#FF0000").pack(side="left", pady=20)
        
        ctk.CTkButton(self, text="📁 開啟 FiberCatch 照片 (Open Photo)", command=self.load_image, fg_color="#1E3A8A", height=45).pack(pady=20)
        self.img_label = ctk.CTkLabel(self, text="尚未選擇照片 / No Image Selected", fg_color="#E0E0E0", width=600, height=400, corner_radius=10)
        self.img_label.pack(pady=10)
        self.res_label = ctk.CTkLabel(self, text="WPI = --", font=("Microsoft JhengHei", 48, "bold"), text_color="#1E3A8A")
        self.res_label.pack(pady=20)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            try:
                img_array = np.fromfile(path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None: return
                
                # 預處理：維持溫和降噪與高對比
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(img_blurred)
                
                h, w = enhanced.shape
                x_start = max(0, w // 2 - 450)
                x_end = min(w, x_start + 900)
                roi = enhanced[:, x_start:x_end]

                # 梯度與投影
                grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = cv2.magnitude(grad_x, grad_y)
                
                projection = np.mean(grad_mag, axis=0).astype(np.float32)
                projection -= np.mean(projection)
                
                # 自相關
                n = len(projection)
                corr = np.correlate(projection, projection, mode='full')[n-1:]
                
                # --- 【倍頻鎖定優化】 ---
                # WPI 15-60 對應 Lag 15-60 (這是最常見的穩定區間)
                # 我們縮短搜尋範圍，避免 AI 去抓過大的間隔 (Lag > 60 對應 WPI < 15)
                search_start, search_end = 15, 65 
                lags = corr[search_start:search_end]
                
                # 尋找最強峰值
                peak_idx = np.argmax(lags)
                best_lag = peak_idx + search_start
                
                # 計算初步結果
                wpi_result = round(900 / best_lag)
                # -----------------------

                display_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                self.img_label.configure(image=ctk.CTkImage(display_img, size=(600, 400)), text="")
                self.res_label.configure(text=f"偵測結果 (Result)：WPI = {wpi_result}")
                
            except Exception as e:
                messagebox.showerror("Error", f"分析發生錯誤: {str(e)}")

if __name__ == "__main__":
    GoangLihApp().mainloop()




