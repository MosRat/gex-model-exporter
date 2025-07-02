# modified from https://github.com/RQLuo/MixTeX-Latex-OCR/blob/main/mixtexgui/mixtex_ui.py
# Renqing Luo
# Commercial use prohibited

import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import pystray
from pystray import MenuItem as item
import threading
import pyperclip
import time
import sys
import os
import csv
import re
import ctypes
import io
from pathlib import Path

# --- GEX Backend (gex.py) ---
# This part defines the interface to the libgex.dll backend.
# In a real project, this would typically be in its own gex.py file.

# Determine DLL path for both script and bundled executable
if hasattr(sys, '_MEIPASS'):
    _dll_path = Path(sys._MEIPASS) / 'libgex.dll'
else:
    _dll_path = Path(__file__).resolve().parent / 'libgex.dll'

# Load the DLL
try:
    _lib = ctypes.CDLL(str(_dll_path))
except OSError as e:
    # Provide a more user-friendly error if the DLL is missing.
    ctypes.windll.user32.MessageBoxW(0, f"无法加载核心库 libgex.dll。\n请确保它与 MixTeX.exe 在同一目录下。\n\n错误: {e}", "库加载失败", 0x10)
    raise ImportError(f"Failed to load libgex.dll: {e}") from e

# Constants
MAX_DEVICE_NUM = 12

# Type definitions
class GexError(ctypes.Structure):
    _fields_ = [("msg", ctypes.c_char_p)]
    def __str__(self):
        return f"GexError: {self.msg.decode('utf-8') if self.msg else 'Unknown error'}"

class GexDevice(ctypes.Structure):
    _fields_ = [("idx", ctypes.c_size_t), ("des", ctypes.c_char_p), ("name", ctypes.c_char_p), ("cap", ctypes.c_size_t)]
    def __str__(self):
        return (f"GexDevice(idx={self.idx}, name={self.name.decode('utf-8') if self.name else None}, "
                f"description={self.des.decode('utf-8') if self.des else None}, cap={self.cap})")

class GexDeviceList(ctypes.Structure):
    _fields_ = [("devices", GexDevice * MAX_DEVICE_NUM), ("num_devices", ctypes.c_size_t)]
    def __str__(self):
        devices_str = "\n  ".join(str(self.devices[i]) for i in range(self.num_devices))
        return f"GexDeviceList(num_devices={self.num_devices}):\n  {devices_str}"

# Function prototypes
_lib.gex_error_set.argtypes = [ctypes.c_char_p]
_lib.gex_error_set.restype = None
_lib.get_last_error.argtypes = []
_lib.get_last_error.restype = GexError
_lib.gex_init_with_onnx.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.gex_init_with_onnx.restype = ctypes.c_void_p
_lib.gex_free.argtypes = [ctypes.c_void_p]
_lib.gex_free.restype = None
GEX_STREAM_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_lib.gex_inference_mem_stream.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t, GEX_STREAM_CALLBACK]
_lib.gex_inference_mem_stream.restype = ctypes.c_char_p

class GexContext:
    def __init__(self, ptr: int):
        self._ptr = ptr
        if not self._ptr:
            error = _lib.get_last_error()
            raise RuntimeError(f"Failed to create GexContext: {error}")
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.gex_free(self._ptr)
            self._ptr = None
    @property
    def ptr(self) -> int:
        if not self._ptr:
            raise RuntimeError("GexContext has been freed")
        return self._ptr
    @classmethod
    def init_with_onnx(cls, model_path: str, onnx_path: str) -> 'GexContext':
        model_path_bytes = model_path.encode('utf-8')
        onnx_path_bytes = onnx_path.encode('utf-8')
        ptr = _lib.gex_init_with_onnx(model_path_bytes, onnx_path_bytes)
        return cls(ptr)
    def inference_mem_stream(self, image_data: bytes, callback: callable) -> str:
        buf = (ctypes.c_ubyte * len(image_data)).from_buffer_copy(image_data)
        def wrapped_callback(token: bytes, _: int) -> int:
            return callback(token.decode('utf-8'))
        cb = GEX_STREAM_CALLBACK(wrapped_callback)
        result = _lib.gex_inference_mem_stream(self.ptr, buf, len(image_data), cb)
        if not result:
            error = _lib.get_last_error()
            raise RuntimeError(f"Inference failed: {error}")
        return result.decode('utf-8')

# --- END GEX Backend ---

if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

class MixTeXApp:
    def __init__(self, root):
        self.root = root
        
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            self.dpi_scale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
            self.root.tk.call('tk', 'scaling', self.dpi_scale)
        except Exception as e:
            print(f"DPI 设置失败: {e}")
            self.dpi_scale = 1.0
        
        self.root.title('MixTeX')
        self.root.resizable(False, False)
        self.root.overrideredirect(True)
        self.root.wm_attributes('-topmost', 1)
        self.root.attributes('-alpha', 0.85)
        self.TRANSCOLOUR = '#a9abc6'
        self.is_only_parse_when_show = False
        self.icon = self.load_scaled_image(os.path.join(base_path, "icon.png"))
        self.icon_tk = ImageTk.PhotoImage(self.icon)

        self.main_frame = tk.Frame(self.root, bg=self.TRANSCOLOUR)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.icon_label = tk.Label(self.main_frame, image=self.icon_tk, bg=self.TRANSCOLOUR)
        self.icon_label.pack(pady=self.scale_size(10))

        self.text_frame = tk.Frame(self.main_frame, bg='white', bd=1, relief=tk.SOLID)
        self.text_frame.pack(padx=self.scale_size(5), pady=self.scale_size(5), fill=tk.BOTH, expand=True)

        font_size = self.scale_size(9)
        self.text_box = tk.Text(self.text_frame, wrap=tk.WORD, bg='white', fg='black', 
                               height=6, width=30, font=('Arial', font_size))
        self.text_box.pack(padx=self.scale_size(2), pady=self.scale_size(2), fill=tk.BOTH, expand=True)

        self.icon_label.bind('<ButtonPress-1>', self.start_move)
        self.icon_label.bind('<B1-Motion>', self.do_move)
        self.icon_label.bind('<ButtonPress-3>', self.show_menu)
        self.data_folder = "data"
        self.metadata_file = os.path.join(self.data_folder, "metadata.csv")
        self.use_dollars_for_inline_math = False
        self.convert_align_to_equations_enabled = False
        self.ocr_paused = False
        self.annotation_window = None
        self.current_image = None
        self.output = None
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['file_name', 'text', 'feedback'])

        self.menu = tk.Menu(self.root, tearoff=0)
        settings_menu = tk.Menu(self.menu, tearoff=0)
        settings_menu.add_checkbutton(label="$ 公式 $", onvalue=1, offvalue=0, command=self.toggle_latex_replacement, variable=tk.BooleanVar(value=self.use_dollars_for_inline_math))
        settings_menu.add_checkbutton(label="$$ 单行公式 $$", onvalue=1, offvalue=0, command=self.toggle_convert_align_to_equations, variable=tk.BooleanVar(value=self.convert_align_to_equations_enabled))
        self.menu.add_cascade(label="设置", menu=settings_menu)
        self.menu.add_command(label="反馈标注", command=self.show_feedback_options)
        self.menu.add_command(label="最小化", command=self.minimize)
        self.menu.add_command(label="关于", command=self.show_about)
        self.menu.add_command(label="打赏", command=self.show_donate)
        self.menu.add_command(label="退出", command=self.quit)
        if sys.platform == 'darwin':
            self.root.config(menu=self.menu)
        else:
            self.root.bind('<Button-3>', self.show_menu)
            self.root.wm_attributes("-transparentcolor", self.TRANSCOLOUR)

        self.create_tray_icon()

        self.gex_context = self.load_gex_model()
        if self.gex_context is None:
            self.log("模型加载失败，OCR功能已禁用。")
            self.ocr_paused = True
            self.update_icon()
        else:
            self.ocr_thread = threading.Thread(target=self.ocr_loop, daemon=True)
            self.ocr_thread.start()

        self.donate_window = None
        self.is_only_parse_when_show = False
    
    def scale_size(self, size):
        return int(size * self.dpi_scale)
    
    def load_scaled_image(self, image_path, custom_scale=None):
        scale = custom_scale if custom_scale is not None else getattr(self, 'dpi_scale', 1.0)
        if not os.path.exists(image_path):
            print(f"找不到图像文件: {image_path}")
            return Image.new('RGB', (64, 64), (200, 200, 200))
        original = Image.open(image_path)
        if scale > 1.0:
            new_size = (int(original.width * scale), int(original.height * scale))
            return original.resize(new_size, Image.LANCZOS)
        return original

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    def show_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    def save_data(self, image, text, feedback):
        file_name = f"{int(time.time())}.png"
        file_path = os.path.join(self.data_folder, file_name)
        image.save(file_path, 'PNG')
        rows = []
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
        updated = False
        for row in rows[1:]:
            if row[1] == text:
                row[2] = feedback
                updated = True
                break
        if not updated:
            rows.append([file_name, text, feedback])
        with open(self.metadata_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def toggle_latex_replacement(self):
        self.use_dollars_for_inline_math = not self.use_dollars_for_inline_math

    def toggle_convert_align_to_equations(self):
        self.convert_align_to_equations_enabled = not self.convert_align_to_equations_enabled

    def minimize(self):
        self.root.withdraw()
        self.tray_icon.visible = True

    def show_about(self):
        about_text = "MixTeX\n版本: 3.2.4b-gex \n作者: lrqlrqlrq \nQQ群：612725068 \nB站：bilibili.com/8922788 \nGithub:github.com/RQLuo"
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, about_text)

    def show_donate(self):
        donate_text = "\n!!!感谢您的支持!!!\n"
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, donate_text)
        donate_frame = tk.Frame(self.main_frame, bg='white')
        donate_frame.pack(padx=self.scale_size(5), pady=self.scale_size(5), fill=tk.BOTH, expand=True)
        donate_size = self.scale_size(400)
        donate_image = self.load_scaled_image(os.path.join(base_path, "donate.png"))
        donate_image = donate_image.resize((donate_size, donate_size), Image.LANCZOS)
        donate_photo = ImageTk.PhotoImage(donate_image)
        image_label = tk.Label(donate_frame, image=donate_photo)
        image_label.image = donate_photo
        image_label.pack(expand=True, fill=tk.BOTH)
        close_button = tk.Button(donate_frame, text="☒", command=lambda: donate_frame.destroy())
        close_button.place(relx=1.0, rely=0.0, x=-self.scale_size(15), y=self.scale_size(5), width=self.scale_size(12), height=self.scale_size(12), anchor="ne")

    def quit(self):
        self.tray_icon.stop()
        self.root.quit()

    def only_parse_when_show(self):
        self.is_only_parse_when_show = not self.is_only_parse_when_show
        
    def create_tray_icon(self):
        menu = pystray.Menu(
            item('显示', self.show_window),
            item("仅窗口显示时启用", self.only_parse_when_show),
            item('退出', self.quit)
        )
        self.tray_icon = pystray.Icon("MixTeX", self.icon, "MixTeX", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def show_window(self):
        self.root.deiconify()
        self.tray_icon.visible = False

    def load_gex_model(self):
        try:
            model_dir = os.path.join(base_path, 'gex_model')
            if not os.path.exists(model_dir):
                self.log(f"错误: 模型文件夹 'gex_model' 不存在。")
                ctypes.windll.user32.MessageBoxW(0, "模型文件夹 'gex_model' 不存在。\n请确保它与程序在同一目录下。", "模型加载错误", 0)
                return None

            gguf_file, onnx_file = None, None
            for file in os.listdir(model_dir):
                if file.endswith(".gguf"):
                    gguf_file = os.path.join(model_dir, file)
                elif file.lower() == "encoder.onnx":
                    onnx_file = os.path.join(model_dir, file)

            if not gguf_file or not onnx_file:
                msg = "在 'gex_model' 文件夹中找不到 .gguf 或 encoder.onnx 文件。"
                self.log(f"错误: {msg}")
                ctypes.windll.user32.MessageBoxW(0, msg, "模型加载错误", 0)
                return None
            
            self.log(f"加载模型: {os.path.basename(gguf_file)}")
            ctx = GexContext.init_with_onnx(gguf_file, onnx_file)
            self.log('\n===成功加载 Gex 模型===\n')
            return ctx
        except Exception as e:
            self.log(f"模型加载失败: {e}")
            ctypes.windll.user32.MessageBoxW(0, f"模型加载失败: {str(e)}\n请检查模型文件和 libgex.dll 是否完整且兼容。", "模型加载错误", 0)
            return None

    def show_feedback_options(self):
        feedback_menu = tk.Menu(self.menu, tearoff=0)
        feedback_menu.add_command(label="完美", command=lambda: self.handle_feedback("Perfect"))
        feedback_menu.add_command(label="普通", command=lambda: self.handle_feedback("Normal"))
        feedback_menu.add_command(label="失误", command=lambda: self.handle_feedback("Mistake"))
        feedback_menu.add_command(label="错误", command=lambda: self.handle_feedback("Error"))
        feedback_menu.add_command(label="标注", command=self.add_annotation)
        feedback_menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def handle_feedback(self, feedback_type):
        if self.current_image and self.output:
            if self.check_repetition(self.output):
                self.log("反馈已记录: Repeat")
            else:
                self.save_data(self.current_image, self.output, feedback_type)
                self.log(f"反馈已记录: {feedback_type}")
        else:
            self.log("反馈无法记录: 缺少图片或推理输出")

    def add_annotation(self):
        if self.annotation_window: return
        self.annotation_window = tk.Toplevel(self.root)
        self.annotation_window.wm_attributes("-alpha", 0.85)
        self.annotation_window.overrideredirect(True)
        self.annotation_window.wm_attributes('-topmost', 1)
        self.update_annotation_position()
        font_size = self.scale_size(11)
        entry = tk.Entry(self.annotation_window, width=45, font=('Arial', font_size))
        entry.pack(padx=self.scale_size(10), pady=self.scale_size(10))
        entry.focus_set()
        confirm_button = tk.Button(self.annotation_window, text="确认", command=lambda: self.confirm_annotation(entry))
        confirm_button.pack(pady=(0, self.scale_size(10)))
        self.root.bind('<Configure>', lambda e: self.update_annotation_position())

    def confirm_annotation(self, entry):
        annotation = entry.get()
        if annotation and self.current_image and self.output:
            self.handle_feedback(f"Annotation: {annotation}")
            self.log(f"标注已添加: {annotation}")
        else:
            self.log("反馈无法记录: 缺少图片、推理输出或输入标注。")
        self.close_annotation()

    def update_annotation_position(self):
        if self.annotation_window:
            x = self.root.winfo_x() + self.scale_size(10)
            y = self.root.winfo_y() + self.root.winfo_height() + self.scale_size(10)
            self.annotation_window.geometry(f"+{x}+{y}")

    def close_annotation(self):
        if self.annotation_window:
            self.annotation_window.destroy()
        self.annotation_window = None

    def check_repetition(self, s, repeats=12):
        for pattern_length in range(1, len(s) // repeats + 1):
            for start in range(len(s) - repeats * pattern_length + 1):
                pattern = s[start:start + pattern_length]
                if s[start:start + repeats * pattern_length] == pattern * repeats:
                    return True
        return False

    def convert_align_to_equations(self, text):
        text = re.sub(r'\\begin\{align\*\}|\\end\{align\*\}', '', text).replace('&','')
        equations = text.strip().split('\\\\')
        converted = []
        for eq in equations:
            eq = eq.strip().replace('\\[','').replace('\\]','').replace('\n','')
            if eq:
                converted.append(f"$$ {eq} $$")
        return '\n'.join(converted)

    def pad_image(self, img, out_size):
        x_img, y_img = out_size
        background = Image.new('RGB', (x_img, y_img), (255, 255, 255))
        width, height = img.size
        if width < x_img and height < y_img:
            background.paste(img, ((x_img - width) // 2, (y_img - height) // 2))
        else:
            scale = min(x_img / width, y_img / height)
            new_width, new_height = int(width * scale), int(height * scale)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            background.paste(img_resized, ((x_img - new_width) // 2, (y_img - new_height) // 2))
        return background

    def ocr_loop(self):
        last_processed_image = None
        while True:
            # Condition to run OCR
            ocr_should_run = not self.ocr_paused and (not self.is_only_parse_when_show or not self.tray_icon.visible)
            if not ocr_should_run:
                time.sleep(0.2)
                continue

            try:
                image = ImageGrab.grabclipboard()
                if image is None or isinstance(image, list) or image == last_processed_image:
                    time.sleep(0.1)
                    continue

                last_processed_image = image
                self.root.after(0, lambda: (self.text_box.delete(1.0, tk.END), self.log("正在识别...", end='')))
                
                self.current_image = image.convert("RGB")

                with io.BytesIO() as img_byte_arr:
                    self.current_image.save(img_byte_arr, format='PNG')
                    image_data = img_byte_arr.getvalue()
                
                def stream_callback(token: str) -> int:
                    self.root.after(0, self.log, token, '')
                    return 0

                result = self.gex_context.inference_mem_stream(image_data, stream_callback)
                self.root.after(0, self.log, '\n') # Final newline
                
                if self.check_repetition(result, 21):
                    self.log('===检测到重复输出!?===\n')
                    self.save_data(self.current_image, result, 'Repeat')
                    continue

                self.output = result
                
                processed_result = result.replace('\\[', '\\begin{align*}').replace('\\]', '\\end{align*}').replace('%', '\\%')
                if self.convert_align_to_equations_enabled:
                    processed_result = self.convert_align_to_equations(processed_result)
                if self.use_dollars_for_inline_math:
                    processed_result = processed_result.replace('\\(', '$').replace('\\)', '$')

                pyperclip.copy(processed_result)
                self.log('===成功复制到剪切板===\n')

            except Exception as e:
                self.log(f"\nOCR 错误: {e}\n")
                time.sleep(1)

    def toggle_ocr(self, event=None):
        self.ocr_paused = not self.ocr_paused
        self.root.after(0, self.update_icon)

    def update_icon(self):
        if self.ocr_paused:
            new_icon = self.load_scaled_image(os.path.join(base_path, "icon_gray.png"))
        else:
            new_icon = self.load_scaled_image(os.path.join(base_path, "icon.png"))
        self.icon = new_icon
        self.icon_tk = ImageTk.PhotoImage(self.icon)
        self.icon_label.config(image=self.icon_tk)
        self.tray_icon.icon = self.icon

    def log(self, message, end='\n'):
        self.text_box.insert(tk.END, message + end)
        self.text_box.see(tk.END)

if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = MixTeXApp(root)
        root.mainloop()
    except Exception as e:
        with open('error_log.txt', 'w', encoding='utf-8') as f:
            import traceback
            f.write(str(e) + '\n')
            f.write(traceback.format_exc())
        ctypes.windll.user32.MessageBoxW(0, f"程序启动失败: {str(e)}\n详细信息已保存到 error_log.txt", "严重错误", 0)
这个也要改
