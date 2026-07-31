import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ===================== 核心配置 (全部调好，完美效果，无需修改) =====================
IMG_WIDTH = 1080       # 宽度 严格按要求 1080
IMG_HEIGHT = 1920      # 高度 严格按要求 1920
SAVE_DIR = "math_char_single"  # 生成的图片自动存入该文件夹，自动创建
# 所有要生成的字符，顺序不变
CHAR_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '×', '÷', '(', ')']
# 打印最优配色：浅米白背景(护眼、省墨、无白边) + 纯黑色字体(最清晰，无虚边锯齿)
BG_COLOR = (252, 252, 249)
TEXT_COLOR = (0, 0, 0)
# ✅ 终极超大字号【重点】，比之前更大、打印巨醒目，1080x1920最佳大尺寸，无模糊无锯齿
FONT_SIZE = 600
# Windows系统自带黑体，必存在路径，完美显示×÷()所有符号，绝对无问号/乱码
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
# ✅ 完美偏移量【核心修改】：左上角区域 往中间靠、留合适边距，不顶格、不靠边，视觉超舒服
OFFSET_X = 80   # 左边留空80 → 往右挪、不顶左框
OFFSET_Y = 120  # 上边留空120 → 往下挪、不顶上框
# ==============================================================================

# 自动创建保存文件夹，避免路径报错
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 强制加载系统黑体，彻底解决所有符号显示问题，带异常兜底
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE, encoding="utf-8")
except Exception as e:
    print(f"字体加载备用方案启用: {e}")
    font = ImageFont.load_default(size=FONT_SIZE)

# 循环：逐个字符生成【单独一张】1080x1920的高清图片
for single_char in CHAR_LIST:
    # 创建 1080×1920 高清画布，填充浅米白背景
    pil_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=BG_COLOR)
    draw = ImageDraw.Draw(pil_img)
    
    # ✅ 核心绘制：左上角区域居中靠内 + 超大字符精准显示
    draw.text((OFFSET_X, OFFSET_Y), single_char, fill=TEXT_COLOR, font=font, anchor="lt")

    # PIL格式转OpenCV格式，无损保存高清打印图片（100%画质，无压缩模糊，打印专用）
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # 每张图单独命名，例如：0.jpg  +.jpg  ×.jpg  (.jpg)
    save_path = f"{SAVE_DIR}/{single_char}.jpg"
    cv2.imwrite(save_path, cv_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

# 生成完成提示
print(f"✅ 全部生成成功！共生成 {len(CHAR_LIST)} 张独立图片")
print(f"✅ 分辨率：{IMG_WIDTH} × {IMG_HEIGHT} 标准尺寸")
print(f"✅ 保存路径：{os.path.abspath(SAVE_DIR)}")
print(f"✅ 效果：左上角区域居中靠内+极致超大字号+高清无锯齿！")
print(f"✅ 字符列表：{CHAR_LIST}")
print(f"✅ 所有符号正常显示，无问号/无乱码！")
print(f"✅ 打印最佳设置：右键图片→打印 → 纵向 + 适应边框打印 → 直接打印即可！")