import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont

# ===== 配置参数 =====
OUTPUT_DIR = "generated_math"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 支持的运算符（注意：乘除用 × ÷ 更贴近手写/教材）
OPS = ['+', '-', '×', '÷']
DIGITS = list('0123456789')

# 字体路径（请替换为你本地存在的 .ttf 文件）
FONT_PATHS = [
    "/System/Library/Fonts/Arial.ttf",          # macOS
    "C:/Windows/Fonts/arial.ttf",               # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    # 可添加更多，或使用绝对路径
]

# 过滤掉不存在的字体
FONT_PATHS = [f for f in FONT_PATHS if os.path.exists(f)]

if not FONT_PATHS:
    raise FileNotFoundError("未找到可用字体文件，请检查 FONT_PATHS")

# ===== 辅助函数：生成合法四则运算表达式 =====
def generate_expression(max_depth=2, max_num=100):
    """递归生成带括号的四则运算表达式"""
    def _gen(depth):
        if depth <= 0 or random.random() < 0.4:
            # 返回数字
            return str(random.randint(0, max_num))
        else:
            left = _gen(depth - 1)
            right = _gen(depth - 1)
            op = random.choice(OPS)
            # 随机加括号（避免太多）
            if random.random() < 0.3:
                return f"({left} {op} {right})"
            else:
                return f"{left} {op} {right}"
    
    expr = _gen(max_depth)
    # 移除空格（或保留，根据需求）
    expr = expr.replace(' ', '')
    # 确保不以运算符开头/结尾
    if expr[0] in '×÷' or expr[-1] in '+-×÷':
        return generate_expression(max_depth, max_num)
    return expr

# ===== 渲染表达式为图像 =====
def render_expression_to_image(expr, img_size=(320, 64)):
    # 创建白色背景图像（PIL）
    img_pil = Image.new("RGB", img_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)

    # 随机选择字体和大小
    font_path = random.choice(FONT_PATHS)
    font_size = random.randint(24, 40)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()

    # 计算文本位置（居中）
    bbox = draw.textbbox((0, 0), expr, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img_size[0] - text_width) // 2
    y = (img_size[1] - text_height) // 2

    # 随机颜色（深色文字）
    text_color = (
        random.randint(0, 50),
        random.randint(0, 50),
        random.randint(0, 50)
    )
    draw.text((x, y), expr, fill=text_color, font=font)

    # 转为 OpenCV 格式 (BGR)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # === 可选：添加噪声/模糊/旋转 ===
    if random.random() < 0.3:
        # 高斯模糊
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
    if random.random() < 0.2:
        # 添加高斯噪声
        noise = np.random.normal(0, 5, img_cv.shape).astype(np.uint8)
        img_cv = cv2.add(img_cv, noise)

    return img_cv

# ===== 主生成循环 =====
def main(num_samples=1000):
    for i in range(num_samples):
        expr = generate_expression(max_depth=2, max_num=99)
        # 可选：加上 "= ?"
        full_expr = expr + " = ?"

        img = render_expression_to_image(full_expr, img_size=(416, 64))

        # 保存图像
        img_path = os.path.join(OUTPUT_DIR, f"math_{i:05d}.jpg")
        cv2.imwrite(img_path, img)

        # 保存标签（用于训练时读取）
        label_path = os.path.join(OUTPUT_DIR, f"math_{i:05d}.txt")
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(full_expr)

        if i % 100 == 0:
            print(f"已生成 {i}/{num_samples}")

    print("✅ 生成完成！")

if __name__ == "__main__":
    main(num_samples=1000)