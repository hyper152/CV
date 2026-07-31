import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ===================== 【可自定义配置项，重点修改分辨率】 =====================
# 1. 生成的总样本数量 (训练集+验证集)
TOTAL_NUM = 5000
# 2. 验证集占比
VAL_RATIO = 0.2
# 3. 【核心修改：高分辨率设置】支持 640x640 / 800x600 等任意尺寸
IMG_WIDTH, IMG_HEIGHT = 640, 640  # 改为640x640高清分辨率，可自行调整
# 4. 算式配置
MIN_NUM = 0
MAX_NUM = 199
MIN_LEN = 6
MAX_LEN = 10
# 5. 类别映射（含括号，共17类，顺序不变）
CATEGORIES = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "+", 11: "-", 12: "×", 13: "÷", 14: "=",
    15: "(", 16: ")"
}
# 6. 随机字体大小配置（按分辨率比例自动适配）
BASE_FONT_SCALE = 0.08  # 字体大小 = 图片高度 × 比例，0.08适配640x640
MIN_FONT_SCALE = 0.05   # 最小字体比例
MAX_FONT_SCALE = 0.12   # 最大字体比例
TARGET_WIDTH_RATIO = 0.8  # 文本宽度上限（图片宽度的80%）
# 7. 背景配置（按分辨率比例调整密度）
BG_COLOR_RANGE = [(245,245,245), (250,250,250)]
NOISE_PROB = 0.3
NOISE_DENSITY = 0.0008  # 高分辨率下降低密度，避免噪点过多
TEXTURE_PROB = 0.4
# =============================================================================

# 生成随机四则运算算式（带括号）
def generate_math_expr():
    ops = ['+', '-', '×', '÷']
    expr_parts = []
    expr_len = random.randint(MIN_LEN, MAX_LEN)
    has_bracket = random.choice([True, False])
    
    if has_bracket and expr_len >= 8:
        sub_num1 = str(random.randint(MIN_NUM, 99))
        sub_op = random.choice(ops)
        sub_num2 = str(random.randint(MIN_NUM, 99))
        sub_expr = f"({sub_num1}{sub_op}{sub_num2})"
        expr_parts.append(sub_expr)
        while len(''.join(expr_parts)) < expr_len - 1:
            expr_parts.append(random.choice(ops))
            expr_parts.append(str(random.randint(MIN_NUM, 99)))
    else:
        while len(''.join(expr_parts)) < expr_len - 1:
            expr_parts.append(str(random.randint(MIN_NUM, 99)))
            expr_parts.append(random.choice(ops))
    
    math_expr = ''.join(expr_parts) + '='
    return math_expr

# 生成带纹理/噪点的背景（按分辨率适配密度）
def generate_background():
    bg_color = tuple(random.randint(BG_COLOR_RANGE[0][i], BG_COLOR_RANGE[1][i]) for i in range(3))
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # 噪点数量按分辨率比例计算
    noise_count = int(IMG_WIDTH * IMG_HEIGHT * NOISE_DENSITY)
    if random.random() < NOISE_PROB and noise_count > 0:
        for _ in range(noise_count):
            x = random.randint(0, IMG_WIDTH-1)
            y = random.randint(0, IMG_HEIGHT-1)
            noise_color = (random.randint(200,230), random.randint(200,230), random.randint(200,230))
            draw.point((x, y), fill=noise_color)
    
    # 高斯模糊半径按分辨率比例调整
    blur_radius = IMG_WIDTH / 2000  # 640x640对应半径0.32，更自然
    if random.random() < TEXTURE_PROB:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return img

# 随机字体大小 + 高分辨率适配防溢出
def get_random_font(text):
    # 按图片高度比例计算字体大小，实现分辨率自适应
    min_font_size = int(IMG_HEIGHT * MIN_FONT_SCALE)
    max_font_size = int(IMG_HEIGHT * MAX_FONT_SCALE)
    init_font_size = random.randint(min_font_size, max_font_size)
    font_size = init_font_size
    font = None
    target_width = IMG_WIDTH * TARGET_WIDTH_RATIO

    while True:
        try:
            if os.name == 'nt':  # Windows
                font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
            else:  # Linux/Mac
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default(size=font_size)
        
        # 计算文本宽度
        test_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT))
        test_draw = ImageDraw.Draw(test_img)
        bbox = test_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= target_width:
            break
        elif font_size > min_font_size:
            font_size -= 2
        else:
            break
    
    return font, font_size

# 生成单张高分辨率图片+标注
def generate_img_and_anno(math_expr, save_path, img_id):
    img = generate_background()
    draw = ImageDraw.Draw(img)
    font, _ = get_random_font(math_expr)
    
    # 计算居中位置
    bbox = draw.textbbox((0, 0), math_expr, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    start_x = (IMG_WIDTH - text_w) // 2
    start_y = (IMG_HEIGHT - text_h) // 2
    
    annotations = []
    current_x = start_x
    char_id = 1

    for char in math_expr:
        char_bbox = draw.textbbox((current_x, start_y), char, font=font)
        char_w = char_bbox[2] - char_bbox[0]
        char_h = char_bbox[3] - char_bbox[1]
        
        draw.text((current_x, start_y), char, fill='black', font=font)
        
        if char_w > 2 and char_h > 2:
            cat_id = [k for k, v in CATEGORIES.items() if v == char][0]
            annotations.append({
                "id": char_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [current_x, start_y, char_w, char_h],
                "area": char_w * char_h,
                "iscrowd": 0,
                "segmentation": []
            })
            char_id += 1
        current_x += char_w

    img.save(save_path, quality=95)
    return annotations

# 初始化COCO标注
def init_coco_json():
    coco = {
        "info": {"description": f"High-Res MathExpr Dataset {IMG_WIDTH}x{IMG_HEIGHT}", "version": "1.0"},
        "licenses": [{"name": "MIT", "id": 1}],
        "categories": [],
        "images": [],
        "annotations": []
    }
    for cat_id, cat_name in CATEGORIES.items():
        coco["categories"].append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "math"
        })
    return coco

# 主函数
def main():
    train_img_dir = "math_dataset/train/images"
    val_img_dir = "math_dataset/val/images"
    train_anno_path = "math_dataset/train/annotations.json"
    val_anno_path = "math_dataset/val/annotations.json"
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    
    train_coco = init_coco_json()
    val_coco = init_coco_json()
    
    train_num = int(TOTAL_NUM * (1 - VAL_RATIO))
    val_num = TOTAL_NUM - train_num
    print(f"开始生成 {IMG_WIDTH}x{IMG_HEIGHT} 高分辨率数据集：")
    print(f"训练集{train_num}张 | 验证集{val_num}张 | 随机字体+背景+括号")

    # 生成训练集
    train_anno_list = []
    for img_id in range(1, train_num + 1):
        math_expr = generate_math_expr()
        img_name = f"math_train_{img_id:05d}.jpg"
        img_path = os.path.join(train_img_dir, img_name)
        anno = generate_img_and_anno(math_expr, img_path, img_id)
        train_anno_list.extend(anno)
        train_coco["images"].append({
            "id": img_id, "file_name": img_name, "width": IMG_WIDTH, "height": IMG_HEIGHT, "license":1
        })
        if img_id % 200 == 0:
            print(f"训练集进度：{img_id}/{train_num}")

    # 生成验证集
    val_anno_list = []
    for img_id in range(1, val_num + 1):
        math_expr = generate_math_expr()
        img_name = f"math_val_{img_id:05d}.jpg"
        img_path = os.path.join(val_img_dir, img_name)
        anno = generate_img_and_anno(math_expr, img_path, img_id)
        val_anno_list.extend(anno)
        val_coco["images"].append({
            "id": img_id, "file_name": img_name, "width": IMG_WIDTH, "height": IMG_HEIGHT, "license":1
        })
        if img_id % 200 == 0:
            print(f"验证集进度：{img_id}/{val_num}")

    # 保存标注
    train_coco["annotations"] = train_anno_list
    val_coco["annotations"] = val_anno_list
    with open(train_anno_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, indent=2, ensure_ascii=False)
    with open(val_anno_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, indent=2, ensure_ascii=False)

    print("="*60)
    print(f"✅ {IMG_WIDTH}x{IMG_HEIGHT} 高分辨率数据集生成完成！✅")
    print(f"📁 路径：{os.path.abspath('math_dataset')}")

if __name__ == "__main__":
    main()