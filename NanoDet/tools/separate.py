import json
import os
import shutil
import random
from typing import Set

def split_coco_data_8_to_2(
    original_coco_json: str,
    original_image_dir: str,
    output_task_dir: str,
    val_split_ratio: float = 0.2,
    random_seed: int = 42  # 固定随机种子，保证拆分结果可复现
) -> None:
    """
    无净化版：按 8:2 比例拆分 COCO 注释和对应图片，生成训练集/验证集
    :param original_coco_json: 原始单个 COCO 注释文件（task-coco.json）
    :param original_image_dir: 原始训练图片目录（train）
    :param output_task_dir: 目标 task 目录（存放拆分后的 images/annotations）
    :param val_split_ratio: 验证集抽取比例（默认 0.2）
    :param random_seed: 随机种子（保证拆分结果可复现）
    """
    # 1. 配置路径参数
    output_images_dir = os.path.join(output_task_dir, "images")
    output_train_image_dir = original_image_dir  # 原始train目录作为最终训练图片目录
    output_val_image_dir = os.path.join(output_images_dir, "val")  # 自动创建验证图片目录
    output_annotations_dir = os.path.join(output_task_dir, "annotations")
    
    # 2. 检查输入文件/目录是否存在
    if not os.path.exists(original_coco_json):
        raise FileNotFoundError(f"原始 COCO 注释不存在：{original_coco_json}")
    if not os.path.exists(original_image_dir):
        raise NotADirectoryError(f"原始训练图片目录不存在：{original_image_dir}")
    
    # 3. 自动创建所需目录（val图片目录、annotations目录）
    os.makedirs(output_val_image_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)
    
    # 4. 加载原始 COCO 注释文件（仅读取，不修改原始数据）
    with open(original_coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 5. 准备图片数据：提取所有图片信息，并随机拆分（固定种子保证可复现）
    all_images = coco_data.get('images', [])
    if not all_images:
        raise ValueError("原始 COCO 注释中无图片数据，无法拆分")
    
    random.seed(random_seed)
    random.shuffle(all_images)  # 打乱图片列表
    
    # 计算拆分边界，抽取 20% 作为验证集
    val_split_count = int(len(all_images) * val_split_ratio)
    val_images = all_images[:val_split_count]
    train_images = all_images[val_split_count:]
    
    # 6. 提取训练/验证集图片 ID 和文件名（用于后续筛选注释、复制图片）
    train_image_ids: Set[int] = set(img.get('id', -1) for img in train_images)
    val_image_ids: Set[int] = set(img.get('id', -1) for img in val_images)
    val_image_filenames: Set[str] = set(img.get('file_name', '') for img in val_images)
    
    # 7. 筛选训练/验证集对应的注释（仅按图片 ID 匹配，无额外净化）
    train_annotations = []
    val_annotations = []
    for ann in coco_data.get('annotations', []):
        img_id = ann.get('image_id', -1)
        if img_id in train_image_ids:
            train_annotations.append(ann)
        elif img_id in val_image_ids:
            val_annotations.append(ann)
    
    # 8. 复制 20% 验证集图片到 val 目录（保留原始图片，仅复制）
    copied_val_image_count = 0
    for img_filename in val_image_filenames:
        src_img_path = os.path.join(original_image_dir, img_filename)
        dst_img_path = os.path.join(output_val_image_dir, img_filename)
        
        # 仅复制存在的图片（避免路径错误）
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
            copied_val_image_count += 1
    
    # 9. 构建拆分后的 COCO 注释数据（保留原始所有额外信息，仅拆分图片和注释）
    base_coco_info = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data.get('categories', [])
    }
    
    train_coco_data = {**base_coco_info, 'images': train_images, 'annotations': train_annotations}
    val_coco_data = {**base_coco_info, 'images': val_images, 'annotations': val_annotations}
    
    # 10. 保存拆分后的注释文件（规范命名，适配 NanoDet）
    train_annotation_output = os.path.join(output_annotations_dir, "instances_train.json")
    val_annotation_output = os.path.join(output_annotations_dir, "instances_val.json")
    
    with open(train_annotation_output, 'w', encoding='utf-8') as f:
        json.dump(train_coco_data, f, ensure_ascii=False, indent=2)
    
    with open(val_annotation_output, 'w', encoding='utf-8') as f:
        json.dump(val_coco_data, f, ensure_ascii=False, indent=2)
    
    # 11. 输出拆分统计信息
    print("="*50)
    print("COCO 数据 8:2 拆分完成（无净化，保留全部原始信息）！")
    print("="*50)
    print(f"原始总图片数：{len(all_images)}")
    print(f"训练集图片数：{len(train_images)}（{1-val_split_ratio:.1%}）| 注释数：{len(train_annotations)}")
    print(f"验证集图片数：{len(val_images)}（{val_split_ratio:.1%}）| 注释数：{len(val_annotations)}")
    print(f"验证集图片复制完成：{copied_val_image_count}/{len(val_image_filenames)} 张")
    print("-"*50)
    print(f"训练注释保存至：{train_annotation_output}")
    print(f"验证注释保存至：{val_annotation_output}")
    print(f"验证图片保存至：{output_val_image_dir}")

# ---------------------- 基于你的实际路径配置（无需修改） ----------------------
if __name__ == "__main__":
    # 配置你的实际文件/目录路径
    ORIGINAL_COCO_JSON = r"C:\Users\hyper\Desktop\program\.Gsing\.data\task-coco.json"
    ORIGINAL_TRAIN_IMAGE_DIR = r"C:\Users\hyper\Desktop\program\.Gsing\.data\task\images\train"
    OUTPUT_TASK_DIR = r"C:\Users\hyper\Desktop\program\.Gsing\.data\task"
    
    # 执行 8:2 拆分操作
    try:
        split_coco_data_8_to_2(
            original_coco_json=ORIGINAL_COCO_JSON,
            original_image_dir=ORIGINAL_TRAIN_IMAGE_DIR,
            output_task_dir=OUTPUT_TASK_DIR,
            val_split_ratio=0.2
        )
    except Exception as e:
        print(f"\n拆分出错：{str(e)}")