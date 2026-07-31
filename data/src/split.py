#python "C:\Users\hyper\Desktop\program\.Gsing\.data\split.py" --img_dir "C:\Users\hyper\Desktop\program\.Gsing\.data\task\images\train" --ann_path "C:\Users\hyper\Desktop\program\.Gsing\.data\task-coco.json" --val_ratio 0.2 --output_dir "C:\Users\hyper\Desktop\program\.Gsing\.data\task\split_dataset"


import json
import os
import shutil
from sklearn.model_selection import train_test_split
import argparse

def split_coco_dataset(
    img_dir,          # 原始图片文件夹路径
    ann_path,         # 原始COCO标注文件路径
    val_ratio=0.2,    # 验证集比例（默认20%）
    output_dir="split_dataset"  # 拆分后保存的目录
):
    """
    自动拆分COCO数据集为训练集和验证集
    """
    # 1. 创建输出目录
    train_img_dir = os.path.join(output_dir, "images/train")
    val_img_dir = os.path.join(output_dir, "images/val")
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # 2. 读取原始标注文件
    with open(ann_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # 3. 提取所有图片ID和文件名，按图片拆分（避免同一图片的标注被拆分）
    img_ids = [img["id"] for img in coco_data["images"]]
    # 随机拆分训练/验证图片ID（stratify=None表示随机拆分）
    train_img_ids, val_img_ids = train_test_split(
        img_ids, test_size=val_ratio, random_state=42  # random_state固定拆分结果
    )

    # 4. 筛选训练集/验证集的图片和标注
    # 训练集
    train_images = [img for img in coco_data["images"] if img["id"] in train_img_ids]
    train_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in train_img_ids]
    # 验证集
    val_images = [img for img in coco_data["images"] if img["id"] in val_img_ids]
    val_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in val_img_ids]

    # 5. 生成拆分后的标注文件
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data["categories"]  # 类别不变
    }
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data["categories"]
    }

    # 6. 保存标注文件
    train_ann_path = os.path.join(ann_dir, "train.json")
    val_ann_path = os.path.join(ann_dir, "val.json")
    with open(train_ann_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, indent=2)
    with open(val_ann_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, indent=2)

    # 7. 复制图片到对应文件夹
    print("开始复制图片...")
    for img in train_images:
        src_img = os.path.join(img_dir, img["file_name"])
        dst_img = os.path.join(train_img_dir, img["file_name"])
        shutil.copy(src_img, dst_img)
    
    for img in val_images:
        src_img = os.path.join(img_dir, img["file_name"])
        dst_img = os.path.join(val_img_dir, img["file_name"])
        shutil.copy(src_img, dst_img)

    # 8. 打印拆分结果
    print(f"数据集拆分完成！")
    print(f"训练集：{len(train_images)} 张图片，{len(train_annotations)} 个标注")
    print(f"验证集：{len(val_images)} 张图片，{len(val_annotations)} 个标注")
    print(f"输出目录：{output_dir}")
    print(f"训练标注：{train_ann_path}")
    print(f"验证标注：{val_ann_path}")

if __name__ == "__main__":
    # 命令行参数（方便运行）
    parser = argparse.ArgumentParser(description="自动拆分COCO数据集为train/val")
    parser.add_argument("--img_dir", required=True, help="原始图片文件夹路径")
    parser.add_argument("--ann_path", required=True, help="原始COCO标注文件路径")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例，默认0.2")
    parser.add_argument("--output_dir", default="split_dataset", help="输出目录")
    args = parser.parse_args()

    # 执行拆分
    split_coco_dataset(
        img_dir=args.img_dir,
        ann_path=args.ann_path,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir
    )