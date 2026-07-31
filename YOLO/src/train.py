from ultralytics import YOLO
import os, random, shutil
from pathlib import Path

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CV/YOLO/
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", "data", "img", "steelball"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# ===================== 数据集自动拆分 =====================
def auto_split(val_ratio=0.1, seed=None):
    """
    从 images/（所有图片平铺）和 labels/ 中自动随机拆分训练/验证集。
    每次运行都会清空旧的 split 子目录并重新随机拆分。
    """
    valid = []
    for f in os.listdir(IMAGES_DIR):
        if f.lower().endswith(".jpg"):
            stem = Path(f).stem
            if os.path.exists(os.path.join(LABELS_DIR, f"{stem}.txt")):
                valid.append(stem)

    if not valid:
        raise RuntimeError(f"未找到有对应标签的 .jpg 图片，请检查 {IMAGES_DIR}")

    rng = random.Random(seed)
    rng.shuffle(valid)

    split = int(len(valid) * (1 - val_ratio))
    train_stems, val_stems = valid[:split], valid[split:]

    # 清空并重建 split 目录
    for subdir in ("train", "val"):
        for d in (os.path.join(IMAGES_DIR, subdir), os.path.join(LABELS_DIR, subdir)):
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

    for stem in train_stems:
        shutil.copy2(os.path.join(IMAGES_DIR, f"{stem}.jpg"), os.path.join(IMAGES_DIR, "train", f"{stem}.jpg"))
        shutil.copy2(os.path.join(LABELS_DIR, f"{stem}.txt"), os.path.join(LABELS_DIR, "train", f"{stem}.txt"))
    for stem in val_stems:
        shutil.copy2(os.path.join(IMAGES_DIR, f"{stem}.jpg"), os.path.join(IMAGES_DIR, "val", f"{stem}.jpg"))
        shutil.copy2(os.path.join(LABELS_DIR, f"{stem}.txt"), os.path.join(LABELS_DIR, "val", f"{stem}.txt"))

    print(f"数据集拆分: {len(train_stems)} 训练 + {len(val_stems)} 验证 = {len(valid)} 张")
    return len(train_stems), len(val_stems)

# ===================== 清理 =====================
def cleanup_split():
    """删除 auto_split 生成的 train/val 子目录，恢复平铺原样"""
    for subdir in ("train", "val"):
        for d in (os.path.join(IMAGES_DIR, subdir), os.path.join(LABELS_DIR, subdir)):
            if os.path.exists(d):
                shutil.rmtree(d)
    print("已清理 auto_split 生成的 train/val 目录")

# ===================== 训练 =====================
if __name__ == "__main__":
    try:
        # 1. 随机拆分
        auto_split(val_ratio=0.2, seed=None)   # seed=None 每次随机不同

        # 2. data.yaml
        DATA = os.path.realpath(os.path.join(BASE_DIR, "..", "data", "yaml", "steelball.yaml"))

        # 3. 权重
        WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        WEIGHTS = os.path.join(WEIGHTS_DIR, "yolov8n.pt")

        model_path = WEIGHTS if os.path.exists(WEIGHTS) else "yolov8n.pt"
        print(f"加载权重：{model_path}")

        model = YOLO(model_path)

        model.train(
            # ---------- 基础 ----------
            data=DATA,
            epochs=200,             # 200 epoch 给够学习时间，配合高 lr 充分探索
            imgsz=640,              # 640 训练让小目标有足够纹理，避免 320 下学成"黑斑检测器"
            batch=16,               # v8n+640 轻量
            device=0,
            workers=4,
            seed=42,
            cache="disk",
            pretrained=True,
            amp=True,

            # ---------- 优化器 ----------
            optimizer="AdamW",
            cos_lr=True,
            lr0=0.002,              # 提高初始学习率，2 个 epoch 收敛说明太保守
            lrf=0.05,               # 最终 lr 因子从 0.01 提高到 0.05，维持更高学习率防早滞
            warmup_epochs=3,
            patience=50,            # 拉长耐心，防止学习率尚有空间时误触发早停
            weight_decay=0.0002,    # 减半 weight decay，降低正则化强度

            # ---------- 数据增强（小目标场景收窄空间扰动，加强实例粘贴） ----------
            # 色彩增强：保留，金属反光/锈蚀场景适用
            hsv_h=0.03,
            hsv_s=0.7,
            hsv_v=0.4,
            # 空间增强：收紧，避免小目标被放大/缩小后消失
            degrees=10,
            translate=0.1,          # 减少平移幅度，防止小目标移出视野
            scale=0,                # 禁用缩放增强，小目标一旦缩小即不可检测
            shear=2,
            flipud=0.1,
            fliplr=0.5,
            # 混合增强：减少 mosaic，小目标在 mosaic 四合一图中只剩 1/4 面积
            mosaic=0.5,             # 从 0.8 降到 0.5，减少小目标被压缩
            mixup=0.15,
            copy_paste=0.3,         # 提高到 0.3，对稀少的小目标注入更多正样本
            close_mosaic=10,        # 提前关闭 mosaic，让模型在真实分辨率多训 5 个 epoch
            erasing=0.1,

            # ---------- 检测 ----------
            conf=0.001,
            iou=0.5,
            max_det=150,            # 640 分辨率下视野更大，适当提高上限

            # ---------- 保存 ----------
            val=True,
            save=True,
            save_period=10,
            project=os.path.join(BASE_DIR, "runs"),
            name="steelball",
            plots=True,             # 生成训练曲线图
        )
    finally:
        cleanup_split()
