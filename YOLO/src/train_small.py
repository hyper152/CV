from ultralytics import YOLO
import os
import torch

# ===================== 路径配置（仅需修改这1行！）=====================
DATA = r"C:\Users\hyper\Desktop\program\.Gsing\YOLO\yaml\math.yaml"  # 你的数据集yaml
# ======================================================================
MODEL = "yolov8n.pt"  # yolov8n是小显存首选，轻量且适配迁移学习
SAVE_DIR = r"C:\Users\hyper\Desktop\program\.Gsing\YOLO\runs\train"  # 训练结果保存路径
# 训练前强制清空GPU显存（解决Windows显存残留问题，核心！）
torch.cuda.empty_cache()

# 创建保存目录，避免Windows路径不存在报错
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载预训练模型（YOLO11兼容YOLOv8权重，迁移学习核心）
model = YOLO(MODEL)

# 开始训练：1080p高分辨率显存适配 + 800张小数据集专属优化 + Windows完美适配
results = model.train(
    # 🔥 核心显存优化：1080p保留 + 8G显存稳定运行（4G显存仅需把batch改1）
    data=DATA,               # 数据集yaml路径（唯一需修改的地方）
    imgsz=1080,              # 保留1080p高分辨率，保证数学公式/数字细节检测
    batch=1,                 # 8G显存设1（核心！原2必爆显存，1是1080p的安全值）
    device=0,                # GPU训练，无GPU改'cpu'（CPU训练1080p会慢，建议GPU）
    workers=0,               # Windows必设0，避免多进程数据集加载报错
    amp=True,                # 混合精度训练保留，1080p下大幅降低显存占用（配合下面参数无碎片）
    project=SAVE_DIR,        # 训练结果保存根目录
    name="math_train",       # 训练任务名，便于区分
    close_mosaic=0,          # 🔥 全程关闭马赛克（核心显存优化），1080p下马赛克既毁细节又爆显存
    
    # 📊 800张小数据集核心优化：慢学+冻结+防过拟合（比原参数更适配小数据集）
    epochs=100,              # 原150偏多，800张100轮足够，减少过拟合风险
    lr0=0.0005,              # 原0.001再降半（小数据集终极慢学），避免权重震荡
    lrf=0.005,               # 最终学习率与lr0匹配（1/100），保持学习率衰减节奏
    patience=30,             # 放宽早停，让模型充分学习，YOLO11会自动保存best.pt
    pretrained=True,         # 必开，迁移学习是小数据集的根本
    freeze=15,               # 原10→15，冻结更多骨干层（yolov8n共22层），先拟合数学目标再微调
    warmup_epochs=5,         # 原3→5，加长热身，适应极低学习率，避免初始训练失效
    weight_decay=0.0005,     # 新增：权重衰减，抑制过拟合（小数据集必备）
    dropout=0.2,             # 新增：随机丢弃神经元，YOLO11兼容，大幅降低过拟合
    
    # 🎨 数学公式专属数据增强：轻量+不失真（比原参数更适配公式/数字）
    hsv_h=0.03,              # 原0.05→0.03，色相几乎不调，防止公式颜色失真
    hsv_s=0.08,              # 原0.1→0.08，饱和度小幅增强
    hsv_v=0.08,              # 原0.1→0.08，亮度小幅增强
    degrees=3,               # 原5→3，公式/数字不宜旋转，极小角度避免形状失真
    translate=0.03,          # 原0.05→0.03，小幅平移
    scale=0.08,              # 原0.1→0.08，小幅缩放
    flipud=0.0,              # 保留0，公式无上下翻转场景
    fliplr=0.4,              # 原0.5→0.4，适度左右翻转，避免过度增强
    mosaic=0.0,              # 关闭马赛克（与close_mosaic=0配合，双重保障）
    mixup=0.0,               # 新增：关闭混合增强，公式目标小，mixup会导致特征重叠
    
    # 📌 基础训练配置：YOLO11自动优化，保留核心
    conf=0.25,               # 低置信度阈值，让模型学习更多小目标（公式/数字多为小目标）
    iou=0.45,                # 匹配YOLO默认IOU阈值
    save=True,               # 保存last.pt，每轮更新
    val=True,                # 每轮验证，自动保存mAP最高的best.pt（小数据集的核心！）
    save_period=5,           # 新增：每5轮保存一次权重，避免频繁保存占用磁盘
)

# 训练完成后，自动用最优模型（best.pt）做最终验证，输出详细指标（适配新路径）
best_model_path = os.path.join(SAVE_DIR, "math_train", "weights", "best.pt")  # 原路径漏了任务名math_train
if os.path.exists(best_model_path):
    # 验证时也做显存适配，batch设1，避免验证阶段爆显存
    model.val(model=best_model_path, data=DATA, imgsz=1080, batch=1, workers=0, device=0)
else:
    print(f"最优模型未找到，路径：{best_model_path}")