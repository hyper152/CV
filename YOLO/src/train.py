from ultralytics import YOLO  # 导入Ultralytics的YOLOv8核心类
import os                     # 导入os库，用于路径处理和文件存在性判断

# ===================== 全局配置：路径与权重设置 =====================
# 1. 你的data.yaml（保持原有路径不变）
DATA = r"C:\Users\23615\Desktop\.hyper\CV\YOLO\yaml\math.yaml"

# 2. 权重路径
WEIGHTS_DIR = r"C:\Users\23615\Desktop\.hyper\CV\YOLO\weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
WEIGHTS = os.path.join(WEIGHTS_DIR, 'yolov8s.pt')

# ===================== 主训练逻辑 =====================
if __name__ == '__main__':
    # 加载权重（本地优先，否则自动下载）
    model_path = WEIGHTS if os.path.exists(WEIGHTS) else 'yolov8s.pt'
    print(f'加载权重：{model_path}')
    
    # 加载YOLOv8模型
    model = YOLO(model_path)
    
    # 启动训练（核心适配1080p，调整imgsz+优化显存相关参数）
    model.train(
        data=DATA,               # 指向正确的data.yaml
        epochs=30,               # 可根据效果调整，1080p特征更丰富，可适当减少
        imgsz=640,               # 适配1080p图片，核心修改点！
        batch=1,                 # 1080p分辨率高，显存占用大，从4降至2（Windows更稳定）
        device=0,                # GPU训练，无GPU改'cpu'（cpu训练1080p会很慢）
        workers=0,               # Windows必须设0，避免多进程报错
        pretrained=True,         # 小数据集必备，迁移学习
        augment=True,            # 数据增强，提升泛化能力
        amp=True,                # 开启混合精度训练！1080p必开，大幅降低显存占用（原代码写反了注释）
        patience=10,             # 早停机制，防止过拟合（可选，实用优化）
        close_mosaic=10          # 关闭马赛克增强（1080p高分辨率，马赛克会破坏细节，可选）
    )