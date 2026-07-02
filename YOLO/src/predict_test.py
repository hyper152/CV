from ultralytics import YOLO
import os

# 1. 加载预训练权重（指定 v8 环境下的权重路径）
model = YOLO(r"C:\Users\hyper\Desktop\program\.Gsing\YOLO\weights\yolov8s.pt")

# 2. 推理官方示例图片（bus.jpg 是 ultralytics 自带的测试图）
results = model.predict(
    source="img\cs.png",  # 测试图片路径
    imgsz=640,          # 推理图片尺寸（YOLOv8 默认）
    device=0,           # 0=使用 GPU，无 GPU 改为 'cpu'
    save=True           # 保存推理结果到 runs/detect/predict 文件夹
)

# 3. 打印推理结果（可选，查看识别到的目标）
for result in results:
    boxes = result.boxes  # 边界框信息
    print("识别到的目标：", boxes.cls)  # 输出目标类别（0=person, 5=bus 等）