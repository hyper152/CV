from ultralytics import YOLO
import cv2
import numpy as np

# ===================== 核心修改：改回你的数学符号训练权重 =====================
model = YOLO(r"C:\Users\hyper\Desktop\program\.Gsing\YOLO\weights\MATH11.pt")
# 你的数学符号类别名称（索引0-15和训练模型一致，保留）
class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '+', '-', 'x', '÷', '(', ')'
]

# ===================== 2. 摄像头初始化 =====================
cap = cv2.VideoCapture(1)  # 0=内置，1=外接（根据实际调整）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("错误：无法打开摄像头！")
    exit()
print("摄像头已启动，按 'q' 键退出识别...")

# ===================== 3. 自定义绘制参数 =====================
box_color = (0, 255, 0)    # 检测框：绿色
label_color = (255, 255, 255)  # 标签文字：白色（配绿色背景更清晰）
box_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2
label_padding = 5

# ===================== 4. 实时推理循环 =====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("错误：无法读取摄像头画面！")
        break

    results = model(
        source=frame,
        imgsz=960,
        device=0,  # 调用RTX5060 GPU
        conf=0.5,
        iou=0.5,
        verbose=False,
        max_det=50,
        agnostic_nms=True
    )

    # 手动绘制框+标签
    result = results[0]
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_idx = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        # 匹配你的数学符号类别名
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"未知_{cls_idx}"
        label_text = f"{cls_name} {conf:.2f}"

        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
        # 绘制带背景的标签
        (label_w, label_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        bg_x1, bg_y1 = x1, y1 - label_h - 2*label_padding
        bg_x2, bg_y2 = x1 + label_w + 2*label_padding, y1
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1)
        cv2.putText(frame, label_text, (x1+label_padding, y1-label_padding), 
                    font, font_scale, label_color, font_thickness)

    cv2.imshow("YOLO 数学符号实时识别（带标签）", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("摄像头已关闭，识别结束！")