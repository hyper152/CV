# YOLO 运行指南

## 环境

| 项目 | 内容 |
|------|------|
| **Conda 环境** | `cv` |
| **Python** | 3.10.20 |
| **PyTorch** | 2.4.1+cu124 (CUDA 12.4) |
| **GPU** | NVIDIA GeForce RTX 2060 SUPER |
| **cuDNN** | 9.1.0 |
| **Ultralytics** | 8.4.105 |
| **路径** | `C:\Users\23615\Desktop\.hyper\CV\YOLO` |

## 激活环境

### Git Bash
```bash
# 直接使用完整 python 路径
PYTHONIOENCODING=utf-8 "C:/Users/23615/.conda/envs/cv/python.exe" script.py

# 或 conda 激活后运行
conda activate cv
PYTHONIOENCODING=utf-8 python script.py
```

### PowerShell
```powershell
# 先设置编码，再运行
$env:PYTHONIOENCODING="utf-8"
C:/Users/23615/.conda/envs/cv/python.exe CV\YOLO\src\test.py

# 或 conda 激活后
conda activate cv
$env:PYTHONIOENCODING="utf-8"
python CV\YOLO\src\test.py
```

## 环境检测

### Git Bash
```bash
PYTHONIOENCODING=utf-8 "C:/Users/23615/.conda/envs/cv/python.exe" CV\YOLO\src\test.py
```

### PowerShell
```powershell
$env:PYTHONIOENCODING="utf-8"; "C:/Users/23615/.conda/envs/cv/python.exe" CV\YOLO\src\test.py
```

## 预测

```bash
# 使用 GPU (device=0)
yolo predict model=weights/yolov8s.pt source="path/to/video_or_image" device=0

# 或通过 python
PYTHONIOENCODING=utf-8 -c "
from ultralytics import YOLO
model = YOLO('weights/yolov8s.pt')
results = model.predict(source='path/to/video', device=0, save=True)
"
```

## 导出模型

```bash
yolo export model=weights/math11.pt format=onnx imgsz=1080
yolo export model=weights/math12.pt format=onnx imgsz=1080 opset=21 half=False
```

## 训练

```bash
$env:PYTHONIOENCODING="utf-8"
& "C:/Users/23615/.conda/envs/cv/python.exe" CV/YOLO/src/train.py
```
