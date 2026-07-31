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
| **路径** | `C:\Users\23615\Desktop\.hyper\PC\CV\YOLO` |

## 激活环境

### Git Bash
```bash
conda activate cv
PYTHONIOENCODING=utf-8 python script.py
```

### PowerShell
```powershell
# 先激活环境，再运行
conda activate cv
$env:PYTHONIOENCODING="utf-8"
python PC\CV\YOLO\src\train.py
```

## 环境检测

### Git Bash
```bash
conda activate cv
PYTHONIOENCODING=utf-8 python PC/CV/YOLO/src/test.py
```

### PowerShell
```powershell
conda activate cv
$env:PYTHONIOENCODING="utf-8"; python PC/CV/YOLO/src/test.py
```

## 预测

```bash
conda activate cv
python -c "
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

### Git Bash
```bash
conda activate cv
PYTHONIOENCODING=utf-8 python PC/CV/YOLO/src/train.py
```

### PowerShell
```powershell
conda activate cv
$env:PYTHONIOENCODING="utf-8"
python PC/CV/YOLO/src/train.py
```
