# hyper CV

计算机视觉实验与项目集合，涵盖目标检测、图像标注、数据预处理和相机采集。

## 目录结构

```
CV/
├── X-AnyLabeling/   # AI 辅助图像标注工具
├── YOLO/             # Ultralytics YOLO 目标检测（训练/推理/部署）
├── NanoDet/          # 超轻量 Anchor-Free 目标检测（多后端部署）
├── camera/           # C++ 摄像头采集与人脸检测
├── data/             # 训练数据与预处理脚本
└── 5060的yolo配置.txt # 本机 GPU 环境信息
```

## 环境配置

| 组件 | 版本 |
|------|------|
| Python | 3.12.12 |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 |
| ultralytics | 8.3.239 |
| OpenCV | 4.12.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8GB) |

**Conda 环境：**
- `pytorch312` — YOLO 训练/推理主环境
- `xany` — X-AnyLabeling 标注工具

```bash
# YOLO 环境（使用前激活）
conda activate pytorch312

# X-AnyLabeling 环境
conda activate xany
```

---

## YOLO — 目标检测训练与推理

基于 Ultralytics YOLO，支持自定义数据集训练、模型导出和实时推理。

### 快速开始

```bash
conda activate pytorch312
cd PC/CV/YOLO
pip install -r requirements.txt
```

### 推理

```bash
# 图片/视频预测
yolo predict model=weights/yolov8s.pt source=<图片或视频路径> device=0

# 使用自训练权重
yolo predict model=weights/math11.pt source=<图片路径> device=0
```

### 训练

```bash
# 使用预配置的 yaml 训练
python src/train.py      # 主训练脚本
python src/train_small.py # 小模型/快速实验
```

数据集配置文件位于 `yaml/` 目录（`math.yaml`、`task.yaml`）。

### 模型导出

```bash
# 导出 ONNX
yolo export model=weights/math11.pt format=onnx imgsz=1080
yolo export model=weights/math12.pt format=onnx imgsz=1080 opset=21 half=False
```

### 主要脚本（`src/`）

| 脚本 | 用途 |
|------|------|
| `train.py` / `train_small.py` | 模型训练 |
| `predict_test.py` | 批量预测/测试 |
| `camera.py` / `capture.py` | 实时摄像头推理 |
| `labeler.py` | 辅助标注工具 |
| `task.py` / `test.py` | 任务特定推理 |
| `MATH.PY` | 数学相关检测 |

---

## X-AnyLabeling — AI 辅助图像标注

数据标注工具，支持多种 AI 模型辅助标注，大幅提升标注效率。

```bash
conda activate xany
cd PC/CV/X-AnyLabeling
python anylabeling_app.py
```

- 支持矩形框、多边形、关键点等多种标注形式
- 内置 YOLO/SAM 等模型进行自动预标注
- 输出格式兼容 YOLO、COCO、VOC 等主流训练格式

---

## NanoDet — 超轻量目标检测

面向移动端和嵌入式的 Anchor-Free 检测模型，支持 ncnn、MNN、OpenVINO、LibTorch 等多后端部署。

- **模型体积：** 仅 980KB (INT8) / 1.8MB (FP16)
- **移动端速度：** 97fps (ARM CPU)
- **精度：** 最高 34.3 mAP@0.5:0.95

```bash
# 安装
pip install -r requirements.txt
python setup.py develop

# PyTorch 推理
python demo/demo.py image --config config/nanodet-plus-m_416.yml --model <权重路径> --path <图片路径>

# 导出 ONNX → 转换为 ncnn/MNN/OpenVINO
python tools/export_onnx.py --cfg_path config/nanodet-plus-m_416.yml --model_path <权重路径>
```

详见 `NanoDet/README.md`。

---

## camera — C++ 摄像头采集

基于 OpenCV C++ 的摄像头工具。

| 文件 | 说明 |
|------|------|
| `camera.cpp` / `camera.exe` | 调用电脑摄像头 |
| `face_detection.cpp` / `face_detection.exe` | 人脸检测 |
| `img.py` | Python 图像处理辅助脚本 |

编译需要 OpenCV C++ 库，已提供预编译的 `.exe`。

---

## data — 训练数据管理

```
data/
├── img/
│   ├── math/    # 数学相关检测图片
│   └── task/    # 任务检测图片
├── src/
│   ├── A4.py        # A4 纸检测辅助
│   ├── MP4toJPG.py  # 视频抽帧
│   ├── img.py       # 图像批处理
│   └── split.py     # 数据集划分（train/val/test）
└── yaml/
    ├── math.yaml    # math 数据集配置
    └── task.yaml    # task 数据集配置
```

### 常用数据流程

```bash
# 1. 视频抽帧
python data/src/MP4toJPG.py

# 2. 数据集划分
python data/src/split.py

# 3. 进入 X-AnyLabeling 标注
conda activate xany && python X-AnyLabeling/anylabeling_app.py

# 4. YOLO 训练
conda activate pytorch312 && cd YOLO && python src/train.py
```

---

## 硬件信息

- **GPU：** NVIDIA GeForce RTX 5060 Laptop GPU
- **CUDA：** 12.8（PyTorch 编译）/ 12.9（系统 nvcc）
- **cuDNN：** 9.1.0

完整环境检测见 `5060的yolo配置.txt`。
