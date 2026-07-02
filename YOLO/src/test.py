import sys
import importlib
import subprocess

def check_python_info():
    """检测Python基础环境信息"""
    print("="*60)
    print("        🐍 Python 基础环境检测        ")
    print("="*60)
    print(f"Python 版本          : {sys.version.split()[0]}")
    print(f"操作系统              : {sys.platform}")
    print(f"Python 执行路径       : {sys.executable}")

def check_pytorch_version():
    """单独检测PyTorch核心版本（突出显示）"""
    print("\n" + "="*60)
    print("        🤖 PyTorch 核心版本检测        ")
    print("="*60)
    try:
        import torch
        print(f"PyTorch 核心版本      : {torch.__version__}")
        print(f"torchvision 版本      : {torchvision.__version__ if 'torchvision' in sys.modules else importlib.import_module('torchvision').__version__}")
        print(f"PyTorch 编译CUDA版本  : {torch.version.cuda if torch.version.cuda else '无（CPU版）'}")
        print(f"PyTorch cuDNN 版本    : {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') and torch.cuda.is_available() else '不可用'}")
    except ImportError:
        print(f"PyTorch 核心版本      : ❌ 未安装PyTorch")
        print(f"torchvision 版本      : ❌ 未安装")
    except Exception as e:
        print(f"PyTorch检测异常       : {str(e)[:50]}")

def check_yolo_dependencies():
    """检测YOLO核心依赖库版本"""
    print("\n" + "="*60)
    print("        📦 YOLO 其他核心依赖检测        ")
    print("="*60)
    
    # 核心依赖库（排除已单独检测的torch/torchvision）
    deps_map = {
        "ultralytics": "ultralytics",
        "opencv-python": "cv2",
        "numpy": "numpy",
        "pillow": "PIL",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn"
    }
    
    for dep_name, import_name in deps_map.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "版本未知")
            print(f"{dep_name:<15} : {version}")
        except ImportError:
            print(f"{dep_name:<15} : 未安装")
        except Exception as e:
            print(f"{dep_name:<15} : 检测失败 - {str(e)[:20]}")

def check_cuda_info():
    """检测CUDA/GPU运行时信息"""
    print("\n" + "="*60)
    print("        🚀 CUDA/GPU 运行时检测        ")
    print("="*60)
    try:
        import torch
        
        # CUDA运行时可用性
        print(f"CUDA 运行时是否可用   : {'✅ 是' if torch.cuda.is_available() else '❌ 否'}")
        
        # 系统CUDA版本
        try:
            nvcc_output = subprocess.check_output(
                ["nvcc", "--version"], 
                stderr=subprocess.STDOUT, 
                text=True,
                timeout=5
            )
            for line in nvcc_output.split('\n'):
                if "release" in line and "V" in line:
                    sys_cuda_ver = line.strip().split("V")[-1].split(",")[0]
                    print(f"系统CUDA版本(nvcc)  : {sys_cuda_ver}")
                    break
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
            print(f"系统CUDA版本(nvcc)  : 未检测到（未安装/环境变量未配置）")
        
        # GPU详细信息
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"GPU 数量              : {gpu_count}")
            for i in range(gpu_count):
                print(f"GPU {i} 名称           : {torch.cuda.get_device_name(i)}")
        else:
            # Mac MPS加速检测
            if torch.backends.mps.is_available():
                print("Apple MPS 加速        : ✅ 可用（Mac苹果芯片）")
            
    except ImportError:
        print("PyTorch 未安装，无法检测CUDA信息")
    except Exception as e:
        print(f"CUDA检测异常          : {str(e)[:50]}")

def check_yolo_basic_function():
    """快速验证YOLO基础功能"""
    print("\n" + "="*60)
    print("        ✅ YOLO 功能快速验证        ")
    print("="*60)
    try:
        from ultralytics import YOLO
        # 加载轻量级模型（首次运行自动下载，约6MB）
        model = YOLO("yolov8n.pt")
        print("YOLO模型加载          : ✅ 成功")
        
        # 虚拟图片测试预测
        import numpy as np
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        print("YOLO预测功能          : ✅ 正常")
    except Exception as e:
        print(f"YOLO功能验证失败      : {str(e)[:60]}")

def main():
    """主函数：执行全量环境检测"""
    print("🔍 YOLO 运行环境全量检测开始 🔍\n")
    
    # 分步执行检测（新增PyTorch专属模块）
    check_python_info()
    check_pytorch_version()  # 独立的PyTorch版本检测
    check_yolo_dependencies()
    check_cuda_info()
    check_yolo_basic_function()
    
    print("\n" + "="*60)
    print("        📋 检测完成        ")
    print("="*60)

if __name__ == "__main__":
    main()