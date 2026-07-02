import onnxruntime as ort
import numpy as np

def check_model_output(model_path):
    """检查模型输出维度"""
    try:
        # 尝试用GPU，如果失败则用CPU
        try:
            session = ort.InferenceSession(model_path, 
                                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            print("使用GPU执行提供者")
        except:
            session = ort.InferenceSession(model_path)
            print("使用CPU执行提供者")
        
        # 获取输入输出信息
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print("\n=== 模型输入信息 ===")
        for inp in inputs:
            print(f"  名称: {inp.name}")
            print(f"  形状: {inp.shape}")
            print(f"  类型: {inp.type}")
            
        print("\n=== 模型输出信息 ===")
        for out in outputs:
            print(f"  名称: {out.name}")
            print(f"  形状: {out.shape}")
            print(f"  类型: {out.type}")
            
            # 推断类别数量
            # YOLOv5输出形状通常为: [1, 25200, 85]
            # 其中85 = 4(坐标) + 1(置信度) + 80(类别数)
            if len(out.shape) == 3:
                total_features = out.shape[2]
                num_classes = total_features - 5  # 减去4个坐标+1个置信度
                print(f"  推断类别数: {num_classes}")
                
        return num_classes
        
    except Exception as e:
        print(f"错误: {e}")
        return None 

# 使用你的模型路径  
model_path = "C:/Users/23615/Desktop/.hyper/Gsing/CV/YOLO/weights/task1.onnx"
num_classes = check_model_output(model_path)

if num_classes:
    print(f"\n模型有 {num_classes} 个输出类别")
    print(f"你的YAML配置需要定义 {num_classes} 个类别名称")