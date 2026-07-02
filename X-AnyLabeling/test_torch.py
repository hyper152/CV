import torch
# 1. 检查torch是否能正常导入（解决之前的ModuleNotFoundError）
print("✅ PyTorch导入成功，版本：", torch.__version__)
# 2. 检查是否能调用GPU（核心！必须返回True）
print("✅ GPU是否可用：", torch.cuda.is_available())
# 3. 验证CUDA版本匹配性+查看RTX5060显卡信息
if torch.cuda.is_available():
    print("✅ PyTorch实际调用的CUDA版本：", torch.version.cuda)
    print("✅ 你的显卡：", torch.cuda.get_device_name(0))
    print("✅ GPU显存：", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), "GB")