# 训练
python tools/train.py yml/math.yml

# 调用摄像头实时检测
python demo/demo.py webcam --config yml/math.yml --model runs_math\model_best\model_best.ckpt --camid 1    

# 检测单张图片
python demo/demo.py image --config yml/math.yml --model weights\MATH.ckpt --path "C:\Users\23615\Desktop\.hyper\PC\CV\data\generated_math_nanodet\images\train\train_000000.png"

# 检测视频
python demo/demo.py video --config 配置文件.yml --model 模型路径.ckpt --path 视频路径.mp4

# 导出 ONNX 模型

python tools/export_onnx.py --cfg_path yml/math.yml --model_path runs_math\model_best\model_best.ckpt

# 1. 创建并激活虚拟环境
conda create -n nanodet python=3.8 -y
conda activate nanodet

# 2. 安装PyTorch（适配CUDA11.1）
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge

# 3. 克隆项目
git clone https://github.com/RangiLyu/nanodet.git
cd nanodet

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装项目（开发模式）
python setup.py develop




