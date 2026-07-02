
## yolo export model=weights\math11.pt format=onnx imgsz=1080


## yolo export model=weights\math12.pt format=onnx imgsz=1080 opset=21 half=False

(E:\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate yolo) 



conda activate Vision
cd C:/Users/23615/Desktop/.hyper/Gsing/CV/YOLO
yolo predict model=weights/yolov8s.pt source="D:\.Captures\games\valorant\集锦\1V5.mp4" device=0
