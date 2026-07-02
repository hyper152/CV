import cv2          # 导入核心库，仅保留必要的
import os           # 路径处理保留原逻辑
import time         # 基础时间处理
from datetime import datetime  # 仅用于保存时的时间戳（可选精简）

# ===================== 保留原存储路径配置 =====================
# 完全保留你最初的路径逻辑，不做任何修改
ROOT = os.path.dirname(os.path.dirname(__file__))
IM_DIR = os.path.join(ROOT,'task', 'images', 'train')
os.makedirs(IM_DIR, exist_ok=True)

# ===================== 流畅化配置 =====================
# 降低分辨率保证流畅（720P兼顾清晰度和性能）
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

def main():
    """
    精简版：保留原路径 + 核心帧率优化 + 最少计算，保证流畅
    """
    # 打开摄像头（仅保留核心后端参数）
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError('无法打开摄像头（索引 1）。请检查权限或更换索引。')

    # 仅保留最关键的帧率优化（去掉所有冗余配置）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 核心：减少缓冲区，降低延迟
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 快速编码

    # 初始化极简变量
    mirror = True  # 保留镜像功能
    idx = int(time.time())  # 保留序号避免重名
    print('提示:')
    print('  空格: 保存当前帧（路径不变）')
    print('  v: 开/关镜像')
    print('  q: 退出')

    # ===================== 极简主循环（无冗余计算） =====================
    while True:
        # 快速读取帧（失败则短间隔重试）
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue

        # 仅保留必要的镜像翻转（无多余操作）
        frame_show = cv2.flip(frame, 1) if mirror else frame

        # 只绘制一行极简提示（减少绘图开销）
        cv2.putText(frame_show, 'Space: save | v: mirror | q: quit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        # 显示窗口（不调整大小，减少系统负担）
        cv2.imshow('Capture', frame_show)

        # 按键处理（极简逻辑，无多余判断）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            mirror = not mirror
        elif key == 32:  # 空格键保存（保留原路径+精简文件名）
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            fname = f'cap_{ts}_{idx:06d}.jpg'
            fpath = os.path.join(IM_DIR, fname)  # 完全保留原路径拼接逻辑

            # 精简保存逻辑（去掉冗余校验，保证速度）
            try:
                ok, buf = cv2.imencode('.jpg', frame_show)
                if ok:
                    buf.tofile(fpath)
                    print('已保存:', os.path.normpath(fpath))
                    idx += 1
            except Exception as e:
                print('保存失败:', e)

    # 释放资源（极简收尾）
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()