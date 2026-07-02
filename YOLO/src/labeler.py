import os           # 导入os库，用于路径处理、文件夹创建
import cv2          # 导入OpenCV库，用于图像读取、绘制、窗口交互
import glob         # 导入glob库，用于批量匹配图片文件
import numpy as np  # 用于图像缩放适配1080P显示

# ===================== 全局配置：适配4类别YOLO配置 =====================
# 多类标注器（对应你的yaml配置）
# 类别映射：class_id -> 类名
CLASS_MAP = {
    0: 'tool',
    1: 'device',
    2: 'food',
    3: 'remedy'
}
# 核心修改：按你的要求调整类别颜色
# OpenCV颜色格式为 (B, G, R)，灰色用(128,128,128)，红色(0,0,255)，蓝色(255,0,0)，绿色(0,255,0)
CLASS_COLORS = {
    0: (128, 128, 128),# tool - 灰色
    1: (255, 0, 0),    # device - 蓝色
    2: (0, 255, 0),     # food - 绿色
    3: (0, 0, 255)    # remedy - 红色
}
# 操作说明新增：
#  - 0/1/2/3 切换当前标注类别（对应device/tool/remedy/food）
#  - 其他操作不变：鼠标画框、d撤销、c清空、n/p切换、s保存、q退出

# 你的项目路径（替换为yaml里的路径，也可保留原逻辑）
TRAIN_IMG_DIR = r'C:\Users\hyper\Desktop\program\.Gsing\YOLO/task/images/train'
TRAIN_LBL_DIR = r'C:\Users\hyper\Desktop\program\.Gsing\YOLO/task/labels/train'
VAL_IMG_DIR = r'C:\Users\hyper\Desktop\program\.Gsing\YOLO/task/images/val'
VAL_LBL_DIR = r'C:\Users\hyper\Desktop\program\.Gsing\YOLO/task/labels/val'

# 选择标注集：train（训练集）或 val（验证集），修改这里切换
CURRENT_SET = 'train'
IM_DIR = TRAIN_IMG_DIR if CURRENT_SET == 'train' else VAL_IMG_DIR
LBL_DIR = TRAIN_LBL_DIR if CURRENT_SET == 'train' else VAL_LBL_DIR

# 确保标签目录存在
os.makedirs(LBL_DIR, exist_ok=True)

# 1080P适配配置
TARGET_DISPLAY_WIDTH = 1920
TARGET_DISPLAY_HEIGHT = 1080
SCALE_FACTOR = 1.0

# ===================== 核心修改：全局当前类别（跨图片保留） =====================
# 初始化全局当前类别ID（默认0=device），切换图片时不重置
global_current_class_id = 0

# ===================== 标注工具核心类：支持4类别标注 =====================
class BoxTool:
    def __init__(self, img_path):
        self.img_path = img_path          
        self.img = self._imread_chinese(img_path)   
        if self.img is None:
            raise RuntimeError(f'无法读取图片: {img_path}')
        
        self.h, self.w = self.img.shape[:2]
        self.clone = self.img.copy()
        # 存储框+类别 [(x1,y1,x2,y2, class_id), ...]
        self.boxes = []                      
        self.drawing = False                 
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        self.dirty = False

        # 1080P缩放适配
        global SCALE_FACTOR
        if self.w > TARGET_DISPLAY_WIDTH or self.h > TARGET_DISPLAY_HEIGHT:
            scale_w = TARGET_DISPLAY_WIDTH / self.w
            scale_h = TARGET_DISPLAY_HEIGHT / self.h
            SCALE_FACTOR = min(scale_w, scale_h)
        else:
            SCALE_FACTOR = 1.0
        self.display_w = int(self.w * SCALE_FACTOR)
        self.display_h = int(self.h * SCALE_FACTOR)

    def _imread_chinese(self, path):
        """读取中文路径图片"""
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img

    def _scale_coords(self, x, y, inverse=False):
        """坐标缩放转换"""
        if inverse:
            return int(x / SCALE_FACTOR), int(y / SCALE_FACTOR)
        else:
            return int(x * SCALE_FACTOR), int(y * SCALE_FACTOR)

    def mouse(self, event, x, y, flags, param):
        """鼠标画框（绑定全局类别）"""
        global global_current_class_id  # 引用全局类别变量
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x1, self.y1 = self._scale_coords(x, y, inverse=True)
            self.x2, self.y2 = self.x1, self.y1
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x2, self.y2 = self._scale_coords(x, y, inverse=True)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.x1, self.x2), min(self.y1, self.y2)
            x2, y2 = max(self.x1, self.x2), max(self.y1, self.y2)
            if x2 - x1 > 3 and y2 - y1 > 3:
                # 保存框+全局类别ID
                self.boxes.append((x1, y1, x2, y2, global_current_class_id))
                self.dirty = True

    def draw(self):
        """绘制框（按新配色显示不同颜色）"""
        global global_current_class_id  # 引用全局类别变量
        canvas = self.img.copy()
        # 绘制正在拖拽的实时框（全局类别颜色）
        if self.drawing:
            dx1, dy1 = self._scale_coords(self.x1, self.y1)
            dx2, dy2 = self._scale_coords(self.x2, self.y2)
            current_color = CLASS_COLORS[global_current_class_id]
            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), current_color, 2)
        
        # 绘制已保存的框（按新配色显示颜色+类名）
        for (x1, y1, x2, y2, cid) in self.boxes:
            dx1, dy1 = self._scale_coords(x1, y1)
            dx2, dy2 = self._scale_coords(x2, y2)
            color = CLASS_COLORS[cid]
            class_name = CLASS_MAP[cid]
            # 画框
            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), color, 2)
            # 左上角绘制类名（对应颜色底+白字，更清晰）
            cv2.rectangle(canvas, (dx1, dy1-20), (dx1+80, dy1), color, -1)
            cv2.putText(canvas, class_name, (dx1+5, dy1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # 缩放画布
        canvas = cv2.resize(canvas, (self.display_w, self.display_h))
        
        # 绘制操作提示（显示全局类别）
        current_class_name = CLASS_MAP[global_current_class_id]
        tips = [
            f'Current Class: {current_class_name} (ID:{global_current_class_id})',
            '0:tool(gray) 1:device(blue) 2:food(green) 3:remedy(red) | S:Save D:Undo C:Clear',
            'N/P:Next/Prev | Q:Quit | Boxes: {}'.format(len(self.boxes))
        ]
        # 第1行提示（当前类别，红色）
        cv2.putText(canvas, tips[0], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # 第2行提示（操作+颜色说明，绿色）
        cv2.putText(canvas, tips[1], (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # 第3行提示（数量，黄色）
        cv2.putText(canvas, tips[2], (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return canvas

    def save_label(self):
        """保存YOLO格式标签（含类别ID）"""
        img_name = os.path.splitext(os.path.basename(self.img_path))[0]
        label_path = os.path.join(LBL_DIR, img_name + '.txt')
        
        if not self.boxes:
            print('无框，未保存:', label_path)
            return
        
        # 转换为YOLO格式：class_id cx cy w h
        lines = []
        for (x1, y1, x2, y2, cid) in self.boxes:
            cx = ((x1 + x2) / 2) / self.w
            cy = ((y1 + y2) / 2) / self.h
            w = (x2 - x1) / self.w
            h = (y2 - y1) / self.h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        # 保存（支持中文路径）
        try:
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print('已保存标注:', label_path)
            self.dirty = False
        except Exception as e:
            print(f'保存失败: {e}')
            label_path_norm = os.path.normpath(label_path)
            with open(label_path_norm, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f'备选方案保存成功: {label_path_norm}')
            self.dirty = False

# ===================== 主函数：适配多类别 =====================
def main():
    global global_current_class_id  # 引用全局类别变量
    # 加载图片列表
    images = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        abs_pattern = os.path.abspath(os.path.join(IM_DIR, ext))
        images.extend(glob.glob(abs_pattern))
    images.sort()
    
    if not images:
        print(f'未在 {IM_DIR} 下找到图片！')
        return

    # 初始化窗口
    win_name = f'YOLO Labeler (4类) - {CURRENT_SET}'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT)

    idx = 0
    # 循环标注
    while True:
        if idx >= len(images) or idx < 0:
            print('已到列表边界！')
            idx = max(0, min(idx, len(images)-1))
            continue
        
        # 加载当前图片
        try:
            tool = BoxTool(images[idx])
        except RuntimeError as e:
            print(f'跳过图片: {e}')
            idx += 1
            continue
        
        cv2.setMouseCallback(win_name, tool.mouse)
        print(f'[{idx+1}/{len(images)}] -> {os.path.normpath(images[idx])}')
        # 打印全局类别+颜色说明
        color_note = {0:'灰色', 1:'蓝色', 2:'绿色', 3:'红色'}
        print(f'当前标注类别：{CLASS_MAP[global_current_class_id]} (ID:{global_current_class_id}，{color_note[global_current_class_id]})')

        # 单张图片交互
        while True:
            canvas = tool.draw()
            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(10) & 0xFF

            # 按键逻辑（新增0-3切换类别）
            if key == ord('q'):
                if tool.boxes and tool.dirty:
                    tool.save_label()
                cv2.destroyAllWindows()
                return
            # 切换类别：0/1/2/3
            elif key == ord('0'):
                global_current_class_id = 0
                print(f'切换到类别：{CLASS_MAP[0]} (ID:0，灰色)')
            elif key == ord('1'):
                global_current_class_id = 1
                print(f'切换到类别：{CLASS_MAP[1]} (ID:1，蓝色)')
            elif key == ord('2'):
                global_current_class_id = 2
                print(f'切换到类别：{CLASS_MAP[2]} (ID:2，绿色)')
            elif key == ord('3'):
                global_current_class_id = 3
                print(f'切换到类别：{CLASS_MAP[3]} (ID:3，红色)')
            # 原有操作
            elif key == ord('d'):
                if tool.boxes:
                    tool.boxes.pop()
                    tool.dirty = True
            elif key == ord('c'):
                tool.boxes.clear()
                tool.dirty = True
            elif key == ord('s'):
                tool.save_label()
            elif key == ord('n'):
                if tool.boxes and tool.dirty:
                    tool.save_label()
                break
            elif key == ord('p'):
                if tool.boxes and tool.dirty:
                    tool.save_label()
                idx -= 2
                break

        idx += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()