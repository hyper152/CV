import cv2
import os
import glob

# ====================== 【所有配置项都在这里，按需修改，其他不用动】 ======================
TARGET_FOLDER = r"C:\Users\23615\Desktop\.hyper\PC\CV\data\img\steelball\mp4"  # 视频文件夹根目录（放 mp4）
SAVE_FOLDER = os.path.join(TARGET_FOLDER, "..", "images")  # 图片保存目录 = 视频文件夹的上一级 images 文件夹
MERGE_SAVE = True                         # ✅核心开关：True=所有图片放同一个文件夹 | False=分视频创建子文件夹
EXTRACT_MODE = 2                          # 提取模式【必选】
                                          # 1=按帧率提取(推荐)  2=提取视频的每一帧  3=按固定秒数提取
FPS = 2                                   # 模式1用：每秒提取1张图 (比如FPS=2=每秒2张)
INTERVAL_SEC = 0.2                          # 模式3用：每N秒提取1张图
# ==========================================================================================

# 支持的视频格式，主流格式全覆盖，按需增减
SUPPORT_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".flv", ".mkv", ".wmv", ".mpeg", ".mpg"]

def video2images(video_path, save_root):
    """单视频提取图片的核心函数，兼容合并/分文件夹两种保存方式"""
    video_full_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_full_name)[0]
    video_suffix = os.path.splitext(video_full_name)[1]
    
    # 根据开关判断保存路径：合并保存/分文件夹保存
    if MERGE_SAVE:
        save_dir = save_root  # 所有图片都放同一个总文件夹
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.join(save_root, video_name)  # 每个视频单独创建子文件夹
        os.makedirs(save_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 打开失败：{video_full_name}，跳过该视频")
        return 0

    # 获取视频基础信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)  # 视频原帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    print(f"\n📹 正在处理：{video_full_name}")
    print(f"📊 视频信息：帧率={video_fps:.1f} | 总帧数={total_frames} | 时长≈{total_frames/video_fps:.1f}秒")

    count = 0  # 帧计数
    save_count = 0  # 保存的图片计数
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break
        
        # ========== 三种提取模式 ==========
        save_flag = False
        if EXTRACT_MODE == 1:
            # 模式1：按指定帧率提取，最常用，不会产生海量图片，推荐！
            if count % int(video_fps / FPS) == 0:
                save_flag = True
        elif EXTRACT_MODE == 2:
            # 模式2：提取每一帧，适合逐帧分析，图片数量较多
            save_flag = True
        elif EXTRACT_MODE == 3:
            # 模式3：按固定秒数提取，精准控制间隔，适合长视频抽样
            if count % int(video_fps * INTERVAL_SEC) == 0:
                save_flag = True

        # 保存图片 - 合并模式下命名带视频名，防止重名；分文件夹模式正常命名
        if save_flag:
            if MERGE_SAVE:
                # 合并保存：图片名格式【视频名_帧序号.jpg】，区分不同视频的图片
                img_name = f"{video_name}_frame_{save_count:06d}.jpg"
            else:
                # 分文件夹保存：图片名格式【帧序号.jpg】，简洁有序
                img_name = f"frame_{save_count:06d}.jpg"
            
            img_path = os.path.join(save_dir, img_name)
            # 高清无损保存图片，100%画质，不压缩
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            save_count += 1

        count += 1

    # 释放视频资源
    cap.release()
    print(f"✅ 处理完成：提取到 {save_count} 张图片 → 保存至：{save_dir}")
    return save_count

def batch_extract():
    """批量遍历文件夹，提取所有视频为图片"""
    # 创建总保存目录
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    # 遍历目标文件夹内所有文件，筛选视频
    all_video_paths = []
    for file_path in glob.glob(os.path.join(TARGET_FOLDER, "*")):
        file_suffix = os.path.splitext(file_path)[1].lower()
        if file_suffix in SUPPORT_VIDEO_FORMATS:
            all_video_paths.append(file_path)

    # 判断是否找到视频
    if len(all_video_paths) == 0:
        print("❌ 未在目标文件夹中找到任何视频文件！")
        return

    # 打印提取配置信息，一目了然
    print("="*70)
    print(f"✅ 扫描完成！共找到 {len(all_video_paths)} 个视频文件")
    print(f"✅ 保存模式：{'【合并保存】所有图片放入同一个文件夹' if MERGE_SAVE else '【分文件夹保存】每个视频单独存放'}")
    print(f"✅ 提取模式：{['按帧率提取', '提取全部帧', '按秒间隔提取'][EXTRACT_MODE-1]}")
    print(f"✅ 保存根目录：{os.path.abspath(SAVE_FOLDER)}")
    print("="*70)

    # 逐个处理所有视频
    total_save = 0
    for idx, video_path in enumerate(all_video_paths, 1):
        save_num = video2images(video_path, SAVE_FOLDER)
        total_save += save_num

    # 提取完成汇总信息
    print("\n" + "="*70)
    print(f"🎉 全部提取完成！✅ 结果汇总")
    print(f"📌 共处理视频数：{len(all_video_paths)} 个")
    print(f"📌 共提取图片数：{total_save} 张")
    print(f"📌 图片保存路径：{os.path.abspath(SAVE_FOLDER)}")
    print("="*70)

if __name__ == "__main__":
    batch_extract()