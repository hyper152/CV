# -*- coding: utf-8 -*-
"""
将指定 mp4 视频按逆时针角度旋转（支持 90/180/270 度），输出 *_rotated.mp4。

使用方法：
    python rotate_mp4_90ccw.py                     # 使用下方默认参数（ROTATION=90）
    python rotate_mp4_90ccw.py 视频路径1 视频路径2  # 命令行指定视频
    python rotate_mp4_90ccw.py --dir 某文件夹      # 批量处理文件夹内所有 .mp4
    python rotate_mp4_90ccw.py --rotate 180        # 改为逆时针旋转 180 度
"""

import argparse
import os
import shutil
import subprocess
import sys

# ============================================================================
# 参数配置区（按需修改）
# ============================================================================

# ffmpeg 可执行文件路径（若已在系统 PATH 中则无需修改）
FFMPEG = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"

# 需要旋转的视频路径（留空则使用命令行参数或 --dir）
VIDEO_PATHS = [
    r"C:\Users\23615\Desktop\.hyper\PC\CV\data\img\steelball\mp4\7.mp4",
    r"C:\Users\23615\Desktop\.hyper\PC\CV\data\img\steelball\mp4\6.mp4",
]

# 旋转角度（逆时针，单位：度），仅支持 0 / 90 / 180 / 270
ROTATION = 90   # 逆时针旋转角度

# 输出文件名后缀
OUT_SUFFIX = "_rotated"

# 输出编码参数
VIDEO_CODEC = "libx264"   # 视频编码器
PRESET = "fast"           # x264 预设（ultrafast/superfast/veryfast/faster/fast/medium/slow/veryslow）
CRF = "23"                # 画质（0-51，越小越清晰，文件越大）
AUDIO_CODEC = "copy"      # 音频处理（copy=原样拷贝）

# 是否覆盖已存在的输出文件
OVERWRITE = True

# ============================================================================
# 以下为程序逻辑，一般无需修改
# ============================================================================


def build_ffmpeg_cmd(input_path: str, out_path: str) -> list:
    """构造 ffmpeg 命令行。"""
    # 逆时针角度 -> 纯旋转滤镜链（transpose 不加翻转）
    #   transpose=1 顺时针90°，transpose=2 逆时针90°
    #   180° = 两次90°（顺/逆皆可）；270° = 一次顺时针90°
    if ROTATION == 90:
        vf = "transpose=2"
    elif ROTATION == 180:
        vf = "transpose=1,transpose=1"
    elif ROTATION == 270:
        vf = "transpose=1"
    elif ROTATION == 0:
        vf = "null"  # 不旋转
    else:
        raise ValueError(f"不支持的旋转角度: {ROTATION}（仅支持 0/90/180/270）")

    cmd = [
        FFMPEG,
        "-y" if OVERWRITE else "-n",
        "-i", input_path,
        "-vf", vf,
        "-c:v", VIDEO_CODEC,
        "-preset", PRESET,
        "-crf", CRF,
        "-c:a", AUDIO_CODEC,
        out_path,
    ]
    return cmd


def rotate_ccw(input_path: str) -> str:
    """将单个视频旋转 90 度，返回输出文件路径。"""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"视频文件不存在: {input_path}")

    dir_name = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(dir_name, base_name + OUT_SUFFIX + ".mp4")

    cmd = build_ffmpeg_cmd(input_path, out_path)
    print("运行:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 执行失败:\n{proc.stderr}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="视频逆时针旋转 90 度工具")
    parser.add_argument("videos", nargs="*", help="视频文件路径（可多个）")
    parser.add_argument("--dir", help="批量处理该目录下所有 .mp4 文件")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270],
                        help="逆时针旋转角度（覆盖上方 ROTATION 参数）")
    args = parser.parse_args()

    global ROTATION
    if args.rotate is not None:
        ROTATION = args.rotate

    # 确定要处理的文件列表
    if args.videos:
        paths = args.videos
    elif args.dir:
        paths = [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if f.lower().endswith(".mp4")
        ]
    else:
        paths = VIDEO_PATHS

    for path in paths:
        try:
            out = rotate_ccw(path)
            print(f"成功: {path} -> {out}")
        except Exception as e:
            print(f"失败: {path} -> {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
