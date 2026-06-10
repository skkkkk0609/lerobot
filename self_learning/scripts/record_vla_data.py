"""
VLA 数据录制脚本
在 d:/lerobot 下运行，使用 .venv 环境。
直接用 scservo_sdk 实现主从跟随 + 摄像头录制 + LeRobot 格式保存。

用法:
    cd d:\lerobot
    D:\lerobot\.venv\Scripts\python.exe record_vla_data.py
"""

import os
import platform

if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime

# 添加 scservo_sdk 路径
sdk_path = r"d:\arm_robot_begin\机械臂资料00\FTServo_Python-main\FTServo_Python-main"
if sdk_path not in sys.path:
    sys.path.append(sdk_path)

try:
    from scservo_sdk import *
except ImportError:
    print(f"错误：无法找到 scservo_sdk，路径: {sdk_path}")
    sys.exit(1)

import cv2
import numpy as np
import pandas as pd
import av as av

# ============ 配置 ============
MASTER_PORT = "COM7"   # 主臂（leader）
SLAVE_PORT = "COM6"    # 从臂（follower）
BAUDRATE = 1000000
SERVO_IDS = [1, 2, 3, 4, 5, 6]
SERVO_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
MOVING_SPEED = 2400
MOVING_ACC = 50
FPS = 30

# 摄像头
CAMERA_INDEX = 1  # 顶部摄像头
CAMERA_W = 640
CAMERA_H = 480

# 数据集
DATA_ROOT = r"d:\lerobot\data\pick_smola_vla_test"
REPO_ID = "pick/smola_vla_test"
TASK_DESCRIPTION = "抓住红色方块放到黑色盒子里"
EPISODE_TIME_S = 30
RESET_TIME_S = 10
TARGET_EPISODES = 50


def init_motors():
    master_port = PortHandler(MASTER_PORT)
    slave_port = PortHandler(SLAVE_PORT)
    master_handler = sms_sts(master_port)
    slave_handler = sms_sts(slave_port)

    if not master_port.openPort():
        print(f"无法打开主臂端口 {MASTER_PORT}")
        return None, None, None, None
    master_port.setBaudRate(BAUDRATE)

    if not slave_port.openPort():
        print(f"无法打开从臂端口 {SLAVE_PORT}")
        master_port.closePort()
        return None, None, None, None
    slave_port.setBaudRate(BAUDRATE)

    print(f"主臂 {MASTER_PORT} 已连接")
    print(f"从臂 {SLAVE_PORT} 已连接")
    return master_port, master_handler, slave_port, slave_handler


def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if cap.isOpened():
        print(f"摄像头 {CAMERA_INDEX} 已打开")
        return cap
    print(f"警告：摄像头 {CAMERA_INDEX} 无法打开，将不录制视频")
    return None


def record_one_episode(master_handler, slave_handler, cap, episode_idx, data_dir, video_dir):
    """录制一个 episode，返回帧数"""
    timestamps = []
    actions = []       # 主臂位置
    observations = []  # 从臂位置
    tasks = []
    frames = []

    start_time = time.time()
    loop_count = 0

    # Enter 键监听
    stop_event = threading.Event()

    def wait_for_enter():
        try:
            input()
        except EOFError:
            pass
        stop_event.set()

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    print(f"  录制中... (按 Enter 停止，最长 {EPISODE_TIME_S} 秒)")

    while not stop_event.is_set() and (time.time() - start_time) < EPISODE_TIME_S:
        loop_start = time.time()

        # 读主臂 → 写从臂
        master_pos = []
        slave_pos = []
        all_ok = True

        for sid in SERVO_IDS:
            pos, speed, res, err = master_handler.ReadPosSpeed(sid)
            if res == COMM_SUCCESS:
                master_pos.append(float(pos))
                write_res, _ = slave_handler.WritePosEx(sid, int(pos), MOVING_SPEED, MOVING_ACC)
                slave_pos.append(float(pos))
            else:
                master_pos.append(0.0)
                slave_pos.append(0.0)
                all_ok = False

        if not all_ok:
            time.sleep(0.01)
            continue

        # 读摄像头
        if cap:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        # 记录
        ts = time.time() - start_time
        timestamps.append(ts)
        actions.append(master_pos.copy())
        observations.append(slave_pos.copy())
        tasks.append(TASK_DESCRIPTION)

        loop_count += 1

        # 显示状态
        elapsed = time.time() - start_time
        angles = [(p * 360.0 / 4096.0) for p in master_pos]
        status = " | ".join([f"{SERVO_NAMES[i]}:{a:5.1f}" for i, a in enumerate(angles)])
        print(f"\r  [{elapsed:5.1f}s] 帧#{loop_count:4d} {status}", end="", flush=True)

        # 保持 FPS
        dt = time.time() - loop_start
        sleep_time = (1.0 / FPS) - dt
        if sleep_time > 0:
            time.sleep(sleep_time)

    stop_event.set()
    t.join(timeout=0.5)

    if loop_count == 0:
        return 0

    # 保存数据
    data = {
        "timestamp": timestamps,
        "task": tasks,
    }
    for i, name in enumerate(SERVO_NAMES):
        data[f"observation.state.{name}"] = [p[i] for p in observations]
        data[f"action.{name}"] = [p[i] for p in actions]

    df = pd.DataFrame(data)
    parquet_path = data_dir / f"episode_{episode_idx:04d}.parquet"
    df.to_parquet(parquet_path, index=False)

    # 保存视频
    if frames:
        video_path = video_dir / f"episode_{episode_idx:04d}_phone.mp4"
        with av.open(str(video_path), mode='w') as container:
            stream = container.add_stream('libx264', rate=FPS)
            stream.width = CAMERA_W
            stream.height = CAMERA_H
            stream.pix_fmt = 'yuv420p'
            stream.options = {'preset': 'medium', 'crf': '30'}

            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
                av_frame = av_frame.reformat(format='yuv420p')
                for packet in stream.encode(av_frame):
                    container.mux(packet)
            # Flush
            for packet in stream.encode():
                container.mux(packet)

    print(f"\n  Episode {episode_idx} 完成: {loop_count} 帧")
    return loop_count


def save_metadata(num_episodes, total_frames):
    meta_dir = Path(DATA_ROOT) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "repo_id": REPO_ID,
        "robot_type": "so100_follower",
        "fps": FPS,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "single_task": TASK_DESCRIPTION,
        "cameras": ["phone"],
        "servo_ids": SERVO_IDS,
        "servo_names": SERVO_NAMES,
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    tasks_df = pd.DataFrame({
        "episode_index": list(range(num_episodes)),
        "task": [TASK_DESCRIPTION] * num_episodes,
    })
    tasks_df.to_parquet(meta_dir / "tasks.parquet", index=False)


def main():
    print("=" * 60)
    print("VLA 数据录制 (跟随模式 + 语言标注)")
    print("=" * 60)

    # 初始化
    master_port, master_handler, slave_port, slave_handler = init_motors()
    if not master_handler:
        return
    cap = init_camera()

    data_dir = Path(DATA_ROOT) / "data"
    video_dir = Path(DATA_ROOT) / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = 0
    total_frames = 0

    print(f"\n目标: {TARGET_EPISODES} 个 episodes")
    print("任务: {TASK_DESCRIPTION}")
    print("按 Enter 开始录制 episode，再按 Enter 停止当前 episode")
    print("按 Ctrl+C 结束录制\n")

    try:
        while num_episodes < TARGET_EPISODES:
            input(f"--- 按 Enter 开始录制 episode {num_episodes} ---")

            frames = record_one_episode(
                master_handler, slave_handler, cap,
                num_episodes, data_dir, video_dir
            )

            if frames > 0:
                total_frames += frames
                num_episodes += 1

                if num_episodes < TARGET_EPISODES:
                    print(f"  重置中... ({RESET_TIME_S} 秒)")
                    time.sleep(RESET_TIME_S)

    except KeyboardInterrupt:
        print("\n\n用户停止录制。")
    finally:
        save_metadata(num_episodes, total_frames)

        if cap:
            cap.release()
        if master_port:
            master_port.closePort()
        if slave_port:
            slave_port.closePort()

        print(f"\n{'='*60}")
        print(f"录制完成!")
        print(f"  Episodes: {num_episodes}")
        print(f"  总帧数: {total_frames}")
        print(f"  数据目录: {DATA_ROOT}")
        print(f"  任务标注: {TASK_DESCRIPTION}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
