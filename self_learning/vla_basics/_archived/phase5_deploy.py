"""阶段 5：用训练好的 VLA 模型控制模拟器（使用真实视频帧）

与之前不同：这次把训练数据中的真实视频帧喂给模型，
让模型看到连续的画面变化，产生连续的完整动作轨迹。

运行:
  终端1: cd d:/lerobot/playground && npm run dev
  终端2: conda activate python312 && python self_learning/vla_basics/phase5_deploy.py
"""

import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path("src")))

from websocket_robot import GenkiPiSimRobot
from lerobot.policies.smolvla import SmolVLAPolicy
from transformers import AutoTokenizer

# ━━━ 配置 ━━━
CHECKPOINT_PATH = "outputs/train/smolvla_sponge2box/final_model.pt"
NORM_STATS_PATH = "outputs/train/smolvla_sponge2box/norm_stats.pt"
VIDEO_DIR = Path("data/vla_from_v3/videos")
TASK_TEXT = "抓取黄色海绵块放到黑色盒子里"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODE = 0     # 使用第 1 个 episode 的视频
IMAGE_SIZE = 256
FPS = 30        # 播放速度
MAX_STEPS = 500  # 播放完整视频（约 15-20 秒）

print(f"设备: {DEVICE}")
print(f"模型: {CHECKPOINT_PATH}")
print(f"Episode: {EPISODE}")
print()

# ━━━ 1. 启动 WebSocket ━━━
print("启动 WebSocket 服务器...")
robot = GenkiPiSimRobot()
robot.start()

print("请在浏览器中打开 http://localhost:5173（确保 npm run dev 已启动）\n")
while not robot.is_connected:
    time.sleep(0.5)
print("浏览器已连接！\n")

# ━━━ 2. 加载归一化参数 ━━━
# 训练时对 state/action 做了 (x - mean) / std 归一化
# 推理时需要：输入 state 归一化，输出 action 反归一化
norm_stats = torch.load(NORM_STATS_PATH, map_location="cpu")
state_mean = norm_stats["state_mean"].to(DEVICE)   # (6,)
state_std = norm_stats["state_std"].to(DEVICE)
action_mean = norm_stats["action_mean"].to(DEVICE)  # (6,)
action_std = norm_stats["action_std"].to(DEVICE)
print(f"  action_mean: {action_mean.tolist()}")
print(f"  action_std:  {action_std.tolist()}")

# ━━━ 3. 加载模型 ━━━
print("加载 VLA 模型...")
model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", local_files_only=True)
model.to(DEVICE)
model.eval()

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"  ✅ 步骤 {checkpoint['step']}, loss={checkpoint.get('loss', 'N/A')}")

tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True
)
print("模型就绪！\n")

# ━━━ 4. 预编码语言 ━━━
encoded = tokenizer(
    TASK_TEXT + "\n",
    return_tensors="pt", padding="max_length", max_length=48, truncation=True,
)
lang_tokens = encoded["input_ids"].to(DEVICE)
lang_mask = encoded["attention_mask"].bool().to(DEVICE)

# ━━━ 5. 验证 action 格式 (绝对角 vs 增量) ━━━
print(f"🎬 加载 episode {EPISODE:06d} 视频...")
print(f"📐 验证 action 格式...")
import pyarrow as pa
import pyarrow.ipc as ipc

try:
    with pa.memory_map("data/vla_from_v3/data-00000-of-00001.arrow") as source:
        reader = ipc.open_file(source)
        table = reader.read_all()
        ep_col = table.column("episode_index").to_pylist()
        state_col = table.column("observation.state").to_pylist()
        action_col = table.column("action").to_pylist()

    # 取 episode 0 的头几帧
    ep0_indices = [i for i, ep in enumerate(ep_col) if ep == 0][:5]
    print(f"  Episode 0, 前 5 帧:")
    for idx in ep0_indices:
        s = state_col[idx]
        a = action_col[idx]
        delta = [a[i] - s[i] for i in range(6)]
        print(f"    Frame {idx}:")
        print(f"      state:  {[round(x,1) for x in s]}")
        print(f"      action: {[round(x,1) for x in a]}")
        print(f"      delta:  {[round(x,2) for x in delta]}")
    print()
except Exception as e:
    print(f"  ⚠ 无法读取 arrow: {e}")
    print()

caps = {}
for cam_name in ["phone", "laptop"]:
    vid_path = VIDEO_DIR / f"observation.images.{cam_name}_episode_{EPISODE:06d}.mp4"
    cap = cv2.VideoCapture(str(vid_path))
    caps[cam_name] = cap

phone_fps = caps["phone"].get(cv2.CAP_PROP_FPS)
num_frames = int(caps["phone"].get(cv2.CAP_PROP_FRAME_COUNT))
print(f"  {num_frames} 视频帧, {phone_fps:.0f} FPS")
print()

# ━━━ 6. 主循环：标准 chunk 推理 ━━━
# SmolVLA 的核心设计：一次观测 → 生成 20 步动作 → 全部执行 → 再观测
# 不每帧重新推理（避免把模型自己的输出当 state 喂回去导致分布漂移）
CHUNK = 20  # 和训练时的 chunk_size 一致
print("=" * 60)
print(f"  任务: {TASK_TEXT}")
print(f"  chunk 推理: 每 {CHUNK} 帧观测一次，中间帧执行规划好的动作")
print("=" * 60)
print()

frame_interval = 1.0 / FPS
total_frames = min(num_frames, MAX_STEPS)

# action 队列: 缓存模型生成的 chunk，逐步执行
action_queue = []  # list of (6,) tensors

for step in range(total_frames):
    t0 = time.perf_counter()

    # 读取视频帧
    images = {}
    ret = True
    for cam_name in ["phone", "laptop"]:
        r, frame = caps[cam_name].read()
        if r:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            images[cam_name] = img_tensor.unsqueeze(0)  # (1, 3, H, W)
        else:
            ret = False
            break

    if not ret:
        break

    # 每 CHUNK 步重新观测 + 生成新 chunk
    if step % CHUNK == 0:
        # 用当前真实 state（从 robot 读取）— 只在这一刻需要
        cur_state_raw = torch.tensor(
            robot._current_angles if hasattr(robot, '_current_angles') else [0]*6,
            dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        cur_state = (cur_state_raw - state_mean) / state_std

        observation = {
            "observation.images.camera1": images["phone"].to(DEVICE),
            "observation.images.camera2": images["laptop"].to(DEVICE),
            "observation.images.camera3": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE),
            "observation.state": cur_state,
            "observation.language.tokens": lang_tokens,
            "observation.language.attention_mask": lang_mask,
        }

        with torch.no_grad():
            action_chunk = model.predict_action_chunk(observation)  # (1, C, 6)

        # 反归一化 + 放入队列
        actions_denorm = action_chunk[0] * action_std + action_mean  # (C, 6)
        action_queue = [actions_denorm[i] for i in range(CHUNK)]

        # 诊断：打印前 3 个 chunk 的完整 20 步
        if step < 60 and step % CHUNK == 0:
            print(f"\n  📊 Chunk @ step={step}:")
            print(f"     模型原始输出 (归一化空间):")
            for j in range(3):
                raw = action_chunk[0, j].cpu().tolist()
                print(f"       raw[{j:2d}]: {[round(x, 3) for x in raw]}")
            print(f"     反归一化后:")
            for j in range(min(5, CHUNK)):
                a = actions_denorm[j].cpu().tolist()
                print(f"       action[{j:2d}]: {[round(x, 1) for x in a]}")
            print(f"       ... (共 {CHUNK} 步)")
            # 检查 20 步是否各不相同
            diff = 0.0
            for j in range(1, CHUNK):
                diff += (actions_denorm[j] - actions_denorm[0]).abs().sum().item()
            print(f"     chunk 内变化总量: {diff:.1f}  (0=全部相同)")
            if diff < 1.0:
                print(f"     ⚠ chunk 内 20 步几乎完全相同！")
            print()

    # 从队列取当前步的动作
    if action_queue:
        angles = action_queue.pop(0).cpu().tolist()
    else:
        angles = [0.0] * 6

    robot.send_angles(angles)

    if step % 10 == 0:
        print(f"  步 {step:3d}: {[round(a, 1) for a in angles]}")

    # 控制播放速度
    elapsed = time.perf_counter() - t0
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)

# ━━━ 6. 收尾 ━━━
for cap in caps.values():
    cap.release()

time.sleep(1)
robot.send_angles([0, 0, 0, 0, 0, 0])
time.sleep(0.5)
robot.stop()

print(f"\n=== 阶段 5 完成 ===")
print(f"模型: {CHECKPOINT_PATH}  |  任务: {TASK_TEXT}")
print(f"播放了 {total_frames} 帧视频 → 模型生成了对应动作")
print()
print("你应该在浏览器里看到了完整的连续运动轨迹！")
print("对比阶段 2 的随机游走 → 这次是有意义的抓取动作")
