"""Teacher Forcing 回放验证

== 做什么 ==
  用训练数据里的 (视频帧, 真实state) → 模型 → 预测action
  和数据集里的真实 action 对比，算 MAE

== 如果 MAE 小 ==
  模型学会了，问题在部署（state分布不匹配等）

== 如果 MAE 大 ==
  模型没学会，需要更多训练步数

== 运行 ==
  python self_learning/vla_basics/phase6_teacher_forcing.py
"""

import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

sys.path.insert(0, str(Path("src")))
from lerobot.policies.smolvla import SmolVLAPolicy
# ═══ 配置 ═══
CHECKPOINT_PATH = "outputs/train/smolvla_sponge2box/final_model.pt"
NORM_STATS_PATH = "outputs/train/smolvla_sponge2box/norm_stats.pt"
DATA_DIR = "data/vla_from_v3"
VIDEO_DIR = Path("data/vla_from_v3/videos")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK_TEXT = "抓取黄色海绵块放到黑色盒子里"
EPISODE = 0
IMAGE_SIZE = 256
EVAL_EVERY = 50  # 每隔多少帧打印一次

print("=" * 60)
print("  Teacher Forcing 回放验证")
print("=" * 60)
print(f"  模型: {CHECKPOINT_PATH}")
print(f"  Episode: {EPISODE}")
print()

# ━━━ 1. 加载模型 + 归一化 ━━━
print("🤖 加载模型 ...")
model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", local_files_only=True)
model.to(DEVICE)
model.eval()
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.config.chunk_size = 20       # 和训练时一致
model.config.n_action_steps = 20
print(f"  ✅ step={ckpt['step']}, chunk_size={model.config.chunk_size}")

norm_stats = torch.load(NORM_STATS_PATH, map_location="cpu")
state_mean = norm_stats["state_mean"]
state_std = norm_stats["state_std"]
action_mean = norm_stats["action_mean"].to(DEVICE)
action_std = norm_stats["action_std"].to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True
)
encoded = tokenizer(TASK_TEXT + "\n", return_tensors="pt",
                    padding="max_length", max_length=48, truncation=True)
lang_tokens = encoded["input_ids"].to(DEVICE)
lang_mask = encoded["attention_mask"].bool().to(DEVICE)

# ━━━ 2. 加载 episode 数据 ━━━
print(f"📂 加载 episode {EPISODE} 数据 ...")
ds = load_from_disk(DATA_DIR)
ep_data = [row for row in ds if row["episode_index"] == EPISODE]
ep_data.sort(key=lambda r: r["timestamp"])

ep_states = torch.tensor(
    [row["observation.state"] for row in ep_data], dtype=torch.float32
)
ep_actions_gt = torch.tensor(
    [row["action"] for row in ep_data], dtype=torch.float32
)
total_frames = len(ep_states)
print(f"  {total_frames} 帧")

# ━━━ 3. 预加载视频帧（回放时顺序读，直接全解码） ━━━
print("🎬 解码视频 ...")
ep_frames = {"phone": [], "laptop": []}
for cam_name in ["phone", "laptop"]:
    vid_path = VIDEO_DIR / f"observation.images.{cam_name}_episode_{EPISODE:06d}.mp4"
    cap = cv2.VideoCapture(str(vid_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        ep_frames[cam_name].append(frame)
    cap.release()
    print(f"  {cam_name}: {len(ep_frames[cam_name])} 帧")
frames_available = min(len(ep_frames["phone"]), len(ep_frames["laptop"]), total_frames)
print(f"  ✅ 可用帧: {frames_available}")

# ━━━ 4. Teacher Forcing 回放 ━━━
print(f"\n{'='*60}")
print(f"  Teacher Forcing: Frame_i + State_i → predict → compare with Action_i")
print(f"{'='*60}\n")

all_mae = []  # 每帧的 MAE
t0 = time.perf_counter()

for i in range(frames_available):
    # 取当前帧画面
    phone_frame = torch.from_numpy(ep_frames["phone"][i]).permute(2, 0, 1).float() / 255.0
    laptop_frame = torch.from_numpy(ep_frames["laptop"][i]).permute(2, 0, 1).float() / 255.0

    # 取当前帧 real state（先在 cpu 归一化，再搬到 GPU）
    cur_state = ((ep_states[i] - state_mean) / state_std).unsqueeze(0).to(DEVICE)

    observation = {
        "observation.images.camera1": phone_frame.unsqueeze(0).to(DEVICE),
        "observation.images.camera2": laptop_frame.unsqueeze(0).to(DEVICE),
        "observation.state": cur_state,
        "observation.language.tokens": lang_tokens,
        "observation.language.attention_mask": lang_mask,
    }

    with torch.no_grad():
        action_chunk = model.predict_action_chunk(observation)
    action_pred = action_chunk[0, 0]  # 只取第 1 步
    action_pred_denorm = action_pred * action_std + action_mean

    # ground truth
    action_gt = ep_actions_gt[i].to(DEVICE)

    # MAE (在原始空间)
    mae = (action_pred_denorm - action_gt).abs().mean().item()
    all_mae.append(mae)

    if i % EVAL_EVERY == 0:
        print(f"  帧 {i:4d}: MAE={mae:6.2f}°"
              f"  pred=[{', '.join(f'{x:6.1f}' for x in action_pred_denorm.cpu().tolist())}]"
              f"  gt=[{', '.join(f'{x:6.1f}' for x in action_gt.cpu().tolist())}]")

elapsed = time.perf_counter() - t0

# ━━━ 5. 统计 ━━━
all_mae_t = torch.tensor(all_mae)
print(f"\n{'='*60}")
print(f"  结果")
print(f"{'='*60}")
print(f"  总帧数:   {len(all_mae)}")
print(f"  平均 MAE: {all_mae_t.mean().item():.2f}°")
print(f"  最小 MAE: {all_mae_t.min().item():.2f}°")
print(f"  最大 MAE: {all_mae_t.max().item():.2f}°")
print(f"  中位 MAE: {all_mae_t.median().item():.2f}°")
print(f"  耗时:     {elapsed:.1f}s ({len(all_mae)/elapsed:.1f} fps)")

# 判断
avg_mae = all_mae_t.mean().item()
if avg_mae < 5:
    print(f"\n  ✅ 模型学会了！ MAE < 5°")
    print(f"     问题在部署代码（state分布不匹配 / camera3噪声等）")
elif avg_mae < 15:
    print(f"\n  ⚠ 模型学到了一些，但不够精确 (MAE={avg_mae:.1f}°)")
    print(f"     建议增加训练步数到 10000+")
else:
    print(f"\n  ❌ 模型没学会 (MAE={avg_mae:.1f}°)")
    print(f"     需要更多训练步数，建议 10000+")

# 检查 predict_action_chunk 输出格式
print(f"\n  predict_action_chunk 输出 shape: {action_chunk.shape}")
print(f"  反归一化后范围: [{action_pred_denorm.min().item():.1f}, {action_pred_denorm.max().item():.1f}]")
print(f"  GT 范围: [{ep_actions_gt.min().item():.1f}, {ep_actions_gt.max().item():.1f}]")
