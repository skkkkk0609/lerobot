"""阶段 6：真机 VLA 闭环控制

=== 每帧独立推理 ===
  不再用 chunk 缓存 → 每一帧都基于最新摄像头画面实时推理
  真摄像头 → VLA → 机械臂，纯闭环

=== 安全 ===
  Ctrl+C 不停扭矩（避免夹爪松脱），只停止发送指令
"""

import os
import platform
import sys
import time
from pathlib import Path

# ━━ Windows MSMF 摄像头修复 ━━
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# ━━ Feetech 官方 SDK ━━
sdk_path = r"D:\arm_robot_begin\机械臂资料00\FTServo_Python-main\FTServo_Python-main"
if sdk_path not in sys.path:
    sys.path.append(sdk_path)

try:
    from scservo_sdk import *
except ImportError:
    print(f"❌ 无法找到 scservo_sdk: {sdk_path}")
    sys.exit(1)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import torch

sys.path.insert(0, str(Path("src")))

from lerobot.policies.smolvla import SmolVLAPolicy
from transformers import AutoTokenizer

# ═══════════════════════════════════════════════════════════════
CHECKPOINT_PATH = "outputs/train/smolvla_sponge2box/final_model.pt"
NORM_STATS_PATH = "outputs/train/smolvla_sponge2box/norm_stats.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK_TEXT = "抓取黄色海绵块放到黑色盒子里"

FOLLOWER_PORT = "COM6"
BAUDRATE = 1000000
SERVO_IDS = [1, 2, 3, 4, 5, 6]
MOVING_SPEED = 2400
MOVING_ACC = 50

CAMERA_CONFIGS = {
    "phone":  {"index": 1, "width": 640, "height": 480, "fps": 30},
    "laptop": {"index": 2, "width": 640, "height": 480, "fps": 30},
}

IMAGE_SIZE = 256
FPS = 15                # 推理速度（每帧都推理，15fps 足够）
MAX_RELATIVE_TARGET = 30

# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  阶段 6: 真机 VLA 闭环控制")
print("=" * 60)
print(f"  从臂: {FOLLOWER_PORT}")
print()

# ━━━ 1. 加载模型 ━━━
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
state_mean = norm_stats["state_mean"].to(DEVICE)
state_std = norm_stats["state_std"].to(DEVICE)
action_mean = norm_stats["action_mean"].to(DEVICE)
action_std = norm_stats["action_std"].to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True
)
encoded = tokenizer(TASK_TEXT + "\n", return_tensors="pt",
                    padding="max_length", max_length=48, truncation=True)
lang_tokens = encoded["input_ids"].to(DEVICE)
lang_mask = encoded["attention_mask"].bool().to(DEVICE)

# ━━━ 2. 连接从臂 ━━━
print(f"\n🔌 连接从臂 ({FOLLOWER_PORT}) ...")
follower_port = PortHandler(FOLLOWER_PORT)
follower_sc = sms_sts(follower_port)

if not follower_port.openPort():
    print(f"  ❌ 无法打开 {FOLLOWER_PORT}")
    sys.exit(1)
if not follower_port.setBaudRate(BAUDRATE):
    follower_port.closePort()
    sys.exit(1)

for sid in SERVO_IDS:
    follower_sc.WritePosEx(sid, 2048, MOVING_SPEED, MOVING_ACC)
    time.sleep(0.02)

print("  ✅ 从臂就绪")

# ━━━ 3. 连接摄像头 ━━━
print(f"\n📷 连接摄像头 (MSMF) ...")
cameras = {}
for cam_name, cfg in CAMERA_CONFIGS.items():
    cap = cv2.VideoCapture(cfg["index"], cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(cfg["index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"    ❌ {cam_name}: 无法打开")
        cameras[cam_name] = None
        continue
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
    cameras[cam_name] = cap
    print(f"    ✅ {cam_name}: {cfg['width']}x{cfg['height']}")

# ━━━ 4. 闭环控制 ━━━
print(f"\n{'='*60}")
print(f"  任务: {TASK_TEXT}")
print(f"  闭环控制: 每帧摄像头 → VLA推理 → 机械臂")
print(f"  Ctrl+C 停止（保留扭矩）")
print(f"{'='*60}\n")

# 打开摄像头预览窗口（与 genkiarm 一致）
cv2.namedWindow("phone", cv2.WINDOW_NORMAL)
cv2.namedWindow("laptop", cv2.WINDOW_NORMAL)
cv2.resizeWindow("phone", 480, 360)
cv2.resizeWindow("laptop", 480, 360)
# 窗口置顶
cv2.setWindowProperty("phone", cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty("laptop", cv2.WND_PROP_TOPMOST, 1)
print("  摄像头预览窗口已打开（phone / laptop）\n")


def read_angles(sc, ids):
    angles = []
    for sid in ids:
        pos, _, _ = sc.ReadPos(sid)
        angles.append((pos - 2048) / 2048 * 180.0)
    return angles


def write_angles(sc, ids, angles, speed=MOVING_SPEED, acc=MOVING_ACC):
    for sid, ang in zip(ids, angles):
        pos = int(ang / 180.0 * 2048 + 2048)
        pos = max(0, min(4095, pos))
        sc.WritePosEx(sid, pos, speed, acc)


step = 0
t_start = time.perf_counter()
prev_angles = read_angles(follower_sc, SERVO_IDS)

try:
    while True:
        t_step = time.perf_counter()

        # ── 读当前关节角 ──
        current_angles_raw = read_angles(follower_sc, SERVO_IDS)
        current_angles = torch.tensor(
            current_angles_raw, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)

        # ── 读摄像头（每帧都读，不缓存） ──
        images = {}
        display_frames = {}  # BGR 原图，用于显示
        for cam_name, cap in cameras.items():
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    display_frames[cam_name] = frame  # BGR 原图
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    images[cam_name] = img_tensor.unsqueeze(0).to(DEVICE)
                    continue
            images[cam_name] = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

        # 显示摄像头画面（每 3 帧刷新一次）
        if step % 3 == 0:
            for cam_name, frame in display_frames.items():
                cv2.imshow(cam_name, frame)
            cv2.waitKey(1)

        # ── 每帧都 VLA 推理（不缓存，纯闭环） ──
        cur_state = (current_angles - state_mean) / state_std

        observation = {
            "observation.images.camera1": images.get("phone"),
            "observation.images.camera2": images.get("laptop"),
            "observation.state": cur_state,
            "observation.language.tokens": lang_tokens,
            "observation.language.attention_mask": lang_mask,
        }

        with torch.no_grad():
            action_chunk = model.predict_action_chunk(observation)
        action = action_chunk[0, 0]  # 只取第 1 步
        target = (action * action_std + action_mean).cpu().tolist()

        # ── 安全限幅 ──
        for j in range(6):
            delta = target[j] - prev_angles[j]
            if abs(delta) > MAX_RELATIVE_TARGET:
                target[j] = prev_angles[j] + (MAX_RELATIVE_TARGET if delta > 0 else -MAX_RELATIVE_TARGET)

        prev_angles = target
        write_angles(follower_sc, SERVO_IDS, target)

        if step % 5 == 0:
            print(f"  {step:4d}: {[round(a,1) for a in target]}   cam={[round(images[c].mean().item(),2) if c in images else 0 for c in ['phone','laptop']]}")

        step += 1
        elapsed = time.perf_counter() - t_step
        sleep_time = (1.0 / FPS) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n\n⚠ 用户中断 — 扭矩保持（机械臂停在当前位置）")

finally:
    print("🔌 清理 ...")
    cv2.destroyAllWindows()
    for cap in cameras.values():
        if cap is not None:
            cap.release()
    follower_port.closePort()
    print(f"✅ 完成  |  步数: {step}  |  时长: {time.perf_counter()-t_start:.1f}s")
