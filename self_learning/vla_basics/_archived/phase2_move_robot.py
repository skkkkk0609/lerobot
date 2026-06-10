"""阶段 2：VLA 模型控制黑马 GenkiPi 模拟器

运行前:
  1. 先启动模拟器前端: cd d:/lerobot/playground && npm run dev
  2. 浏览器打开 http://localhost:5173
  3. 然后跑这个脚本

流程:
  VLA 模型 → 动作值 → 角度映射 → WebSocket → 浏览器 3D 机械臂动起来
"""
import sys

sys.path.insert(0, ".")
import torch
import time

from websocket_robot import GenkiPiSimRobot
from lerobot.policies.smolvla import SmolVLAPolicy
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ━━━ Step 1: 启动 WebSocket 服务器 ━━━
print("启动 WebSocket 服务器...")
robot = GenkiPiSimRobot()
robot.start()

# 等待浏览器连接
print("等待浏览器连接...")
while not robot.is_connected:
    time.sleep(0.5)
print("开始加载模型...\n")

# ━━━ Step 2: 加载 VLA 模型 ━━━
print("加载 SmolVLA 模型...")
model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
model.to(DEVICE)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
print("模型就绪！\n")

# ━━━ Step 3: VLA 推理 + 控制机械臂 ━━━
TASK = "把机械臂伸向右边"
print(f"任务指令: 「{TASK}」")
print("模型将开始控制机械臂...\n")

for step in range(50):
    # 3a. 构造模型输入（随机图片占位，真实摄像头后替换）
    dummy_imgs = {
        "observation.images.camera1": torch.randn(1, 3, 256, 256, device=DEVICE),
        "observation.images.camera2": torch.randn(1, 3, 256, 256, device=DEVICE),
        "observation.images.camera3": torch.randn(1, 3, 256, 256, device=DEVICE),
    }
    dummy_state = torch.randn(1, 6, device=DEVICE)

    encoded = tokenizer(TASK + "\n", return_tensors="pt", padding="max_length", max_length=48, truncation=True)
    lang_tokens = encoded["input_ids"].to(DEVICE)
    lang_mask = encoded["attention_mask"].bool().to(DEVICE)

    observation = {
        **dummy_imgs,
        "observation.state": dummy_state,
        "observation.language.tokens": lang_tokens,
        "observation.language.attention_mask": lang_mask,
    }

    # 3b. VLA 推理
    with torch.no_grad():
        action = model.select_action(observation)

    # 3c. 动作值 → 角度（模型输出归一化值，映射到 ±90°）
    raw = action[0].cpu().tolist()
    angles = robot.scale_action_to_angles(raw)

    # 3d. 发送到模拟器
    robot.send_angles(angles)

    if step % 10 == 0:
        print(f"  步 {step+1}: 角度={[round(a, 1) for a in angles]}")

    time.sleep(0.1)

print("\n=== 阶段 2 完成：VLA 模型已通过 WebSocket 控制模拟器 ===")
print("下一步: 阶段 3 —— 采集你自己的 VLA 训练数据")
print()
robot.stop()
