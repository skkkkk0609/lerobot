"""阶段 1：用随机数据跑通 SmolVLA 推理链（无需硬件）
输入: 随机图片 + 随机状态 + 中文指令
输出: 50 步动作序列，展示模型如何从语言→动作
"""
import torch
from transformers import AutoTokenizer

from lerobot.policies.smolvla import SmolVLAPolicy

MODEL_ID = "lerobot/smolvla_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"设备: {DEVICE}")
print("正在加载 SmolVLA 模型...")
model = SmolVLAPolicy.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()
print("模型加载完成！\n")

total_params = sum(p.numel() for p in model.parameters()) / 1e6
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"总参数量: {total_params:.1f}M")
print(f"可训练参数量: {trainable_params:.1f}M")
print(f"动作块大小 (chunk_size): {model.config.chunk_size} 步")
print(f"输入图像尺寸: {model.config.resize_imgs_with_padding}\n")

# SmolVLA 内部用的是 SmolVLM tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

tasks = [
    "拿起红色方块",
    "把红色方块放到蓝色碗里",
    "推开桌上的瓶子",
    "按下绿色按钮",
]

for task in tasks:
    dummy_imgs = {
        "observation.images.camera1": torch.randn(1, 3, 256, 256, device=DEVICE),
        "observation.images.camera2": torch.randn(1, 3, 256, 256, device=DEVICE),
        "observation.images.camera3": torch.randn(1, 3, 256, 256, device=DEVICE),
    }
    dummy_state = torch.randn(1, 6, device=DEVICE)

    # 把中文指令转成 tokens
    encoded = tokenizer(task + "\n", return_tensors="pt", padding="max_length", max_length=48, truncation=True)
    lang_tokens = encoded["input_ids"].to(DEVICE)
    lang_mask = encoded["attention_mask"].bool().to(DEVICE)

    observation = {
        **dummy_imgs,
        "observation.state": dummy_state,
        "observation.language.tokens": lang_tokens,
        "observation.language.attention_mask": lang_mask,
    }

    with torch.no_grad():
        action = model.select_action(observation)

    print(f"指令: 「{task}」")
    print(f"  输出动作: shape={list(action.shape)}")
    print(f"  含义: 1 步 × {action.shape[1]} 维关节角度（select_action 每次返回单步）")
    print(f"  动作值: {action[0].cpu().tolist()}")
    print()

print("=== 阶段 1 完成：推理链路验证通过 ===")
print("下一步: 阶段 2 —— 把模型输出通过 WebSocket 发给你的模拟器")
