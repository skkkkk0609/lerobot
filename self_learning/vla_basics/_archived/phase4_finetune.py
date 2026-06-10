"""阶段 4: Fine-tune SmolVLA（直接 PyTorch 训练）

=== 做了什么 ===
  避免复杂的 LeRobotDataset 创建流程，直接：
  1. 读取 data/vla_from_v3 的 arrow 数据（state/action/task）
  2. 从 MP4 视频按需解码对应帧（不预加载到内存）
  3. Tokenize 中文任务描述
  4. 喂给 SmolVLA.forward() 训练

=== 训练样本格式 ===
  采用 ACT 风格的 chunk 训练：
    Frame_N 的画面 + State_N  →  预测 [Action_N, Action_N+1, ..., Action_N+CHUNK_SIZE-1]

  滑动窗口 stride=1，每一帧都会作为观测喂给模型一次：
    idx=0: Frame_0  → Action_0..19
    idx=1: Frame_1  → Action_1..20
    idx=2: Frame_2  → Action_2..21
    ...

=== 运行 ===
  conda activate python312
  cd d:/lerobot
  python self_learning/vla_basics/phase4_finetune.py

=== 配置 ===
  RTX 4060 8GB: batch_size=1, chunk_size=20, num_vlm_layers=16
  约 40-60 分钟训练完成 5000 步
"""

import json
import os
import sys
import time
from pathlib import Path

# 强制 HuggingFace 离线（所有模型都在本地缓存里）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ═══════════════════════════════════════════════════════════════
# 配置参数
# ═══════════════════════════════════════════════════════════════

# 数据路径
ANNOTATED_DATA = Path("data/vla_from_v3")       # arrow + meta_data 所在目录
VIDEO_DIR = ANNOTATED_DATA / "videos"            # MP4 视频目录
OUTPUT_DIR = Path("outputs/train/smolvla_sponge2box")  # 模型输出目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 训练超参数 ──
BATCH_SIZE = 2           # 当前代码实际用 batch_size=1 (DataLoader 设置)
NUM_STEPS = 5000         # 总训练步数。1 epoch ≈ 19202 样本 (stride=1)
LR = 1e-4                # 学习率
GRAD_CLIP = 10.0         # 梯度裁剪阈值，防止梯度爆炸
SAVE_EVERY = 1000        # 每 1000 步保存一次 checkpoint
LOG_EVERY = 50           # 每 50 步打印一次日志

# ── 模型参数 ──
CHUNK_SIZE = 20          # 每次预测未来多少步的动作 (默认 50，减小以省显存)
                          # chunk 越大 → 时序信息越丰富，但显存占用越大
IMAGE_SIZE = 256          # 图像缩放到此尺寸，SmolVLA 内部会再 pad→512
CAMERA_NAMES = ["phone", "laptop"]  # 两路摄像头
TASK_TEXT = "抓取黄色海绵块放到黑色盒子里"  # 中文任务描述，会 tokenize 后喂给 VLM

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Dataset: 按需解码的训练数据加载器
# ═══════════════════════════════════════════════════════════════
class VLADataset(Dataset):
    """VLA 训练数据集。

    核心设计:
      - state/action 在 __init__ 时一次性预加载到内存（仅 ~1MB，忽略不计）
      - 视频帧在 __getitem__ 时从 MP4 按时间戳实时解码（不预缓存，省内存）
      - 滑动窗口 stride=1：每帧画面都作为观测喂给模型

    数据流:
      arrow 文件 (state, action, timestamp, episode_index)
        ↓ load_from_disk → 预加载到 self.states, self.actions, self.timestamps
      MP4 视频
        ↓ __getitem__ → decode_video_frames(按时间戳 seek) → 单帧 tensor
    """

    def __init__(self, data_path, video_dir, camera_names, image_size, task_text, tokenizer, chunk_size):
        """
        参数:
          data_path: arrow 数据目录 (如 data/vla_from_v3)
          video_dir: MP4 视频目录
          camera_names: 摄像头 key 列表，如 ["phone", "laptop"]
          image_size: 图像缩放到 (image_size, image_size)
          task_text: 中文任务描述，所有样本共享
          tokenizer: HuggingFace tokenizer，用于编码任务文本
          chunk_size: 每个样本预测未来多少步 action
        """
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.camera_names = camera_names
        self.video_dir = Path(video_dir)

        # ── 步骤 1: 从 arrow 文件加载结构化数据 ──
        print("  加载 arrow 数据...")
        self.data = load_from_disk(str(data_path))
        self.data = self.data.sort("index")  # 按全局 index 排序，保证顺序

        # ── 步骤 2: 预编码任务文本 ──
        # 所有样本共享同一任务描述，所以只需 tokenize 一次
        # tokenizer 输入: "抓取黄色海绵块放到黑色盒子里\n"
        # tokenizer 输出: input_ids (1, 48), attention_mask (1, 48)
        encoded = tokenizer(task_text + "\n", return_tensors="pt", padding="max_length",
                           max_length=48, truncation=True)
        self.lang_tokens = encoded["input_ids"][0]      # (48,) int
        self.lang_mask = encoded["attention_mask"][0].bool()  # (48,) bool

        # ── 步骤 3: 预加载 state 和 action 到内存 ──
        # state: 机械臂 6 个关节的当前角度
        # action: 遥操作主臂的 6 个目标角度（即"正确答案"）
        # 共 ~22000 帧 × 12 个 float32 = ~1MB，完全可以常驻内存
        print("  预加载 state/action...")
        self.states = torch.tensor(
            [row["observation.state"] for row in self.data], dtype=torch.float32
        )  # (N, 6)
        self.actions = torch.tensor(
            [row["action"] for row in self.data], dtype=torch.float32
        )  # (N, 6)
        self.timestamps = torch.tensor(
            [row["timestamp"] for row in self.data], dtype=torch.float32
        )  # (N,) — 每帧的时间戳，用于视频 seek

        # ── 步骤 4: 构建 episode 索引映射 ──
        # 用于快速定位某个 episode 的帧范围（结合箭头中的 episode_index 列）
        episodes = sorted(set(row["episode_index"] for row in self.data))
        self.ep_start_idx = {}   # episode → 该集第一个全局索引
        self.ep_end_idx = {}     # episode → 该集最后一个全局索引 + 1
        for ep in episodes:
            ep_rows = [i for i, row in enumerate(self.data)
                       if row["episode_index"] == ep]
            self.ep_start_idx[ep] = ep_rows[0]
            self.ep_end_idx[ep] = ep_rows[-1] + 1

        # 视频路径缓存（同一个 episode 的视频文件只需检查一次）
        self._video_paths = {}          # (ep, cam) → Path
        self._video_paths_checked = set()  # 已检查过存在的文件

        # ── 步骤 5: 计算归一化统计量 ──
        # 关键！官方训练计算 mean/std 对 state 和 action 做归一化
        # 不做归一化 → 大范围关节(shoulder_lift: ±93°)主导 loss
        # → 小范围关节(gripper: 0~60°, wrist_roll: ±41°)被忽略 → 输出趋近均值
        self.state_mean = self.states.mean(dim=0)   # (6,)
        self.state_std = self.states.std(dim=0).clamp(min=1e-8)
        self.action_mean = self.actions.mean(dim=0)  # (6,)
        self.action_std = self.actions.std(dim=0).clamp(min=1e-8)
        print(f"  state  mean: {self.state_mean.tolist()}")
        print(f"  state  std:  {self.state_std.tolist()}")
        print(f"  action mean: {self.action_mean.tolist()}")
        print(f"  action std:  {self.action_std.tolist()}")

        # ── 步骤 6: 计算样本数 ──
        # 滑动窗口 stride=1，总样本数 = 总帧数 - chunk_size
        # 例: 19222 帧, chunk_size=20 → 19202 个样本
        self.num_chunks = max(0, len(self.data) - self.chunk_size)
        print(f"  ✅ {len(self.data)} 帧, {self.num_chunks} 个样本就绪（按需解码，stride=1）")

    def _get_video_path(self, episode, cam):
        """获取指定 episode 和摄像头的 MP4 文件路径。
        只在第一次访问时检查文件是否存在。
        """
        key = (episode, cam)
        if key not in self._video_paths:
            path = self.video_dir / f"observation.images.{cam}_episode_{episode:06d}.mp4"
            self._video_paths[key] = path
        if key not in self._video_paths_checked:
            if not self._video_paths[key].exists():
                print(f"    ⚠ 视频不存在: {self._video_paths[key].name}")
            self._video_paths_checked.add(key)
        return self._video_paths[key]

    def __len__(self):
        """返回样本总数。stride=1 时 = 总帧数 - chunk_size。"""
        return self.num_chunks

    def __getitem__(self, idx):
        """取第 idx 个训练样本。

        idx 直接就是起始帧的全局索引（0 ~ len(data)-chunk_size）。

        返回:
          dict {
            "state":   (chunk_size, 6)   chunk 内所有帧的关节角度
            "action":  (chunk_size, 6)   chunk 内所有帧的目标角度（训练标签）
            "images":  {cam: (3, H, W)}  每路摄像头第 0 帧的图像
            "lang_tokens": (48,)         任务文本 token ids
            "lang_mask":   (48,) bool    token 有效位掩码
          }

        注意:
          - 只解码第 0 帧的图像，因为 SmolVLA 只用当前帧观测做推理
          - state 和 action 是完整 chunk（供 loss 计算用）
        """
        start = idx
        end = min(start + self.chunk_size, len(self.data))
        indices = list(range(start, end))

        # 取 chunk 范围内所有帧的 state 和 action
        all_states = (self.states[indices] - self.state_mean) / self.state_std     # 归一化
        all_actions = (self.actions[indices] - self.action_mean) / self.action_std  # 归一化

        # 只解码第 0 帧的图像（SmolVLA 只用当前帧观测，不处理多帧视频序列）
        all_images = {}
        ts_first = self.timestamps[start].item()  # 起始时间戳

        for cam in self.camera_names:
            ep = int(self.data[indices[0]]["episode_index"])
            vid_path = self._get_video_path(ep, cam)

            try:
                from lerobot.datasets.video_utils import decode_video_frames
                # decode_video_frames: 从 MP4 按时间戳 seek 到对应帧并解码
                # 返回: (1, 3, H_orig, W_orig) uint8 tensor（按原始分辨率）
                frames = decode_video_frames(
                    vid_path, [ts_first], tolerance_s=0.1,
                    backend="pyav", return_uint8=True,
                )
                # 归一化 + 缩放到 IMAGE_SIZE
                frame = frames[0].float() / 255.0  # (3, H_orig, W_orig) -> [0, 1]
                frame = torch.nn.functional.interpolate(
                    frame.unsqueeze(0), size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)  # (3, 256, 256)
            except Exception as e:
                # 解码失败时用全零图像填充（极少发生，不影响训练整体）
                if start < 500:
                    print(f"    ⚠ 视频解码失败 ep={ep} cam={cam}: {e}")
                frame = torch.zeros(3, self.image_size, self.image_size)

            all_images[cam] = frame  # (3, 256, 256)

        return {
            "state": all_states,
            "action": all_actions,
            "images": all_images,
            "lang_tokens": self.lang_tokens,
            "lang_mask": self.lang_mask,
        }


# ═══════════════════════════════════════════════════════════════
# 训练主函数
# ═══════════════════════════════════════════════════════════════
def train():
    print("=" * 60)
    print("  阶段 4: Fine-tune SmolVLA")
    print("=" * 60)
    print(f"  设备: {DEVICE}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  steps: {NUM_STEPS}")
    print()

    # ── 1. 加载 tokenizer ──
    # 用于把中文任务描述编码为 VLM 可理解的 token ids
    # 和预训练时的 tokenizer 一致 (SmolVLM2-500M)
    print("📝 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        local_files_only=True,  # 离线模式，从本地缓存加载
    )

    # ── 2. 创建 Dataset 和 DataLoader ──
    print("📂 加载数据...")
    dataset = VLADataset(
        ANNOTATED_DATA, VIDEO_DIR, CAMERA_NAMES, IMAGE_SIZE, TASK_TEXT, tokenizer, CHUNK_SIZE
    )

    # num_workers=2:  2 个子进程并行解码视频（遮掩 IO 延迟）
    # prefetch_factor=2: 每个 worker 提前预取 2 个 batch
    # persistent_workers=True: 复用 worker 进程，避免每个 epoch 重启
    # pin_memory=True: 把数据 pin 到 CPU 锁页内存，加速 CPU→GPU 传输
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=2, prefetch_factor=2,
        pin_memory=True, persistent_workers=True,
    )

    # ── 3. 加载预训练的 SmolVLA 模型 ──
    print("🤖 加载 SmolVLA 模型...")
    sys.path.insert(0, str(Path("src")))  # 让 import 能找到 lerobot 模块
    from lerobot.policies.smolvla import SmolVLAPolicy

    # SmolVLA 架构: SmolVLM2(VLM) + Action Expert
    # - SmolVLM2: 视觉编码器(SigLIP) + 语言编码器 + Transformer
    # - Action Expert: 额外的 action 预测头，通过交叉注意力与 VLM 交互
    model = SmolVLAPolicy.from_pretrained(
        "lerobot/smolvla_base",
        local_files_only=True,
    )

    # 冻结策略: 只训练 Action Expert 和 state_proj
    # - freeze_vision_encoder=True:  冻结 VLM 的视觉编码器（SigLIP）
    # - train_expert_only=True:      只训练 Action Expert 部分
    # - train_state_proj=True:       训练 state 投影层（把 6 维关节角度映射到 VLM 维度）
    # 总可训练参数约 100M / 总参数约 500M+
    model.config.freeze_vision_encoder = True
    model.config.train_expert_only = True
    model.config.train_state_proj = True
    model.config.chunk_size = CHUNK_SIZE          # 覆盖 base model 默认的 50
    model.config.n_action_steps = CHUNK_SIZE      # 覆盖 base model 默认的 50
    model.to(DEVICE)
    model.train()  # 切换到训练模式（启用 dropout 等）

    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── 4. 创建优化器 ──
    # AdamW: Adam + weight decay 解耦，防止过拟合
    # betas=(0.9, 0.95): 和 SmolVLA 论文一致
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )

    print(f"\n{'='*60}")
    print(f"  开始训练...")
    print(f"{'='*60}\n")

    step = 0                # 当前训练步数
    total_loss = 0.0        # LOG_EVERY 窗口内的累计 loss
    save_loss = 0.0         # SAVE_EVERY 窗口内的累计 loss（不被 LOG_EVERY 清零）
    best_loss = float("inf")  # 历史最佳 loss，用于保存 best_model.pt
    t0 = time.time()        # 训练开始时间

    # 摄像头 key 映射: 把数据集中的 "phone"/"laptop" 映射到 SmolVLA 期望的 key 名
    # SmolVLA 的预训练 config 中 image_features 是:
    #   "observation.images.camera1", "observation.images.camera2", ...
    camera_key_map = {
        "phone": "observation.images.camera1",
        "laptop": "observation.images.camera2",
    }

    # ── 5. 训练主循环 ──
    while step < NUM_STEPS:
        for batch_raw in dataloader:
            if step >= NUM_STEPS:
                break

            # ═══════════════════════════════════════
            # 构建模型输入 batch
            # ═══════════════════════════════════════
            # batch_raw 来自 DataLoader，每个字段有 batch 维度(1, ...)
            C = CHUNK_SIZE
            state_0 = batch_raw["state"][0].to(DEVICE)        # (C, 6) — 当前 chunk 的 state
            action_0 = batch_raw["action"][0].to(DEVICE)      # (C, 6) — 当前 chunk 的 action（标签）

            batch = {}

            # 图像: 只用 chunk 第 0 帧
            # SmolVLA 的 prepare_images() 只取 [:, -1, :, :, :] 最后一帧
            # 所以我们只喂单帧: (1, 3, 256, 256)
            for src_cam, dst_cam in camera_key_map.items():
                img = batch_raw["images"][src_cam]            # (3, H, W)
                batch[dst_cam] = img.unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

            # state: 第 0 帧的关节角度作为"当前观测"
            batch["observation.state"] = state_0[:1].to(DEVICE)   # (1, 6)

            # action: 全部 chunk 步（模型要预测的未来目标角度）
            batch["action"] = action_0.unsqueeze(0).to(DEVICE)     # (1, C, 6)

            # 语言: 任务描述 token 化结果
            batch["observation.language.tokens"] = batch_raw["lang_tokens"].to(DEVICE)        # (1, 48)
            batch["observation.language.attention_mask"] = batch_raw["lang_mask"].to(DEVICE)  # (1, 48) bool

            # ═══════════════════════════════════════
            # 前向传播 → 反向传播 → 参数更新
            # ═══════════════════════════════════════
            # model.forward() 内部流程:
            #   1. prepare_images() → SigLIP 视觉编码 → 图像特征
            #   2. prepare_state() → MLP 投影 → state 特征
            #   3. embed_prefix() → 拼接 图像+语言+state → VLM 前缀
            #   4. embed_suffix() → action 加噪 + 时间嵌入 → VLM 后缀
            #   5. VLM forward → 交叉注意力融合 → 预测 noise
            #   6. flow matching loss: MSE(预测 velocity, 真实 velocity)
            loss, _ = model.forward(batch)

            # 反向传播
            optimizer.zero_grad()           # 清空上一步的梯度
            loss.backward()                 # 计算梯度
            torch.nn.utils.clip_grad_norm_(  # 梯度裁剪，防止梯度爆炸
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
            )
            optimizer.step()                # 更新权重

            total_loss += loss.item()
            save_loss += loss.item()
            step += 1

            # ── 打印日志 ──
            if step % LOG_EVERY == 0:
                avg_loss = total_loss / LOG_EVERY
                elapsed = time.time() - t0
                sps = step / elapsed if elapsed > 0 else 0  # steps per second
                print(f"  step {step:5d}/{NUM_STEPS} | loss: {avg_loss:.6f} | {sps:.1f} steps/s | {elapsed:.0f}s")
                total_loss = 0.0  # 重置累计 loss

            # ── 保存 checkpoint ──
            if step % SAVE_EVERY == 0:
                avg_loss_save = save_loss / SAVE_EVERY
                save_path = OUTPUT_DIR / f"checkpoint_{step:05d}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss_save,
                }, save_path)
                print(f"  💾 已保存: {save_path}")
                # 同时保存归一化参数
                norm_path = OUTPUT_DIR / "norm_stats.pt"
                torch.save({
                    "state_mean": dataset.state_mean,
                    "state_std": dataset.state_std,
                    "action_mean": dataset.action_mean,
                    "action_std": dataset.action_std,
                }, norm_path)

                if avg_loss_save < best_loss:
                    best_loss = avg_loss_save
                    best_path = OUTPUT_DIR / "best_model.pt"
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "loss": best_loss,
                    }, best_path)
                    print(f"  🏆 最佳模型: {best_path}")

                save_loss = 0.0  # 重置 SAVE_EVERY 窗口

    # ── 6. 训练完成，保存最终模型 ──
    final_path = OUTPUT_DIR / "final_model.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ 训练完成！总耗时: {elapsed/60:.1f} 分钟")
    print(f"  最终模型: {final_path}")
    print(f"  最佳模型: {OUTPUT_DIR / 'best_model.pt'}")
    print()

    # 下一步指引
    print("下一步: 阶段 5 — 用训练的模型控制机械臂")
    print("  python self_learning/vla_basics/phase5_deploy.py")


if __name__ == "__main__":
    train()
