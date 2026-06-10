# VLA 学习项目状态文档

> 给新聊天窗口的上下文：读完本文档即可了解项目全貌。

---

## 项目目标

用 LeRobot 官方 SmolVLA 流程，在 SO100 机械臂上实现"语言指令 → 动作"。

---

## 硬件环境

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) |
| 系统 | Windows |
| 机械臂 | SO100 (GenkiPi)，6 关节 + 夹爪 |
| 摄像头 | USB 摄像头 ×2（正面 + 手腕） |
| Python 环境 | **uv** (LeRobot 官方包管理) |
| 训练 | **云端**（Colab T4 免费 / HF Jobs A10G 付费），4060 不训练 |

---

## 目录结构

```
d:\lerobot\                          ← LeRobot 官方仓库 (clone)

d:\lerobot\self_learning\            ← 学习项目
  └── vla_basics\                    ← SmolVLA 学习核心
      ├── SmolVLA操作文档.md         ← 📖 完整操作流程（从这里开始）
      ├── PROJECT_STATE.md           ← 本文档
      ├── VLA_GUIDE.md               ← 概念速查
      ├── phase1_sim2model.py        ← 模型加载自检
      ├── phase6_teacher_forcing.py  ← 离线验证
      └── _archived/                 ← 已归档的旧脚本
```

---

## 当前进度

### 已废弃（4060 本地训练路线）

旧方案用 websocket 模拟器 + 自定义 PyTorch 训练，和官方流程完全脱节，已全部归档到 `_archived/`。

### 当前方案（官方标准流程）

| 步骤 | 对应文档章节 | 状态 |
|------|-------------|:--:|
| 1. 环境安装 | 操作文档 §1 | ⬜ 需要装 CUDA 版 PyTorch |
| 2. 采集数据 | 操作文档 §2 | ⬜ `lerobot-record` 录制 |
| 3. 上传 Hub | 操作文档 §4.0 | ⬜ 推送到 HuggingFace |
| 4. 云端训练 | 操作文档 §4.1-4.3 | ⬜ Colab 或 HF Jobs |
| 5. 下载模型 | 操作文档 §4.4 | ⬜ 下载到本地 |
| 6. 真机部署 | 操作文档 §5.2 | ⬜ `lerobot-rollout` |

---

## 数据管理

```
D:\arm_robot_begin\genkiarm\data\pick\
  ├── so100_test/      ← 第1次录制（50 episodes, PNG, 已不用）
  ├── so100_v2/         ← 第2次录制（已不用）
  ├── so100_v3/         ← 第3次录制 ✅ 50 episodes, MP4 双摄像头
  ├── eval_so100_v2/    ← 评估数据
  └── eval_so100_v3/    ← 评估数据

# 未来使用 lerobot-record 采集的数据会放在：
d:\lerobot\data\
  └── pick/             ← 按照操作文档 §2 采集
```

---

## 关键跑过的坑（仍有效）

### SmolVLA 模型相关

1. **图片 key** → SmolVLA 用 `observation.images.camera1/2/3`，不是 `observation.images.top`
2. **图片尺寸** → 必须 256×256（模型内部会 pad 到 512×512）
3. **语言输入必须 tokenize** → 使用 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 的 tokenizer，格式 `task + "\n"`, max_length=48
4. **attention_mask 必须是 bool** → `.bool().to(DEVICE)`
5. **action shape** → `select_action()` 返回 `[1, 6]` 单步

### 训练相关

6. **HF Hub 连接超时** → 离线环境下需要 `local_files_only=True`（但云端训练不存在此问题）
7. **scheduler_decay_steps** → 必须约等于 `--steps`，否则 LR 不衰减

### 数据相关

8. **旧数据 VideoFrame 类型** → 旧版 genkiarm 注册了自定义 VideoFrame 类型，新版 datasets 不认识，需手动注册（仅用到旧数据时需要）

---

## 运行命令速查

```powershell
cd d:\lerobot

# 模型自检
uv run python self_learning\vla_basics\phase1_sim2model.py

# 离线验证
uv run python self_learning\vla_basics\phase6_teacher_forcing.py

# 详细操作命令见 SmolVLA操作文档.md
```