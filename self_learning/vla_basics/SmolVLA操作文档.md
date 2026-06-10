# SmolVLA 操作文档

> 适用环境：Windows（本地采集/评估） + 云端 GPU（训练）  
> 本地：RTX 4060 Laptop 8GB（仅用于数据采集 + 推理，不训练）  
> 云端：A10G/A100 GPU（训练 SmolVLA，按量付费或免费 Colab）  
> 基于 LeRobot 官方 SmolVLA 流程，适配你的 SO100 机械臂

---

## 1. 环境安装

### 1.1 核心问题说明

**uv 和 CUDA torch 不兼容**。uv 的 PyPI 索引只有 CPU 版 torch，每次 `uv run` 或 `uv sync` 都会把 CUDA 版覆盖成 CPU 版。解决方案：用 `uv sync` 装其他依赖时跳过 torch，然后从本地缓存复制 CUDA 版。

### 1.2 安装步骤

```powershell
# 1. 确保 conda python312 环境已激活，当前在 d:\lerobot
conda activate python312
cd d:\lerobot

# 2. 锁定 Python 3.12（防止 uv 自动选 3.13，导致 cp312 的 torch 不兼容）
"3.12" | Out-File -FilePath .python-version -Encoding utf8 -NoNewline

# 3. 创建 .venv
uv venv --python 3.12

# 4. 安装依赖，跳过 torch 和 torchvision（不下 CPU 版）
uv sync --extra smolvla --extra dataset --no-install-package torch --no-install-package torchvision

# 5. 从本地缓存复制 CUDA 版 torch（D:\torch_cuda 是预下载的 GPU 版）
Remove-Item -Recurse -Force .venv\Lib\site-packages\torch -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .venv\Lib\site-packages\torchvision -ErrorAction SilentlyContinue
Copy-Item -Recurse D:\torch_cuda\torch .venv\Lib\site-packages\torch
Copy-Item -Recurse D:\torch_cuda\torchvision .venv\Lib\site-packages\torchvision
Copy-Item -Recurse D:\torch_cuda\torchgen .venv\Lib\site-packages\torchgen
Copy-Item -Recurse D:\torch_cuda\functorch .venv\Lib\site-packages\functorch

# 6. 验证（必须用 .venv\Scripts\python.exe，不能用 uv run！）
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# 正确输出: 2.6.0+cu124 CUDA: True
```

> **铁律**：本项目所有 Python 命令用 `.venv\Scripts\python.exe`，禁止使用 `uv run`（它会自动 uv sync 覆盖 CUDA torch）。

### 1.3 如果本地没有 CUDA torch 缓存

需要手动下载一次（约 2.5 GB）：

```powershell
# 下载到 D:\torch_cuda
pip install torch torchvision `
    --index-url https://download.pytorch.org/whl/cu124 `
    --target D:\torch_cuda `
    --python-version 312 `
    --implementation cp `
    --no-deps
```

> 网络不稳会反复断线，pip 有断点续传，多试几次。下载完按 1.2 的第 5 步复制进 .venv。

---

## 2. 采集数据（带语言标注）

SmolVLA 是**语言条件**模型，采集数据时**必须给每个 episode 加上任务描述**，这是它和 ACT 最核心的区别。

### 2.1 采集命令

```powershell
cd d:\lerobot

lerobot-record `
    --robot.type=so100_follower `
    --robot.port=COM6 `
    --robot.id=my_follower `
    --teleop.type=so100_leader `
    --teleop.port=COM7 `
    --teleop.id=my_leader `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, phone: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
    --display_data=true `
    --dataset.repo_id=pick/smola_test `
    --dataset.single_task="抓住黄色海绵块放到黑色盒子里" `
    --dataset.root=data `
    --dataset.fps=30 `
    --dataset.episode_time_s=30 `
    --dataset.reset_time_s=10 `
    --dataset.num_episodes=50 `
    --dataset.push_to_hub=false `
    --dataset.streaming_encoding=true
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--robot.type=so100_follower` | 从臂（被控制的机械臂） |
| `--robot.port=COM6` | 从臂串口（已改为你的实际端口） |
| `--robot.id=my_follower` | 从臂标识（用于校准文件） |
| `--teleop.type=so100_leader` | 主臂（你用手操作的遥操作臂） |
| `--teleop.port=COM7` | 主臂串口（已改为你的实际端口） |
| `--teleop.id=my_leader` | 主臂标识 |
| `--robot.cameras="{...}"` | 摄像头配置，建议 2 个：正面 + 手腕 |
| `--display_data=true` | 实时显示摄像头画面 |
| `--dataset.repo_id=pick/smola_test` | 数据集标识 |
| `--dataset.single_task="..."` | **最关键**——任务描述。SmolVLA 用它学习"听指令做事" |
| `--dataset.root=data` | 数据保存到 `data` 目录 |
| `--dataset.fps=30` | 帧率 30 帧/秒 |
| `--dataset.episode_time_s=30` | 每段 30 秒 |
| `--dataset.streaming_encoding=true` | 实时编码视频（省磁盘空间） |

> **端口说明**：你的 SO100 主臂和从臂各接一条 USB 线，在设备管理器中会显示两个不同的 COM 口。主臂（leader）一般是你用手掰的那个，从臂（follower）是执行动作的那个。填错的话机械臂不会动，换一下就行。

### 2.2 多任务采集

如果你想训练一个能听懂多条指令的模型，需要采集多个不同任务：

```powershell
# 任务 1: 抓取
lerobot-record `
    --robot.type=so100_follower --robot.port=COM6 --robot.id=my_follower `
    --teleop.type=so100_leader --teleop.port=COM7 --teleop.id=my_leader `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --dataset.repo_id=pick/grasp_block `
    --dataset.single_task="抓住桌子上的方块" `
    --dataset.root=data `
    --dataset.fps=30 --dataset.episode_time_s=20 --dataset.num_episodes=20

# 任务 2: 放置
lerobot-record `
    --robot.type=so100_follower --robot.port=COM6 --robot.id=my_follower `
    --teleop.type=so100_leader --teleop.port=COM7 --teleop.id=my_leader `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --dataset.repo_id=pick/place_block `
    --dataset.single_task="把方块放到碗里" `
    --dataset.root=data `
    --dataset.fps=30 --dataset.episode_time_s=20 --dataset.num_episodes=20

# 任务 3: 推物体
lerobot-record `
    --robot.type=so100_follower --robot.port=COM6 --robot.id=my_follower `
    --teleop.type=so100_leader --teleop.port=COM7 --teleop.id=my_leader `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --dataset.repo_id=pick/push_bottle `
    --dataset.single_task="把瓶子推到桌子左边" `
    --dataset.root=data `
    --dataset.fps=30 --dataset.episode_time_s=20 --dataset.num_episodes=20
```

> **采集原则**：每个任务至少 10-20 次演示，任务之间要有足够差异，让模型学会区分不同指令。参考：SmolVLA 官方在 5 个不同位置各采集了 10 个 episode，共 50 个。

---

## 3. 查看数据

```powershell
cd d:\lerobot

uv run lerobot-replay `
    --dataset.repo_id=pick/smola_test `
    --dataset.root=data `
    --episode=0
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--dataset.repo_id=pick/smola_test` | 数据集标识 |
| `--dataset.root=data` | 数据目录 |
| `--episode=0` | 查看第 0 个 episode |

---

## 4. 训练 SmolVLA（云端）

RTX 4060 8GB 显存太小，训练 SmolVLA 慢且容易 OOM。**推荐全部走云端**，本地只负责采集数据和部署推理。

### 4.0 前置：把数据上传到 HuggingFace Hub

云端训练需要能从 Hub 拉取数据。先把本地数据集推送上去：

```powershell
cd d:\lerobot

# 登录 HuggingFace（只需做一次）
uv run huggingface-cli login

# 推送数据集到 Hub
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset.from_preloaded('pick/smola_test', root='data')
ds.push_to_hub('你的用户名/smola_test')
"
```

> 如果没有 HuggingFace 账号，去 https://huggingface.co 免费注册一个。数据可以设为私有（private）。

### 4.1 方案 A：Google Colab（免费，推荐新手）

官方提供了开箱即用的 Colab Notebook，用免费 T4 GPU 即可训练。

🔗 **[一键打开 SmolVLA 训练 Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb)**

#### 操作步骤

1. 点击上面链接打开 Colab
2. 顶部菜单：**Runtime → Change runtime type → 选 T4 GPU**
3. 按 Cell 顺序运行，在训练配置 Cell 里改三个地方：
   - `dataset_repo_id` → 改成 `你的用户名/smola_test`
   - `steps` → 改成 `30000`
   - `batch_size` → 改成 `8`（T4 的 16GB 显存足够）
4. 训练完成后，模型自动存到你的 Google Drive

#### 训练时间参考（T4 GPU）

| batch_size | steps | 约耗时 |
|:----------:|:-----:|:------:|
| 8 | 30,000 | ~3-4 小时 |
| 8 | 50,000 | ~5-6 小时 |

> **缺点**：Colab 免费版可能断连（一般 4-6 小时上限），建议每 5000 步保存一次 checkpoint。

### 4.2 方案 B：HuggingFace Training Jobs（按量付费，速度快）

HuggingFace Hub 内置的云端训练服务，用 A10G（24GB）或 A100（80GB）。

#### 4.2.1 浏览器操作（最简单）

1. 访问你的数据集页面：`https://huggingface.co/datasets/你的用户名/smola_test`
2. 点击 **Train with AutoTrain** 按钮
3. 选择 **SmolVLA** 模板
4. 配置参数：
   - Hardware: **A10G Small**（$1.05/小时）
   - Training steps: `30000`
   - Batch size: `64`（显存大，尽情开）
5. 点击 **Start Training**，训练完自动推到 Hub

#### 4.2.2 命令行操作（高级）

```bash
pip install huggingface-hub

hf jobs run \
  --flavor a10g-small \
  --timeout 6h \
  --command '
    pip install lerobot[smolvla] && \
    lerobot-train \
      --policy.path=lerobot/smolvla_base \
      --dataset.repo_id=你的用户名/smola_test \
      --batch_size=64 \
      --steps=30000 \
      --output_dir=outputs/train/smolvla_pick \
      --policy.device=cuda \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.scheduler_decay_steps=30000 \
      --save_freq=5000
  ' \
  YOUR_HF_DATASET/pick/smola_test \
  YOUR_HF_USERNAME/smolvla_pick
```

> 注意：`hf jobs` 命令目前只在 Linux/Mac 终端的 bash 下工作。在 Windows 上建议用方案 4.2.1（浏览器）或方案 A（Colab）。

#### 训练时间参考（A10G GPU）

| batch_size | steps | 约耗时 | 约费用 |
|:----------:|:-----:|:------:|:------:|
| 64 | 20,000 | ~20 分钟 | ~$0.35 |
| 64 | 50,000 | ~1 小时 | ~$1.05 |

### 4.3 方案 C：AutoTrain（零代码，网页点点点）

AutoTrain 是 HuggingFace 的无代码训练平台，直接在网页上配置。

1. 打开 https://huggingface.co/autotrain
2. 创建新项目 → 选 **LeRobot SmolVLA Fine-Tuning**
3. 选择你的数据集 `你的用户名/smola_test`
4. 选择硬件（A10G Small 最划算）
5. 点 Start，等训练完成自动推 Hub

### 4.4 训练后下载模型到本地

训练完成后，模型会出现在你的 HuggingFace 主页。下载到本地：

```powershell
cd d:\lerobot

# 方法 1: huggingface-cli 下载
uv run huggingface-cli download 你的用户名/smolvla_pick --local-dir outputs/train/smolvla_pick

# 方法 2: 模型卡上直接下载 zip
# 浏览器打开 https://huggingface.co/你的用户名/smolvla_pick → 点 Download
```

### 4.5 云端训练 vs 本地训练

| | 本地 RTX 4060 | Colab T4 | HF Jobs A10G |
|---|---|---|---|
| 显存 | 8 GB | 16 GB | 24 GB |
| batch_size | 4 | 8 | 64 |
| 20k steps | ~3 小时 | ~2 小时 | ~20 分钟 |
| 费用 | 电费 | 免费 | ~$1/小时 |
| 是否可解冻 VLM | 否（OOM） | 勉强 | 轻松 |
| 适合 | 仅推理 | 入门体验 | 正式训练 |

---

## 5. 模型评估

### 5.1 真机评估（录制评估数据）

```powershell
cd d:\lerobot

lerobot-record `
    --robot.type=so100_follower `
    --robot.port=COM6 `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --dataset.repo_id=pick/eval_smolvla `
    --dataset.single_task="抓住黄色海绵块放到黑色盒子里" `
    --dataset.root=data `
    --dataset.fps=30 `
    --dataset.episode_time_s=30 `
    --dataset.reset_time_s=10 `
    --dataset.num_episodes=10 `
    --policy.path=outputs/train/smolvla_pick/checkpoints/last/pretrained_model
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--policy.path=outputs/.../pretrained_model` | 指定你训练好的模型路径 |
| `--dataset.num_episodes=10` | 评估 10 个 episode，统计成功率 |
| 其他参数 | 和数据采集一样 |

> **数据存储**：评估数据会保存到 `data/pick/eval_smolvla/`，可以事后回放分析哪里出错。

### 5.2 真机部署（只干活，不录数据）

```powershell
cd d:\lerobot

uv run lerobot-rollout `
    --strategy.type=base `
    --robot.type=so100_follower `
    --robot.port=COM6 `
    --robot.id=my_follower `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --task="抓住黄色海绵块放到黑色盒子里" `
    --policy.path=outputs/train/smolvla_pick/checkpoints/last/pretrained_model
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--task="..."` | 任务指令，用你训练时的同一条描述 |
| `--policy.path=outputs/.../pretrained_model` | 指定模型路径 |

### 5.3 低算力加速：RTC 模式（可选）

如果推理太慢（8GB 显存可能出现），开启 RTC（Real-Time Chunking）优化：

```powershell
uv run lerobot-rollout `
    --strategy.type=base `
    --robot.type=so100_follower --robot.port=COM6 `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --task="抓住黄色海绵块放到黑色盒子里" `
    --inference.type=rtc `
    --inference.rtc.execution_horizon=10 `
    --inference.rtc.max_guidance_weight=10.0 `
    --policy.path=outputs/train/smolvla_pick/checkpoints/last/pretrained_model
```

---

## 6. 回放训练好的模型

```powershell
cd d:\lerobot

# 回放评估数据
uv run lerobot-replay `
    --dataset.repo_id=pick/eval_smolvla `
    --dataset.root=data `
    --episode=0 `
    --policy.path=outputs/train/smolvla_pick/checkpoints/last/pretrained_model
```

---

## 7. 参数速查表

### 云端训练参数

| 参数 | Colab 推荐 | HF Jobs 推荐 | 说明 |
|------|:------:|:------:|------|
| `--policy.path` | `lerobot/smolvla_base` | `lerobot/smolvla_base` | 预训练模型 |
| `--batch_size` | 8 | 64 | Colab T4 16GB / A10G 24GB |
| `--steps` | 30000-50000 | 30000-50000 | 50 episodes 参考值 |
| `--optimizer.lr` | 1e-4 | 1e-4 | 微调学习率 |
| `--policy.scheduler_decay_steps` | 等于 `--steps` | 等于 `--steps` | 必须匹配 |
| `--policy.freeze_vision_encoder` | true | false（推荐） | A10G 显存够，解冻效果更好 |
| `--policy.train_expert_only` | true | false（推荐） | 同上 |
| `--save_freq` | 5000 | 5000 | 检查点保存频率 |

### 评估/部署参数

| 参数 | 推荐值 | 说明 |
|------|:------:|------|
| `--task` | 与训练相同的句子 | 任务指令必须一致 |
| `--dataset.num_episodes` | 10 | 评估 episode 数 |
| `--inference.rtc.execution_horizon` | 10 | RTC 执行窗口 |

### 数据集要求

| 项目 | 推荐值 | 说明 |
|------|:------:|------|
| Episodes | 50 | 起步值，每个位置 10 次 |
| Camera | 2 个 | 正面 + 手腕 |
| FPS | 30 | 标准帧率 |
| Episode 时长 | 20-45 秒 | 抓取/放置 30 秒足够 |
| 任务描述 | 具体动作短语 | 如"抓住黄色方块放到黑色盒子里" |

---

## 8. 常见问题

### Q: 为什么要用云端训练，4060 不行吗？

SmolVLA 虽然叫 "Small"，但 450M 参数对 8GB 显存来说仍然很吃紧：
- 冻结模式 batch=1 已占 ~4GB，batch=4 逼近上限
- 解冻视觉编码器直接 OOM
- batch 太小训练不稳定，收敛慢

云端 A10G 24GB 显存可以开 batch=64 + 解冻 VLM，20 分钟就能完成训练，效果还好得多。

### Q: 三种云端方案怎么选？

| 你的情况 | 推荐方案 |
|----------|----------|
| 没钱、想先试试 | **Colab（免费 T4）** |
| 想快速出结果、愿意花几块钱 | **HF Jobs A10G（~$1/次）** |
| 完全不懂命令行 | **AutoTrain（浏览器点点点）** |

### Q: 数据要上传到 HuggingFace Hub，隐私怎么办？

数据集可以设为 **Private**，只有你自己能访问。HF Jobs 也能读取私有数据集。免费账号也有私有仓库额度。

### Q: 训练和 ACT 有什么区别？

| | ACT | SmolVLA |
|---|---|---|
| 输入 | 图片 + 关节状态 | 图片 + 关节状态 + **语言指令** |
| 能力 | 单任务 | 多任务（听懂不同指令） |
| 预训练 | 无 | 预训练 VLM，微调收敛快 |
| 显存 | batch=4 约 0.94GB | batch=1 约 3.93GB |

### Q: 任务描述怎么写？

好的例子：
- "抓住桌上黄色海绵块放到黑色盒子里"（具体、有对象）
- "把方块推到桌子左边"（动作 + 方向）

坏的例子：
- "完成抓取任务"（太笼统）
- "pick"（太短）

### Q: scheduler_decay_steps 是什么？

学习率调度器的衰减总步数。如果你设 `--steps=10000` 但 `scheduler_decay_steps=30000`（默认值），LR 永远不会衰减到终点。**必须设置 `--policy.scheduler_decay_steps` 约等于 `--steps`。**

### Q: Colab 训练断连了怎么办？

Colab 免费版一般 4-6 小时会自动断开。对策：
1. 训练时挂上 Google Drive，checkpoint 自动存 Drive
2. 重新打开 Notebook，从最近的 checkpoint 继续
3. 或者直接用 HF Jobs，按量付费不会断连

---

## 9. 流程总结

```
本地数据采集(带语言标注) → 上传 HuggingFace Hub → 云端训练(Colab/HF Jobs) → 下载模型 → 本地真机评估 → 部署使用
      (1)                          (2)                     (3)                 (4)           (5)             (6)
```

**核心思路**：SmolVLA 在 ACT 的"图片→动作"基础上多加了一个"语言理解"环节。你采集数据时告诉它"这是在做什么"，训练后它就能听懂你的指令决定做什么动作。你现有的机械臂、摄像头、遥操作全部复用，唯一多做的事就是——**给每次演示配一句话**。

**训练这件事交给云端**：RTX 4060 干不好训练，但在本地跑推理（部署）完全够用。采完数据往 Hub 一推，Colab 或 HF Jobs 训练完下载回来，本地部署即可。

---

## 10. 错误排查记录

以下是实际安装过程中遇到的所有错误及解决方案，踩过的坑别再踩。

### 错误 1：`Torch not compiled with CUDA enabled`

**现象**：
```
AssertionError: Torch not compiled with CUDA enabled
```

**原因**：`uv run` 或 `uv sync` 从 PyPI 下载了 CPU 版 torch（`2.x.x+cpu`），覆盖了 CUDA 版。

**解决**：
1. 禁止使用 `uv run`，改用 `.venv\Scripts\python.exe`
2. 如果必须 `uv sync`，加 `--no-install-package torch --no-install-package torchvision`
3. 重新从 `D:\torch_cuda` 复制 CUDA torch 到 `.venv`

### 错误 2：`ModuleNotFoundError: No module named 'torch._strobelight'`

**现象**：
```
File "torch\__init__.py", line 57, in <module>
    from torch._utils_internal import (...)
File "torch\_utils_internal.py", line 11, in <module>
    from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler
ModuleNotFoundError: No module named 'torch._strobelight'
```

**原因**：只复制了 `torch` 和 `torchvision` 目录，漏了 `torchgen` 和 `functorch`。

**解决**：4 个目录全复制：
```powershell
Copy-Item -Recurse D:\torch_cuda\torch .venv\Lib\site-packages\torch
Copy-Item -Recurse D:\torch_cuda\torchvision .venv\Lib\site-packages\torchvision
Copy-Item -Recurse D:\torch_cuda\torchgen .venv\Lib\site-packages\torchgen
Copy-Item -Recurse D:\torch_cuda\functorch .venv\Lib\site-packages\functorch
```

### 错误 3：`.python-version` 文件编码异常

**现象**：
```
error: failed to read from file `.python-version`: stream did not contain valid UTF-8
```

**原因**：PowerShell 的 `echo "3.12" > .python-version` 写入的是 UTF-16 LE，uv 只能读 UTF-8。

**解决**：
```powershell
Remove-Item .python-version -Force
"3.12" | Out-File -FilePath .python-version -Encoding utf8 -NoNewline
```

### 错误 4：下载 torch 反复断线

**现象**：
```
IncompleteRead(482081536 bytes read, 2050226683 more expected)
peer closed connection without sending TLS close_notify
```

**原因**：你的网络到 `download.pytorch.org` 不稳定，2.5 GB 大文件下载容易断。

**解决**：pip 自带断点续传，重复运行直到成功。或者用浏览器下载 whl 文件后本地安装：
1. 浏览器打开 https://download.pytorch.org/whl/cu124
2. 搜 `torch-2.6.0+cu124-cp312-cp312-win_amd64.whl`，下载到本地
3. `pip install D:\下载路径\torch-2.6.0+cu124-cp312-cp312-win_amd64.whl`

### 错误 5：uv sync 解析到 `torch==2.6.0+cu124` 但项目要求 `>=2.7`

**现象**：
```
Because only torch<=2.6.0+cu124 is available and your project depends on
torch>=2.7, we can conclude that your project's requirements are unsatisfiable.
```

**原因**：Windows cu124 索引最高只有 torch 2.6.0，但 `pyproject.toml` 要求 `>=2.7`。

**解决**：已修改 `pyproject.toml` 将 torch 版本要求从 `>=2.7` 降为 `>=2.6`。

### 错误 6：uv run 每次都重新下载 2.5 GB torch

**现象**：跑任何带 `uv run` 的命令，终端都显示下载 2.36 GiB 的 torch，反复断线重试。

**原因**：`uv run` 内部先执行 `uv sync`，发现 torch 缺了/版本不对，就从 PyPI 重新下 CPU 版。

**解决**：永远不用 `uv run`。直接用 `.venv\Scripts\python.exe` 跑所有脚本。

### 错误 7：C 盘空间被吃光

**现象**：多次下载 torch 失败后，C 盘可用空间大幅减少。

**原因**：每次下载失败，半成品文件留在以下两个位置：
- `C:\Users\<用户名>\AppData\Local\uv\cache` — uv 的包缓存
- `C:\Users\<用户名>\AppData\Local\Temp\pip-*` — pip 的临时文件

**解决**：
```powershell
uv cache clean
Remove-Item -Recurse -Force $env:TEMP\pip-* -ErrorAction SilentlyContinue
```