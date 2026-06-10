# SmolVLA 概念速查

> 详细操作命令见 `SmolVLA操作文档.md`。本文档只做概念对比和方案选型。

---

## VLA vs ACT：一句话区别

```
ACT:    图片 → 动作（只会做一件事）
SmolVLA: 图片 + 中文指令 → 动作（能听懂多句话，做多件事）
```

你现有的机械臂、摄像头、数据采集全部复用。唯一多做的事：**给每次演示配一句话**。

---

## 训练方案选型

| | 本地 RTX 4060 | Colab T4 | HF Jobs A10G |
|---|---|---|---|
| 显存 | 8 GB | 16 GB | 24 GB |
| batch_size | 4 | 8 | 64 |
| 20k steps | ~3 小时 | ~2 小时 | ~20 分钟 |
| 费用 | 电费 | 免费 | ~$1/次 |
| 是否可解冻 VLM | ❌ OOM | ⚠️ 勉强 | ✅ 轻松 |
| 推荐场景 | 仅推理 | 入门体验 | **正式训练** |

**结论**：本地 4060 只负责**采集数据 + 部署推理**，训练全部甩云端。

---

## 三种云端方案对比

| 方案 | 门槛 | 速度 | 费用 | 推荐人群 |
|------|:--:|:--:|:--:|------|
| **Google Colab** | 极低 | ⭐⭐ | 免费 | 首次尝试 |
| **HF Jobs 浏览器** | 低 | ⭐⭐⭐ | ~$1/次 | 想快速出结果 |
| **AutoTrain** | 零 | ⭐⭐⭐ | ~$1/次 | 不想碰命令行 |

---

## 数据要求

| 项目 | 推荐值 |
|------|:------:|
| Episodes | 50（5 个位置 × 10 次） |
| Cameras | 2 个（正面 + 手腕） |
| FPS | 30 |
| Episode 时长 | 20-45 秒 |
| 任务描述 | 具体动作短语，如"抓住黄色方块放到黑色盒子里" |

---

## 官方资源

| 资源 | 链接 |
|------|------|
| SmolVLA 论文 | https://arxiv.org/abs/2506.01844 |
| Colab 训练 Notebook | https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb |
| 官方 SmolVLA 文档 | `docs/source/smolvla.mdx` |
| 官方参考数据集 | https://huggingface.co/datasets/lerobot/svla_so100_pickplace |
| LeRobot 用户指南 | `AGENT_GUIDE.md` |