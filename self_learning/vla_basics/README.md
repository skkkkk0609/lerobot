# SmolVLA 学习目录

学习目标：用 SmolVLA 实现"语言指令 → 动作"的完整闭环，适配 SO100 机械臂。

## 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `SmolVLA操作文档.md` | 📖 | **主文档**：完整操作流程，从采集到部署 |
| `PROJECT_STATE.md` | 📋 | 项目状态、硬件环境、踩坑记录 |
| `VLA_GUIDE.md` | 📝 | 概念速查：VLA vs ACT、云端方案选型 |
| `phase1_sim2model.py` | 🔧 | 模型自检：随机数据跑通 SmolVLA 推理链 |
| `phase6_teacher_forcing.py` | 🔬 | 离线验证：用训练数据回放评估模型精度 |

## 目录结构

```
vla_basics/
├── SmolVLA操作文档.md          ← 📖 从这里开始！
├── PROJECT_STATE.md            ← 硬件环境 & 踩坑记录
├── VLA_GUIDE.md                ← 概念速查
├── phase1_sim2model.py         ← 模型加载自检脚本
├── phase6_teacher_forcing.py   ← 离线验证脚本
└── _archived/                  ← 已归档的旧自学脚本
```

## 前置条件

- Windows + RTX 4060 8GB（本地采集/推理，训练走云端）
- SO100 机械臂 + USB 摄像头
- Python 3.12 + PyTorch CUDA
- 已安装 `lerobot[smolvla,dataset]`
- HuggingFace 账号（免费注册，用于云端训练）