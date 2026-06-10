# SmolVLA 训练技术总结文档 (Colab)

## 1. 环境配置与依赖管理
- **核心挑战**：`lerobot` 库版本更新频繁，导致 `lerobot-train` 入口命令有时无法被系统路径识别。
- **解决方案**：采用 `python3 -m lerobot.scripts.lerobot_train` 的方式启动训练，这确保了 Python 解释器能正确处理包内的相对导入，并手动将 `src` 目录加入 `PYTHONPATH`。
- **依赖修复**：手动安装 `draccus` 和 `lerobot[train,dataset,smolvla]` 以确保 VLM 相关组件（如 SmolVLM）的完整性。

## 2. 动态路径解析 (Dynamic Pathing)
- **挑战**：代码库结构变动导致 `train.py` 可能出现在不同目录下。
- **解决方案**：实现了一套自动化的路径搜索逻辑，遍历可能的位置（包括 `src/lerobot/scripts/`），动态锁定脚本位置，提高了代码的鲁棒性。

## 3. 性能优化 (Tesla T4 适配)
- **显存管理**：将 `batch_size` 设为 **16** 以适配 T4 GPU (16GB VRAM)。
- **计算加速**：启用混合精度训练 `--policy.use_amp=true`，显著提升了训练速度。
- **数据瓶颈**：设置 `num_workers=1` 避免 Colab CPU 调度冲突导致的训练卡顿。

## 4. 数据与认证
- **安全访问**：使用 Colab Secrets (`HF_TOKEN`) 安全加载 Hugging Face 令牌。
- **本地同步**：使用 `snapshot_download` 将数据集完整镜像到本地 `/content`，避免了训练过程中的流式传输延迟。

## 5. 训练状态回溯
- **Loss 趋势**：模型初始 Loss 为 **1.388**，经过约 3000 步训练后降至 **0.106**，收敛情况良好。
- **输出路径**：所有模型权重和日志保存在 `outputs/train/my_smolvla_optimized`。
