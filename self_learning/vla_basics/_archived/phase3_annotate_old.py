"""给旧的模仿学习数据添加语言标签，转为 VLA 格式

=== 做什么 ===
  旧数据: 50 episodes x 抓取黄色海绵块
  问题: 没有语言描述（task 字段）
  解决: 用 datasets 加载 → 添加 task → 新建数据集

=== 运行 ===
  conda activate python312
  cd d:/lerobot
  python self_learning/vla_basics/phase3_annotate_old.py
"""

import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
from datasets import Dataset, Features, Value, Sequence, load_from_disk
from datasets.features.features import register_feature

# 注册旧版 VideoFrame 类型（genkiarm 旧数据用了这个自定义类型）
@dataclass
class VideoFrame:
    pa_type = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)
    def __call__(self):
        return self.pa_type

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    register_feature(VideoFrame, "VideoFrame")

# ═══ 配置 ═══
OLD_PATH = Path(r"D:\arm_robot_begin\genkiarm\data\pick\so100_v3\train")
# 视频在上一级目录
OLD_VIDEOS = Path(r"D:\arm_robot_begin\genkiarm\data\pick\so100_v3\videos")
NEW_PATH = Path("data/vla_from_v3")
TASK_TEXT = "抓取黄色海绵块放到黑色盒子里"
FPS = 30

# ═══ 主流程 ═══
def main():
    print("=" * 60)
    print("  给旧数据添加语言标签 -> VLA 格式")
    print("=" * 60)
    print(f"  源: {OLD_PATH}")
    print(f"  目标: {NEW_PATH}")
    print(f"  任务: {TASK_TEXT}")
    print()

    # 1. 用 datasets 加载
    print("📂 读取原始数据...")
    old_ds = load_from_disk(str(OLD_PATH))

    # 排除 VideoFrame 类型的图像列
    safe_cols = [c for c in old_ds.column_names if not c.startswith("observation.images")]
    old_ds = old_ds.select_columns(safe_cols)

    episodes = sorted(old_ds.unique("episode_index"))
    print(f"  {len(old_ds)} 帧, {len(episodes)} episodes")
    print(f"  列: {old_ds.column_names}")
    print()

    # 2. 逐 episode 提取 + 加标签
    print("🏗 构建新数据（添加 task 字段）...")
    rows = []
    for ep in episodes:
        ep_data = old_ds.filter(lambda x: x["episode_index"] == ep)
        ep_data = ep_data.sort("frame_index")
        for row in ep_data:
            rows.append({
                "observation.state": list(row["observation.state"]),
                "action": list(row["action"]),
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "timestamp": float(row["timestamp"]),
                "index": int(row["index"]),
                "task": TASK_TEXT,
            })

    features = Features({
        "observation.state": Sequence(Value("float32"), length=6),
        "action": Sequence(Value("float32"), length=6),
        "episode_index": Value("int64"),
        "frame_index": Value("int64"),
        "timestamp": Value("float32"),
        "index": Value("int64"),
        "task": Value("string"),
    })

    new_ds = Dataset.from_list(rows, features=features, split="train")
    print(f"  ✅ {len(new_ds)} 帧")
    print()

    # 3. 保存
    print("💾 保存新数据集...")
    NEW_PATH.mkdir(parents=True, exist_ok=True)
    new_ds.save_to_disk(str(NEW_PATH))
    print(f"  ✅ {NEW_PATH}")
    print()

    # 4. 复制视频
    print("🎬 复制视频文件...")
    new_videos = NEW_PATH / "videos"
    new_videos.mkdir(exist_ok=True)
    vid_count = 0
    for vid in sorted(OLD_VIDEOS.iterdir()):
        if vid.is_file():
            shutil.copy2(str(vid), str(new_videos / vid.name))
            vid_count += 1
    print(f"  {vid_count} 个文件")
    print()

    # 5. 计算 stats
    print("📊 计算统计量...")
    states = np.array([r["observation.state"] for r in rows], dtype=np.float32)
    actions = np.array([r["action"] for r in rows], dtype=np.float32)

    meta_dir = NEW_PATH / "meta_data"
    meta_dir.mkdir(exist_ok=True)

    stats = {
        "observation.state/mean": states.mean(axis=0).tolist(),
        "observation.state/std": states.std(axis=0).tolist(),
        "observation.state/min": states.min(axis=0).tolist(),
        "observation.state/max": states.max(axis=0).tolist(),
        "action/mean": actions.mean(axis=0).tolist(),
        "action/std": actions.std(axis=0).tolist(),
        "action/min": actions.min(axis=0).tolist(),
        "action/max": actions.max(axis=0).tolist(),
    }
    json.dump(stats, (meta_dir / "stats.json").open("w"), indent=2)
    info = {"codebase_version": "v2.0", "fps": FPS, "video": True}
    json.dump(info, (meta_dir / "info.json").open("w"), indent=2)
    print("  ✅ done")
    print()

    # ── 汇总 ──
    print(f"✅ 数据集就绪: {NEW_PATH.resolve()}")
    print(f"  {len(episodes)} episodes x {TASK_TEXT}")
    print(f"  {len(rows)} 帧, {FPS} FPS")
    print(f"  步骤 A 完成，进入阶段 4")


if __name__ == "__main__":
    main()
