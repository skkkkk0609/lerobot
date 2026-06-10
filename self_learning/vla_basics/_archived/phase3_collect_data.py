"""阶段 3：采集 VLA 训练数据（遥操作 + 语言标注）

核心思路:
  - 复用你现有的遥操作（手柄/主臂控制）
  - 每次操作前输入一句中文任务描述
  - 同时记录: 摄像头画面 + 关节角度 + 目标角度 + 任务文本

数据保存格式:
  data/
    task_抓红色方块_ep01/
      frames/      ← 每帧的图片 (000001.jpg, 000002.jpg, ...)
      states.npy   ← 每帧的关节角度  (N, 6)
      actions.npy  ← 每帧的目标角度  (N, 6)
      meta.json    ← {"task": "抓住红色方块", "fps": 30, "joint_names": [...]}

使用方法（伪代码，需要根据你的实际遥操作代码调整）:

    from phase3_collect_data import EpisodeRecorder

    recorder = EpisodeRecorder("data")
    recorder.start("把红色方块放到蓝色碗里")

    while teleop_running:
        frame = get_camera_image()    # 你的摄像头读图函数
        state = get_joint_angles()    # 你的 WebSocket 读角度函数
        target = get_leader_angles()  # 遥操作主臂的目标角度
        recorder.record_frame(frame, state, target)

    recorder.save()
"""
import json
import time
from pathlib import Path

import numpy as np


class EpisodeRecorder:
    """采集单次遥操作的所有帧数据"""

    def __init__(self, save_dir: str = "data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task = ""
        self.joint_names: list[str] = []
        self.fps = 30
        self.frames: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.episode_counter: dict[str, int] = {}
        self._start_time: float = 0

    def start(self, task: str, joint_names: list[str] | None = None, fps: int = 30):
        self.task = task
        self.fps = fps
        self.joint_names = joint_names or [f"joint_{i}" for i in range(6)]
        self.frames.clear()
        self.states.clear()
        self.actions.clear()
        self._start_time = time.perf_counter()
        print(f"开始录制 | 任务: {task} | FPS: {fps}")

    def record_frame(self, image: np.ndarray, state: np.ndarray, action: np.ndarray):
        """每帧调用一次，记录画面、当前状态、目标动作"""
        self.frames.append(image.copy())
        self.states.append(np.array(state).flatten())
        self.actions.append(np.array(action).flatten())

    def save(self) -> str:
        """保存到 data/task_xxx_epNN/ 目录"""
        task_slug = self.task[:12].replace(" ", "_").replace("/", "_")
        self.episode_counter.setdefault(task_slug, 0)
        self.episode_counter[task_slug] += 1
        ep_id = self.episode_counter[task_slug]

        ep_dir = self.save_dir / f"{task_slug}_ep{ep_id:02d}"
        frames_dir = ep_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(self.frames):
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(frames_dir / f"{i+1:06d}.jpg", quality=85)

        np.save(ep_dir / "states.npy", np.array(self.states))
        np.save(ep_dir / "actions.npy", np.array(self.actions))

        meta = {
            "task": self.task,
            "fps": self.fps,
            "joint_names": self.joint_names,
            "num_frames": len(self.frames),
            "duration_sec": round(time.perf_counter() - self._start_time, 2),
        }
        json.dump(meta, (ep_dir / "meta.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

        print(f"已保存: {ep_dir} | {len(self.frames)} 帧 | {meta['duration_sec']} 秒")
        return str(ep_dir)


if __name__ == "__main__":
    print(
        "阶段 3 使用说明:\n"
        "  这个文件是数据采集工具类，不在命令行直接运行。\n"
        "  把它 import 到你的遥操作主循环中。\n"
        "\n"
        "示例伪代码:\n"
        "  from phase3_collect_data import EpisodeRecorder\n"
        "  recorder = EpisodeRecorder('data')\n"
        "  recorder.start('把红色方块放到蓝色碗里')\n"
        "  while teleop_running:\n"
        "      frame = read_camera()\n"
        "      state = read_robot_state()\n"
        "      target = read_leader_state()\n"
        "      recorder.record_frame(frame, state, target)\n"
        "  recorder.save()\n"
    )
