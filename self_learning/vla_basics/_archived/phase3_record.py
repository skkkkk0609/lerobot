"""阶段 3：采集 VLA 训练数据（真实机械臂 + 摄像头 + 语言标注）

=== 基于你的 genkiarm 项目 ===
  项目路径: D:\arm_robot_begin\genkiarm
  主臂: COM5 (leader), 从臂: COM25 (follower) — 如不对请改下面的配置
  摄像头: index 1 = 手机, index 0 = 笔记本
  舵机: STS3215 x6, 波特率 1M

=== 运行前 ===
  1. 机械臂通电，两个 COM 口都连上
  2. 安装依赖: pip install opencv-python
  3. 主臂和从臂都处于自由状态（舵机力矩打开）

=== 运行 ===
  conda activate python312
  cd d:\lerobot
  python self_learning\vla_basics\phase3_record.py

=== 每次录制流程 ===
  1. 输入中文任务指令
  2. 用主臂遥操作从臂完成任务
  3. 按 q 保存并结束本轮
  4. 物块归位，开始下一轮

=== 输出 ===
  data/vla_recordings/
    ├── index.json
    ├── 抓取红色方块_ep01/
    │   ├── camera_0/frames/  ← 笔记本摄像头
    │   ├── camera_1/frames/  ← 手机摄像头
    │   ├── states.npy        ← 从臂实时角度 (N,6)
    │   ├── actions.npy       ← 主臂角度 = 目标动作 (N,6)
    │   └── meta.json
    └── ...

=== 采集建议 ===
  3-5 个任务，每个 10-20 次演示，同一任务每次换起始位置
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ━━━ Windows OpenCV 修复 ━━━
import platform
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np

# ━━━ GenkiArm SDK ━━━
SDK_PATH = os.environ.get(
    "GENKIARM_SDK_PATH",
    r"D:\arm_robot_begin\机械臂资料00\FTServo_Python-main\FTServo_Python-main",
)
sys.path.insert(0, SDK_PATH)
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS, GroupSyncWrite

# ═══════════════════════════════════════════════
#  配置 —— 按你的实际情况修改这里
# ═══════════════════════════════════════════════
LEADER_PORT = "COM5"     # 主臂串口号（so100.yaml 里的 leader_arms.main.port）
FOLLOWER_PORT = "COM25"  # 从臂串口号（so100.yaml 里的 follower_arms.main.port）
BAUDRATE = 1000000
SERVO_IDS = [1, 2, 3, 4, 5, 6]
FPS = 30
CAMERA_INDICES = [0, 1]  # 0=笔记本, 1=手机 — 只用一个改 [1]
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
OUTPUT_DIR = Path("data/vla_recordings")

# ═══════════════════════════════════════════════
#  舵机工具
# ═══════════════════════════════════════════════
def pos_to_angle(servo_id, pos):
    """舵机步数 (0-4095) → 角度 (°)"""
    angle = (pos - 2048) * 360.0 / 4096.0
    if servo_id == 6:
        return angle
    return -angle

def angle_to_pos(servo_id, angle):
    """角度 (°) → 舵机步数"""
    raw = angle if servo_id == 6 else -angle
    return max(0, min(4095, int(raw * 4096.0 / 360.0 + 2048)))

def read_angles(ph, pkt):
    """读取所有舵机角度"""
    angles = []
    for sid in SERVO_IDS:
        pos, speed, res, err = pkt.ReadPosSpeed(sid)
        angles.append(pos_to_angle(sid, pos) if res == COMM_SUCCESS else 0.0)
    return angles

def write_angles(ph, pkt, gw, targets):
    """同步写入所有舵机"""
    for i, sid in enumerate(SERVO_IDS):
        gw.addParam(sid, angle_to_pos(sid, targets[i]), 0, 0)
    gw.txPacket()

# ═══════════════════════════════════════════════
#  录制主循环
# ═══════════════════════════════════════════════
class VLARecorder:
    def __init__(self):
        self.ep_counter = {}

    def record(self, task, leader_ph, leader_pkt, follower_ph, follower_pkt, f_gw, caps):
        task_slug = task[:16].replace(" ", "_").replace("/", "_")
        self.ep_counter.setdefault(task_slug, 0)
        self.ep_counter[task_slug] += 1
        ep_name = f"{task_slug}_ep{self.ep_counter[task_slug]:02d}"
        ep_dir = OUTPUT_DIR / ep_name

        # 创建目录
        for ci in range(len(caps)):
            (ep_dir / f"camera_{ci}" / "frames").mkdir(parents=True, exist_ok=True)

        all_frames = [[] for _ in caps]  # 每个摄像头一列表
        states = []
        actions = []

        print(f"\n{'='*60}")
        print(f"  录制: {ep_name}")
        print(f"  任务: {task}")
        print(f"  按 q 结束")
        print(f"{'='*60}")

        start_t = time.perf_counter()
        interval = 1.0 / FPS
        frame_idx = 0

        try:
            while True:
                t0 = time.perf_counter()

                # 1) 读从臂状态
                state = read_angles(follower_ph, follower_pkt)

                # 2) 读主臂 → 目标动作
                action = read_angles(leader_ph, leader_pkt)

                # 3) 写从臂跟踪主臂
                write_angles(follower_ph, follower_pkt, f_gw, action)

                # 4) 读摄像头
                for ci, cap in enumerate(caps):
                    ret, frame = cap.read()
                    if ret:
                        all_frames[ci].append(frame.copy())
                    else:
                        all_frames[ci].append(np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8))

                states.append(np.array(state, dtype=np.float32))
                actions.append(np.array(action, dtype=np.float32))
                frame_idx += 1

                # 5) 显示预览
                if len(caps) > 0 and len(all_frames[0]) > 0:
                    cv2.imshow("VLA Record - Camera 0", all_frames[0][-1])
                if len(caps) > 1 and len(all_frames[1]) > 0:
                    cv2.imshow("VLA Record - Camera 1", all_frames[1][-1])

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  已停止")
                    break

                # 帧率控制
                elapsed = time.perf_counter() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        except KeyboardInterrupt:
            print("\n  已中断")

        if frame_idx == 0:
            print("  ⚠ 无数据，跳过")
            return

        # ── 保存 ──
        print(f"  保存 {frame_idx} 帧...")
        for ci in range(len(caps)):
            frames_dir = ep_dir / f"camera_{ci}" / "frames"
            for i, frm in enumerate(all_frames[ci]):
                cv2.imwrite(str(frames_dir / f"{i+1:06d}.jpg"), frm)

        np.save(str(ep_dir / "states.npy"), np.array(states))
        np.save(str(ep_dir / "actions.npy"), np.array(actions))

        duration = round(time.perf_counter() - start_t, 2)
        meta = {
            "task": task,
            "fps": FPS,
            "num_frames": frame_idx,
            "duration_sec": duration,
            "servo_ids": SERVO_IDS,
            "image_size": [CAMERA_WIDTH, CAMERA_HEIGHT],
            "camera_indices": CAMERA_INDICES,
            "recorded_at": datetime.now().isoformat(),
        }
        json.dump(meta, (ep_dir / "meta.json").open("w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  ✅ {ep_name} | {frame_idx}帧 | {duration:.1f}s | ~{frame_idx/duration:.1f} FPS\n")

    def save_index(self):
        eps = []
        for d in sorted(OUTPUT_DIR.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                eps.append(json.load((d / "meta.json").open("r", encoding="utf-8")) | {"dir": d.name})
        json.dump({"episodes": eps, "total": len(eps)},
                  (OUTPUT_DIR / "index.json").open("w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  索引: {OUTPUT_DIR / 'index.json'} ({len(eps)} episodes)")


# ═══════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  VLA 数据采集器 - 阶段 3")
    print("=" * 60)
    print(f"  主臂: {LEADER_PORT}  |  从臂: {FOLLOWER_PORT}")
    print(f"  摄像头: {CAMERA_INDICES}")
    print()

    # ── 摄像头 ──
    caps = []
    for ci in CAMERA_INDICES:
        cap = cv2.VideoCapture(ci, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        if cap.isOpened():
            caps.append(cap)
            print(f"  ✅ 摄像头 {ci} 已连接")
        else:
            print(f"  ⚠ 摄像头 {ci} 无法打开，跳过")
            cap.release()

    if not caps:
        print("\n❌ 没有可用摄像头！请检查 USB 连接和 CAMERA_INDICES 配置")
        return

    # ── 机械臂 ──
    print(f"\n  连接主臂 {LEADER_PORT}...")
    l_ph = PortHandler(LEADER_PORT)
    l_pkt = sms_sts(l_ph)
    if not l_ph.openPort():
        print(f"  ❌ 无法打开 {LEADER_PORT}")
        for c in caps: c.release()
        return
    l_ph.setBaudRate(BAUDRATE)
    print(f"  ✅ 主臂已连接")

    print(f"  连接从臂 {FOLLOWER_PORT}...")
    f_ph = PortHandler(FOLLOWER_PORT)
    f_pkt = sms_sts(f_ph)
    if not f_ph.openPort():
        print(f"  ❌ 无法打开 {FOLLOWER_PORT}")
        l_ph.closePort()
        for c in caps: c.release()
        return
    f_ph.setBaudRate(BAUDRATE)
    print(f"  ✅ 从臂已连接")
    f_gw = GroupSyncWrite(f_ph, f_pkt, 41, 7)

    # ── 主循环 ──
    recorder = VLARecorder()
    print(f"\n{'='*60}")
    print("  开始录制！输入中文任务指令（q 退出）")
    print(f"{'='*60}")

    try:
        while True:
            task = input("\n任务指令: ").strip()
            if task.lower() == 'q':
                break
            if not task:
                continue
            recorder.record(task, l_ph, l_pkt, f_ph, f_pkt, f_gw, caps)
            recorder.save_index()
    except KeyboardInterrupt:
        print("\n\n  中断")
    finally:
        # 归零
        print("  机械臂归零...")
        try:
            write_angles(f_ph, f_pkt, f_gw, [0, 0, 0, 0, 0, 0])
        except:
            pass
        time.sleep(1)

        cv2.destroyAllWindows()
        for c in caps: c.release()
        l_ph.closePort()
        f_ph.closePort()
        recorder.save_index()
        print("  ✅ 已释放\n")

    total = sum(recorder.ep_counter.values())
    print(f"  本次共录制 {total} 个 episode")


if __name__ == "__main__":
    main()
