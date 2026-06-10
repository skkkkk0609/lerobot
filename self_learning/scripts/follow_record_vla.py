"""
基于 genkiarm 的 control_robot.py 改编的 VLA 数据录制脚本。
使用 LeRobot 官方 Robot 类（leader + follower），自动完成主从跟随 + 数据录制。
数据格式与 LeRobot 官方一致，可直接用于云端 SmolVLA 训练。

用法:
    python follow_record_vla.py
"""

import os
import platform

if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys
import time
import threading
from pathlib import Path

# 添加 genkiarm lerobot 路径
sys.path.insert(0, r"d:\arm_robot_begin\genkiarm")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.populate_dataset import (
    create_lerobot_dataset,
    delete_current_episode,
    init_dataset,
    save_current_episode,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    record_episode,
    reset_environment,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_logging


def record_vla_data():
    """
    录制 VLA 数据集。
    使用 genkiarm 的 so100 配置，自动处理主从跟随 + 数据保存。
    """
    init_logging()

    # 数据集配置
    repo_id = "pick/smola_vla_test"
    root = r"d:\lerobot\data"
    fps = 30
    num_episodes = 50
    episode_time_s = 30
    reset_time_s = 10
    warmup_time_s = 2

    # 创建数据集
    dataset = init_dataset(
        repo_id,
        root,
        force_override=False,
        fps=fps,
        video=True,
        write_images=True,
        num_image_writer_processes=0,
        num_image_writer_threads=4,
    )

    # 机器人配置（使用 genkiarm 的 so100 配置）
    robot_cfg_path = r"d:\arm_robot_begin\genkiarm\lerobot\configs\robot\so100.yaml"

    # 覆盖端口配置
    robot_overrides = [
        "follower_arms.main.port=COM6",
        "leader_arms.main.port=COM7",
    ]

    # 如果配置路径找不到，手动构建配置
    try:
        from lerobot.common.utils.utils import init_hydra_config
        robot_cfg = init_hydra_config(robot_cfg_path, robot_overrides)
        robot = make_robot(robot_cfg)
    except Exception as e:
        print(f"无法加载 genkiarm 配置: {e}")
        print("请确保 genkiarm 项目路径正确")
        return

    # 连接机器人
    if not robot.is_connected:
        robot.connect()

    # 初始化键盘监听
    listener, events = init_keyboard_listener()

    # 热身
    print("Warmup 2 秒...")
    warmup_record(
        robot=robot,
        events=events,
        enable_teleoperation=True,
        warmup_time_s=warmup_time_s,
        display_cameras=False,
        fps=fps,
    )

    # 录制循环
    while dataset["num_episodes"] < num_episodes:
        episode_index = dataset["num_episodes"]
        print(f"\n录制 episode {episode_index}...")

        record_episode(
            dataset=dataset,
            robot=robot,
            events=events,
            episode_time_s=episode_time_s,
            display_cameras=False,
            policy=None,
            device=None,
            use_amp=None,
            fps=fps,
        )

        # 重置环境
        if not events["stop_recording"] and (
            (episode_index < num_episodes - 1) or events["rerecord_episode"]
        ):
            print(f"重置环境 {reset_time_s} 秒...")
            reset_environment(robot, events, reset_time_s)

        # 重录逻辑
        if events["rerecord_episode"]:
            print("重录 episode...")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            delete_current_episode(dataset)
            continue

        # 保存 episode
        save_current_episode(dataset)

        if events["stop_recording"]:
            break

    # 完成
    print("停止录制...")
    stop_recording(robot, listener, display_cameras=False)

    # 创建 LeRobotDataset 并推送
    lerobot_dataset = create_lerobot_dataset(
        dataset,
        run_compute_stats=False,
        push_to_hub=False,
        tags=None,
        play_sounds=True,
    )

    print(f"\n录制完成! 共 {lerobot_dataset.num_episodes} 个 episodes")
    print(f"数据目录: {root}/{repo_id}")


if __name__ == "__main__":
    record_vla_data()
