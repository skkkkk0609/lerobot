"""阶段 2 核心文件：WebSocket 机械臂适配器（适配黑马 GenkiPi 模拟器）

架构:
  ┌─────────┐   WebSocket 服务器    ┌──────────────┐
  │ Python   │ ───[角度数组]──────→ │ 浏览器模拟器   │
  │ VLA 模型  │  ws://localhost:8765 │ (URDF 机械臂) │
  └─────────┘                       └──────────────┘

协议:
  - 浏览器连接 Python 的 WebSocket 服务器（端口 8765）
  - Python 发送 JSON 数组 [角度0, 角度1, ..., 角度5]
  - 角度单位: 度 (°), 范围大致 -90 到 90
  - 关节 1-5: 机械臂, 关节 6: 夹爪

使用方式:
  from websocket_robot import GenkiPiSimRobot
  robot = GenkiPiSimRobot()
  robot.start()        # 启动服务器，等待浏览器连接
  robot.send_angles([0, 30, -45, 20, 15, 60])  # 发送角度
  robot.stop()
"""
import asyncio
import json
import threading
import time
from typing import Optional

import websockets

NUM_JOINTS = 6


class GenkiPiSimRobot:
    """黑马 GenkiPi 模拟器适配器"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._server: Optional[websockets.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: set = set()
        self._running = False
        self._current_angles = [0.0] * NUM_JOINTS

    # ── 启动/停止 ──

    def start(self):
        """阻塞式启动（在线程中运行 asyncio 事件循环）"""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_async_server, daemon=True
        )
        self._thread.start()
        time.sleep(0.5)  # 等一下服务器起来
        print(f"WebSocket 服务器已启动: ws://{self.host}:{self.port}")
        print("请在浏览器中打开 http://localhost:5173 连接模拟器")

    def stop(self):
        self._running = False
        if self._loop and self._server:
            self._loop.call_soon_threadsafe(self._shutdown)
        if self._thread:
            self._thread.join(timeout=3)

    def _shutdown(self):
        """线程安全地关闭 WebSocket 服务器，让协程自然退出"""
        self._close_server()

    def _close_server(self):
        """同步关闭服务器（在线程内调用）"""
        async def _close():
            if self._server:
                self._server.close()
                await self._server.wait_closed()
        asyncio.ensure_future(_close())

    def _run_async_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._loop.close()

    async def _start_server(self):
        async def handler(websocket):
            self._clients.add(websocket)
            print("浏览器已连接！")
            try:
                await websocket.wait_closed()
            finally:
                self._clients.discard(websocket)

        self._server = await websockets.serve(handler, self.host, self.port)
        try:
            # 等待服务器关闭（通过 _close_server 触发）
            await self._server.wait_closed()
        except Exception:
            pass

    # ── 核心方法 ──

    @property
    def is_connected(self) -> bool:
        return len(self._clients) > 0

    def send_angles(self, angles: list[float]):
        """发送 6 个关节角度到浏览器模拟器（单位：度）"""
        self._current_angles = list(angles)
        data = json.dumps(angles)
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(data), self._loop
            )

    async def _broadcast(self, data: str):
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception:
                self._clients.discard(ws)

    def get_current_angles(self) -> list[float]:
        """获取当前发送的最后一次角度"""
        return self._current_angles.copy()

    @staticmethod
    def scale_action_to_angles(
        action_values: list[float], default_range: float = 90.0
    ) -> list[float]:
        """把 VLA 模型输出（-1 到 1 范围）映射到角度（-90° 到 90°）

        模型输出的值是归一化到 [-1, 1] 的，需要映射到实际角度范围。
        """
        return [v * default_range for v in action_values]


# ── 兼容旧接口 ──

JOINT_NAMES = [f"joint_{i}" for i in range(NUM_JOINTS)]


def send_random_motion(robot: GenkiPiSimRobot, steps: int = 20):
    """发送随机动作测试通信链路"""
    print(f"\n发送 {steps} 步随机动作，测试通信链路...")
    print("（浏览器里的机械臂应该会随机抖动）\n")

    base_angles = [0, 30, -45, 0, 0, 60]
    for i in range(steps):
        import random

        angles = [a + random.uniform(-5, 5) for a in base_angles]
        robot.send_angles(angles)
        time.sleep(0.1)
        print(f"  步 {i+1}: {[round(a, 1) for a in angles[:3]]}...", end="\r")

    print("\n通信链路测试完成！")
    robot.send_angles([0, 0, 0, 0, 0, 0])  # 归零


if __name__ == "__main__":
    robot = GenkiPiSimRobot()
    robot.start()
    try:
        send_random_motion(robot, steps=10)
        input("\n按回车停止...")
    finally:
        robot.stop()
