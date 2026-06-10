import asyncio
import websockets
import json
import math
import time

async def send_robot_angles():
    # 模拟器默认监听地址
    uri = "ws://localhost:8765"
    
    # 注意：这个脚本是作为一个 WebSocket 服务器运行的，因为浏览器通常作为客户端连接
    # 这样你可以在本地运行这个 Python 脚本，浏览器会自动连接上来
    print(f"WebSocket server starting on ws://localhost:8765")
    print("Sending dynamic test angles to simulator...")
    
    async def handler(websocket):
        print("Browser connected!")
        try:
            while True:
                t = time.time()
                # 发送动态变化的角度，让机械臂动起来
                angles = [
                    45 * math.sin(t * 0.5),    # 1号舵机 - 腰部旋转
                    30 * math.cos(t * 0.3),    # 2号舵机 - 大臂
                    20 * math.sin(t * 0.7),    # 3号舵机 - 小臂
                    15 * math.cos(t * 0.4),    # 4号舵机 - 腕部
                    30 * math.sin(t * 0.6),    # 5号舵机 - 腕部旋转
                    45 + 45 * math.sin(t * 0.8) # 6号舵机 - 爪子 (0-90度)
                ]
                
                # 打印发送的数据，方便调试
                print(f"Sending angles: [{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}, {angles[3]:.1f}, {angles[4]:.1f}, {angles[5]:.1f}]", end="\r")
                
                # 发送 JSON 数据
                await websocket.send(json.dumps(angles))
                
                await asyncio.sleep(0.1) # 降低更新频率，每秒10次即可
                
        except websockets.exceptions.ConnectionClosed:
            print("\nBrowser disconnected")

    # 启动服务器
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.get_running_loop().create_future() # 保持运行

if __name__ == "__main__":
    try:
        asyncio.run(send_robot_angles())
    except KeyboardInterrupt:
        print("\nServer stopped")
