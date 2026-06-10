import os
import sys
import time
import asyncio
import websockets
import json
import math

print("脚本开始执行...")

# 1. 动态添加官方 SDK 的路径
sdk_path = os.environ.get(
    "GENKIARM_SDK_PATH",
    r"D:\arm_robot_begin\机械臂资料00\FTServo_Python-main\FTServo_Python-main"
)
if sdk_path not in sys.path:
    sys.path.append(sdk_path)

try:
    from scservo_sdk import *  # 导入官方库
    print("SDK 导入成功")
except ImportError as e:
    print(f"错误：无法在路径 {sdk_path} 中找到 scservo_sdk。")
    print(f"错误详情: {e}")
    sys.exit(1)

# 2. 配置参数
DEVICE_NAME = 'COM25'      # 舵机连接的端口
BAUDRATE    = 1000000      # STS3215 默认波特率 1M
SERVO_IDS   = [1, 2, 3, 4, 5, 6]  # 六个舵机的 ID 列表

def get_angle_from_pos(servo_id, pos):
    """
    将舵机步数 (0-4095) 转换为浏览器模拟器需要的度数 (-90 到 90)
    """
    # 基础转换：2048 步对应 0 度
    angle = (pos - 2048) * 360.0 / 4096.0
    
    # 对 6 号舵机（爪子）做特殊偏移处理
    if servo_id == 6:
        # 6号舵机：去掉之前的90°偏移
        return angle
    
    # 1-5 号舵机：运动方向反了，所以取反
    return -angle

async def main_logic():
    print("main_logic 开始执行...")
    print("=" * 50)
    print("真实舵机角度读取服务器")
    print("=" * 50)
    print(f"尝试连接串口: {DEVICE_NAME}")
    print(f"波特率: {BAUDRATE}")
    print(f"舵机ID: {SERVO_IDS}")
    print()
    
    # 初始化串口
    portHandler = PortHandler(DEVICE_NAME)
    packetHandler = sms_sts(portHandler)

    if not portHandler.openPort():
        print(f"❌ 无法打开端口: {DEVICE_NAME}")
        print("请检查：")
        print(f"  1. 串口号是否正确（当前是 {DEVICE_NAME}）")
        print("  2. USB 线是否连接")
        print("  3. 设备管理器中查看实际的串口号")
        return
    
    print(f"✅ 串口 {DEVICE_NAME} 已打开")
    
    if not portHandler.setBaudRate(BAUDRATE):
        print(f"❌ 无法设置波特率: {BAUDRATE}")
        portHandler.closePort()
        return
    
    print(f"✅ 波特率已设置为 {BAUDRATE}")
    print()

    # 先逐个 ping 舵机，测试连接
    print("正在测试舵机连接...")
    for servo_id in SERVO_IDS:
        model_number, result, error = packetHandler.ping(servo_id)
        if result == COMM_SUCCESS:
            print(f"  ✅ 舵机 ID {servo_id}: 连接成功 (Model: {model_number})")
        else:
            print(f"  ❌ 舵机 ID {servo_id}: 连接失败 (Result: {result}, Error: {error})")
    print()

    print(f"串口 {DEVICE_NAME} 已就绪，准备发送数据至 WebSocket...")
    print("WebSocket 服务器地址: ws://localhost:8765")
    print()

    # 存储连接的客户端
    connected_clients = set()

    async def handler(websocket):
        print("🌐 浏览器已连接！开始实时同步舵机角度...")
        connected_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            connected_clients.remove(websocket)
            print("🌐 浏览器连接已断开")

    # 启动 WebSocket 服务器
    server = await websockets.serve(handler, "localhost", 8765)
    print("✅ WebSocket 服务器已启动")
    print()
    
    try:
        while True:
            if connected_clients:
                angles = []
                debug_info = []
                for servo_id in SERVO_IDS:
                    # 读取位置
                    pos, speed, res, err = packetHandler.ReadPosSpeed(servo_id)
                    if res == COMM_SUCCESS:
                        angle = get_angle_from_pos(servo_id, pos)
                        angles.append(angle)
                        debug_info.append(f"ID{servo_id}:{pos}→{angle:.1f}°")
                    else:
                        angles.append(0.0)
                        debug_info.append(f"ID{servo_id}:ERR")
                
                # 打印调试信息
                print(f"\r{' | '.join(debug_info)}", end="", flush=True)
                
                # 发送给所有连接的浏览器
                message = json.dumps(angles)
                # 复制集合以防在迭代时发生变化
                for client in list(connected_clients):
                    try:
                        await client.send(message)
                    except:
                        pass
            
            await asyncio.sleep(0.05)  # 20Hz 更新频率
            
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.close()
        portHandler.closePort()
        print("\n✅ 资源已释放")

if __name__ == "__main__":
    print("进入 __main__")
    try:
        print("调用 asyncio.run(main_logic())")
        asyncio.run(main_logic())
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")
