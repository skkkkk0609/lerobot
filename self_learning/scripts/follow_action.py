
import os
import sys
import time

# 1. 动态添加官方 SDK 的路径
# 优先使用环境变量 GENKIARM_SDK_PATH，未设置时回退到默认路径
sdk_path = os.environ.get(
    "GENKIARM_SDK_PATH",
    r"d:\arm_robot_begin\FTServo_Python-main\FTServo_Python-main"
)
if sdk_path not in sys.path:
    sys.path.append(sdk_path)

try:
    from scservo_sdk import *  # 导入官方库
except ImportError:
    print(f"错误：无法在路径 {sdk_path} 中找到 scservo_sdk。")
    print("请确认该路径下存在 scservo_sdk 文件夹。")
    sys.exit(1)

# 2. 配置参数
MASTER_DEVICE = 'COM21'  # 主端端口
SLAVE_DEVICE  = 'COM20'  # 从端端口
BAUDRATE      = 1000000  # STS3215 默认波特率 1M
SERVO_IDS     = [1, 2, 3, 4, 5, 6]  # 六个舵机的 ID 列表
MOVING_SPEED  = 2400     # 移动速度
MOVING_ACC    = 50       # 移动加速度

def main():
    # 初始化主端 PortHandler 和 PacketHandler
    master_port = PortHandler(MASTER_DEVICE)
    master_handler = sms_sts(master_port)

    # 初始化从端 PortHandler 和 PacketHandler
    slave_port = PortHandler(SLAVE_DEVICE)
    slave_handler = sms_sts(slave_port)

    # 打开主端端口
    if master_port.openPort():
        print(f"成功打开主端端口: {MASTER_DEVICE}")
    else:
        print(f"无法打开主端端口: {MASTER_DEVICE}")
        return

    # 设置主端波特率
    if master_port.setBaudRate(BAUDRATE):
        print(f"成功设置主端波特率为: {BAUDRATE}")
    else:
        print(f"无法设置主端波特率")
        master_port.closePort()
        return

    # 打开从端端口
    if slave_port.openPort():
        print(f"成功打开从端端口: {SLAVE_DEVICE}")
    else:
        print(f"无法打开从端端口: {SLAVE_DEVICE}")
        master_port.closePort()
        return

    # 设置从端波特率
    if slave_port.setBaudRate(BAUDRATE):
        print(f"成功设置从端波特率为: {BAUDRATE}")
    else:
        print(f"无法设置从端波特率")
        master_port.closePort()
        slave_port.closePort()
        return

    print(f"\n开始主从跟随模式 (Ctrl+C 停止)...")
    print(f"主端端口: {MASTER_DEVICE} | 从端端口: {SLAVE_DEVICE}")
    print("-" * 80)

    try:
        while True:
            display_line = ""
            for servo_id in SERVO_IDS:
                # 读取主端舵机位置
                pos, speed, res, err = master_handler.ReadPosSpeed(servo_id)

                if res == COMM_SUCCESS:
                    # 将主端位置写入从端舵机
                    write_res, write_err = slave_handler.WritePosEx(servo_id, pos, MOVING_SPEED, MOVING_ACC)
                    
                    # 计算角度用于显示
                    angle = (pos * 360.0) / 4096.0
                    status = "OK" if write_res == COMM_SUCCESS else "ERR"
                    display_line += f"ID{servo_id}:{angle:6.1f}° [{status}] | "
                else:
                    display_line += f"ID{servo_id}: READ_ERR | "
            
            # 打印当前状态
            print(f"\r{display_line}", end="", flush=True)
            
            time.sleep(0.03)  # 适当延时

    except KeyboardInterrupt:
        print("\n\n用户停止跟随模式。")
    finally:
        # 关闭端口
        master_port.closePort()
        slave_port.closePort()
        print("串口已关闭。")

if __name__ == "__main__":
    main()

